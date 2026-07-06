"""Remote robot environment for Stage 2 online RL training.

Provides ``RemoteWebSocketEnv``, a WebSocket server that bridges a
remote Robot PC (running robodeploy + openpi_client) with the
``OnlineRLTrainer`` environment interface.

The Robot PC connects as a WebSocket client and streams observations.
The Training PC (this side) drives episodes through ``reset()`` and
``step()`` — actions are sent back over the same WebSocket.

Usage (env_factory for Stage 2)::

    python scripts/train_online_rl.py \\
        --env-factory rlt.rollout.remote_env.make_remote_env \\
        --vla-checkpoint-dir ... --rl-token-checkpoint ...
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

import numpy as np
import websockets
import websockets.asyncio.server as _ws_server

from openpi_client import msgpack_numpy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

DEFAULT_RL_PORT = 5556
CONNECT_TIMEOUT = 120.0  # 等待 Robot PC 连接
STEP_TIMEOUT = 300.0  # 等待下一次观测（reset 后需等待操作员按 Enter）


class RemoteWebSocketEnv:
    """Chunk-level environment backed by a remote Robot PC.

    Implements the same interface as :class:`~rlt.rollout.robot_env.RobotEnv`
    (``reset``, ``step``, ``action_dim``, ``chunk_length``) so it works
    with ``RolloutWorker`` and ``OnlineRLTrainer``.

    Args:
        host: Bind address for the WebSocket server.
        port: Port for the WebSocket server.
        action_dim: Single-step action dimension.
        chunk_length: C, number of actions per chunk.
        max_episode_chunks: Max chunks per episode (used as ``info`` hint).
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = DEFAULT_RL_PORT,
        action_dim: int = 14,
        chunk_length: int = 10,
        max_episode_chunks: int = 150,
    ) -> None:
        self._host = host
        self._port = port
        self._action_dim = action_dim
        self._chunk_length = chunk_length
        self._max_episode_chunks = max_episode_chunks

        # ---- synchronisation primitives ----
        self._obs_event = threading.Event()
        self._resp_event = threading.Event()
        self._connected = threading.Event()
        self._stop = threading.Event()

        self._latest_msg: dict[str, Any] | None = None
        self._response: dict[str, Any] = {}
        self._error: Exception | None = None
        self._metadata: dict[str, Any] = {
            "model_type": "rl_trainer",
            "action_dim": action_dim,
            "chunk_length": chunk_length,
        }

        # VLA reference action to send alongside actor actions (set before each step)
        self._vla_action_chunk: np.ndarray | None = None

        # ---- start server in background thread ----
        self._server_thread = threading.Thread(
            target=self._run_server, name="rl-ws-server", daemon=True
        )
        self._server_thread.start()

        # Wait for Robot PC to connect
        if not self._connected.wait(timeout=CONNECT_TIMEOUT):
            self._stop.set()
            raise RuntimeError(
                f"Timed out waiting for Robot PC to connect on port {self._port} "
                f"(waited {CONNECT_TIMEOUT}s)"
            )
        logger.info("Robot PC connected on port %d", self._port)

    # ------------------------------------------------------------------
    # Public env interface
    # ------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def chunk_length(self) -> int:
        return self._chunk_length

    def reset(self, **kwargs: Any) -> dict[str, Any]:
        """Request a robot reset and return the first observation.

        Returns:
            Observation dict with ``"state"``, ``"images"``, and ``"prompt"``.
        """
        self._check_error()

        # If there is a pending observation (e.g. the Robot PC sent one
        # while we were between episodes), discard it — we want a fresh
        # post-reset observation.
        if self._obs_event.is_set():
            self._obs_event.clear()

        # Send reset signal immediately.  After warmup (or between
        # episodes) the handler is blocked in "wait_resp" and the Robot
        # PC is blocked in recv() waiting for a response.  Sending reset
        # first breaks this three-way deadlock.
        self._set_response({"actions": np.zeros((self._chunk_length, self._action_dim), dtype=np.float32), "reset": True})
        self._resp_event.set()

        # Wait for post-reset observation (human operator may need time to press Enter)
        if not self._obs_event.wait(timeout=STEP_TIMEOUT):
            raise RuntimeError(f"Timed out waiting for Robot PC to reset (waited {STEP_TIMEOUT}s)")
        self._obs_event.clear()

        return self._extract_obs()

    def set_vla_action(self, vla_action: np.ndarray) -> None:
        """Set the VLA reference action to send alongside actor actions.

        Call before ``step()`` so the Robot PC can choose which to execute
        based on the RL toggle.
        """
        self._vla_action_chunk = np.asarray(vla_action, dtype=np.float32)

    def step(
        self, action_chunk: np.ndarray
    ) -> tuple[dict[str, Any], np.ndarray, bool, dict[str, Any]]:
        """Send an action chunk and collect the result.

        Args:
            action_chunk: Actions ``[C, action_dim]``.

        Returns:
            ``(next_obs, rewards, done, info)``.
            干预时 ``info["intervention"]`` = True，``info["action"]`` 为单帧 ``[d]``，
            ``info["steps_executed"]`` = 1，rewards[0] 携带奖励信号。
        """
        self._check_error()

        # Send action (actor + VLA reference for RL toggle)
        actions = np.asarray(action_chunk, dtype=np.float32)
        response: dict[str, Any] = {"actions": actions, "reset": False}
        if self._vla_action_chunk is not None:
            response["vla_actions"] = self._vla_action_chunk
        self._set_response(response)
        self._resp_event.set()
        self._vla_action_chunk = None  # 已发送，清空防止下次误用

        # Wait for next observation (includes reward / done)
        if not self._obs_event.wait(timeout=STEP_TIMEOUT):
            raise RuntimeError("Timed out waiting for Robot PC response")
        self._obs_event.clear()

        msg = self._latest_msg
        assert msg is not None

        obs = self._extract_obs()
        reward_val = float(msg.get("reward", 0.0))
        done = bool(msg.get("done", False))
        info: dict[str, Any] = {
            "success": bool(msg.get("success", False)),
            "rl_active": bool(msg.get("rl_active", True)),  # default True = backward compat
        }

        # 单帧干预：reward 仅在位置 0，steps_executed=1
        if msg.get("intervention", False) and "action" in msg:
            info["intervention"] = True
            info["action"] = np.asarray(msg["action"], dtype=np.float32)  # [d]
            info["steps_executed"] = 1
            rewards = np.zeros(self._chunk_length, dtype=np.float32)
            rewards[0] = reward_val
        else:
            # 正常 chunk：C 帧，reward 均摊（兼容旧协议，后续逐步废弃）
            rewards = np.full(self._chunk_length, reward_val / max(self._chunk_length, 1), dtype=np.float32)

        return obs, rewards, done, info

    def close(self) -> None:
        """Shut down the WebSocket server."""
        self._stop.set()
        self._obs_event.set()  # unblock any waiters
        self._resp_event.set()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_obs(self) -> dict[str, Any]:
        msg = self._latest_msg
        assert msg is not None
        return {
            "state": np.asarray(msg.get("state", []), dtype=np.float32),
            "images": {k: np.asarray(v) for k, v in msg.get("images", {}).items()},
            "prompt": msg.get("prompt", ""),
        }

    def _set_response(self, resp: dict[str, Any]) -> None:
        self._response = resp

    def _check_error(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"WebSocket server error: {self._error}") from self._error

    # ------------------------------------------------------------------
    # Asyncio server (background thread)
    # ------------------------------------------------------------------

    def _run_server(self) -> None:
        """Entry point for the background thread."""
        try:
            asyncio.run(self._serve())
        except Exception as exc:
            self._error = exc
            self._obs_event.set()
            self._resp_event.set()
            logger.exception("WebSocket server crashed")

    async def _serve(self) -> None:
        """Start the asyncio WebSocket server."""
        async with _ws_server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ):
            # Keep running until stopped
            while not self._stop.is_set():
                await asyncio.sleep(0.5)

    async def _handler(self, websocket: _ws_server.ServerConnection) -> None:
        """Handle one Robot PC connection."""
        remote = websocket.remote_address
        logger.info("Robot PC connected from %s", remote)
        self._connected.set()
        packer = msgpack_numpy.Packer()

        # Send metadata first (matching websocket_policy_server protocol)
        await websocket.send(packer.pack(self._metadata))

        try:
            while not self._stop.is_set():
                # ---- check for pending response FIRST ----
                # If the main thread already set a response (e.g.
                # reset() after warmup when both sides are waiting),
                # skip recv and send it directly.  This prevents the
                # three-way deadlock where handler waits for _resp_event,
                # Robot PC waits for a response, and reset() waits for
                # an observation.
                if not self._resp_event.is_set():
                    # ---- receive observation ----
                    raw = await asyncio.wait_for(websocket.recv(), timeout=STEP_TIMEOUT)
                    msg = msgpack_numpy.unpackb(raw)
                    # Unwrap {"method": "infer", "obs": {...}} envelope from
                    # WebsocketClientPolicy.infer() — the actual observation
                    # (state, images, prompt, reward, done, success) is under "obs".
                    if isinstance(msg, dict) and "obs" in msg:
                        msg = msg["obs"]
                    self._latest_msg = msg
                    self._obs_event.set()

                # ---- wait for main-thread response ----
                while not self._resp_event.is_set() and not self._stop.is_set():
                    await asyncio.sleep(0.05)
                if self._stop.is_set():
                    break
                self._resp_event.clear()

                # ---- send response ----
                await websocket.send(packer.pack(self._response))

        except websockets.ConnectionClosed:
            logger.info("Robot PC %s disconnected", remote)
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for observation from %s", remote)
        except Exception:
            logger.exception("Error in WebSocket handler for %s", remote)
        finally:
            self._connected.clear()
            self._obs_event.set()  # unblock main thread


# ---------------------------------------------------------------------------
# Factory function (for --env-factory CLI flag)
# ---------------------------------------------------------------------------


def make_remote_env(
    action_dim: int = 14,
    chunk_length: int = 10,
    task_prompt: str = "",
    max_episode_chunks: int = 150,
    **kwargs: Any,
) -> RemoteWebSocketEnv:
    """Create a ``RemoteWebSocketEnv`` wired to a remote Robot PC.

    The Robot PC must run the ``rl_robot_bridge.py`` script which
    connects to this machine on the configured port.

    Args:
        action_dim: Single-step action dimension (bi_s1: 14).
        chunk_length: C, actions per chunk.
        task_prompt: Task instruction forwarded to the VLA.
        max_episode_chunks: Max chunks per episode.

    Returns:
        An env ready for ``OnlineRLTrainer``.
    """
    logger.info(
        "Creating RemoteWebSocketEnv: port=%d action_dim=%d chunk_length=%d prompt=%r",
        DEFAULT_RL_PORT,
        action_dim,
        chunk_length,
        task_prompt,
    )
    return RemoteWebSocketEnv(
        host="0.0.0.0",
        port=DEFAULT_RL_PORT,
        action_dim=action_dim,
        chunk_length=chunk_length,
        max_episode_chunks=max_episode_chunks,
    )
