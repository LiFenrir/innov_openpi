#!/usr/bin/env python3
"""从训练日志中提取指标并绘制训练曲线。

用法:
    python scripts/plot_training_metrics.py <train.log路径> [输出图片路径]

示例:
    python scripts/plot_training_metrics.py checkpoints/bi_s1_pi05_sft/bi_s1_sft_run/train.log
    python scripts/plot_training_metrics.py train.log training_metrics.png
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_log(log_path: str) -> dict[str, np.ndarray]:
    """从 train.log 中提取指标数组，兼容 SFT / Stage1 / Stage2 格式。"""
    with open(log_path) as f:
        lines = f.readlines()

    # Detect format by scanning for header line
    header = ""
    for line in lines:
        if line.startswith("# step"):
            header = line.strip()
            break

    # ── Stage 1 / joint format: fixed-width columns ──
    if "loss" in header and "l_ro" in header:
        has_l_vla = "l_vla" in header
        rows = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                step = int(parts[0])
            except ValueError:
                continue
            if has_l_vla and len(parts) >= 7:
                rows.append({
                    "step": step,
                    "loss": float(parts[1]),
                    "l_ro": float(parts[2]),
                    "l_vla": float(parts[3]),
                    "grad_norm": float(parts[4]),
                    "vla_grad_norm": float(parts[5]),
                    "lr": float(parts[6]),
                })
            elif not has_l_vla and len(parts) >= 4:
                rows.append({
                    "step": step,
                    "loss": float(parts[1]),
                    "l_ro": float(parts[1]),   # loss == l_ro in frozen mode
                    "grad_norm": float(parts[2]),
                    "lr": float(parts[3]),
                })

        if rows:
            steps = np.array([r["step"] for r in rows])
            losses = np.array([r["loss"] for r in rows])
            lrs = np.array([r["lr"] for r in rows])
            grad_norms = np.array([r["grad_norm"] for r in rows])
            times = np.zeros_like(steps)  # no timing in stage1 log
            result = {"step": steps, "loss": losses, "lr": lrs, "grad_norm": grad_norms, "time": times}
            if has_l_vla:
                result["l_ro"] = np.array([r["l_ro"] for r in rows])
                result["l_vla"] = np.array([r["l_vla"] for r in rows])
                result["vla_grad_norm"] = np.array([r["vla_grad_norm"] for r in rows])
            return result

    # ── SFT format: step=100 loss=0.4 lr=1e-4 ... ──
    content = "".join(lines)
    pattern = r"step=(\d+)\s+loss=([\d.]+)\s+lr=([\de.\-+]+)\s+grad_norm=([\d.]+)\s+time=([\d.]+)s"
    matches = re.findall(pattern, content)

    if not matches:
        print(f"错误: 日志中未匹配到 step 记录行，请确认格式。", file=sys.stderr)
        sys.exit(1)

    steps = np.array([int(m[0]) for m in matches])
    losses = np.array([float(m[1]) for m in matches])
    lrs = np.array([float(m[2]) for m in matches])
    grad_norms = np.array([float(m[3]) for m in matches])
    times = np.array([float(m[4]) for m in matches])

    return {"step": steps, "loss": losses, "lr": lrs, "grad_norm": grad_norms, "time": times}


def smooth(arr: np.ndarray, window: int = 5) -> np.ndarray:
    """简单滑动平均，返回与输入等长的数组（头部用截断窗口）。"""
    if len(arr) < window:
        return arr.copy()
    kernel = np.ones(window) / window
    smoothed = np.convolve(arr, kernel, mode="valid")
    # 首尾填充，保持长度一致
    pad_left = window // 2
    pad_right = window - pad_left - 1
    return np.concatenate([
        arr[:pad_left],
        smoothed,
        arr[-pad_right:] if pad_right > 0 else []
    ])


def print_summary(data: dict[str, np.ndarray]) -> None:
    """打印训练指标汇总到 stdout。"""
    steps = data["step"]
    losses = data["loss"]
    lrs = data["lr"]
    grad_norms = data["grad_norm"]
    times = data["time"]
    l_ro = data.get("l_ro")
    l_vla = data.get("l_vla")

    loss_min_idx = losses.argmin()
    total_time_s = times.sum() if times.any() else 0

    print("=" * 70)
    print("训练指标汇总")
    print("=" * 70)
    print(f"总训练步数:       {steps[-1]:,}")
    print(f"记录数据点数:     {len(steps)}")
    if total_time_s > 0:
        print(f"总耗时:           {total_time_s:.0f}s = {total_time_s / 3600:.1f}h")
    print(f"初始 loss:        {losses[0]:.4f}")
    print(f"最终 loss:        {losses[-1]:.4f}")
    print(f"最小 loss:        {losses.min():.4f} (step {steps[loss_min_idx]})")
    print(f"loss 下降幅度:    {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
    if l_ro is not None:
        print(f"初始 L_ro:        {l_ro[0]:.4f}")
        print(f"最终 L_ro:        {l_ro[-1]:.4f}")
        print(f"最小 L_ro:        {l_ro.min():.4f}")
    if l_vla is not None:
        print(f"最终 L_vla:       {l_vla[-1]:.4f}")
    print(f"峰值 LR:          {lrs.max():.2e}")
    print(f"最终 LR:          {lrs[-1]:.2e}")
    print(f"最大 grad norm:   {grad_norms.max():.2f} (step {steps[grad_norms.argmax()]})")
    print(f"最终 grad norm:   {grad_norms[-1]:.2f}")

    # 阶段分析
    warmup_end = 1000
    plateau_end = 2000
    warmup_mask = steps < warmup_end
    plateau_mask = (steps >= warmup_end) & (steps < plateau_end)
    decay_mask = steps >= plateau_end

    print(f"\n阶段分析:")
    if warmup_mask.any():
        print(f"  Warmup   (0-{warmup_end}):       avg loss={losses[warmup_mask].mean():.4f}, "
              f"avg grad_norm={grad_norms[warmup_mask].mean():.2f}")
    if plateau_mask.any():
        print(f"  Peak LR  ({warmup_end}-{plateau_end}):     avg loss={losses[plateau_mask].mean():.4f}, "
              f"avg grad_norm={grad_norms[plateau_mask].mean():.2f}")
    if decay_mask.any():
        print(f"  Decay    ({plateau_end}-{steps[-1]}):   avg loss={losses[decay_mask].mean():.4f}, "
              f"avg grad_norm={grad_norms[decay_mask].mean():.2f}")


def plot_metrics(data: dict[str, np.ndarray], output_path: str, smooth_window: int = 5) -> None:
    """绘制训练指标图并保存，兼容 SFT / Stage1 / Stage2 格式。"""
    steps = data["step"]
    losses = data["loss"]
    lrs = data["lr"]
    grad_norms = data["grad_norm"]
    times = data["time"]
    l_ro = data.get("l_ro")
    l_vla = data.get("l_vla")
    vla_grad_norm = data.get("vla_grad_norm")
    has_timing = times.any()
    is_stage1 = l_ro is not None

    if is_stage1 and l_vla is not None:
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        fig.suptitle("Stage 1 Joint Training Metrics", fontsize=16, fontweight="bold")
    elif is_stage1:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("Stage 1 Frozen Training Metrics", fontsize=16, fontweight="bold")
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("Training Metrics", fontsize=16, fontweight="bold")

    # Flatten axes for easy indexing
    ax_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    # --- Loss (L_ro for Stage1) ---
    ax = ax_flat[0]
    if is_stage1:
        ax.plot(steps, l_ro, linewidth=1.0, color="#1f77b4", alpha=0.5, label="L_ro (raw)")
        smoothed_l_ro = smooth(l_ro, smooth_window)
        ax.plot(steps, smoothed_l_ro, linewidth=1.8, color="#d62728", label=f"L_ro MA({smooth_window})")
        ax.set_ylabel("L_ro")
        ax.set_title("Reconstruction Loss (L_ro)")
    else:
        ax.plot(steps, losses, linewidth=1.0, color="#1f77b4", alpha=0.5, label="Raw")
        smoothed_loss = smooth(losses, smooth_window)
        ax.plot(steps, smoothed_loss, linewidth=1.8, color="#d62728", label=f"MA({smooth_window})")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
    ax.set_xlabel("Step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # --- L_vla (joint only) or LR ---
    ax = ax_flat[1]
    if l_vla is not None:
        ax.plot(steps, l_vla, linewidth=1.0, color="#1f77b4", alpha=0.5, label="L_vla (raw)")
        smoothed_l_vla = smooth(l_vla, smooth_window)
        ax.plot(steps, smoothed_l_vla, linewidth=1.8, color="#d62728", label=f"L_vla MA({smooth_window})")
        ax.set_ylabel("L_vla")
        ax.set_title("VLA Flow-Matching Loss (L_vla)")
        ax.legend()
    else:
        ax.plot(steps, lrs * 1e6, linewidth=1.5, color="#2ca02c")
        ax.set_ylabel("LR (×10⁻⁶)")
        ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)

    # --- Grad Norm ---
    ax = ax_flat[2] if len(ax_flat) > 2 else ax_flat[1]
    ax.plot(steps, grad_norms, linewidth=1.0, color="#ff7f0e", alpha=0.5, label="RL Token")
    smoothed_gn = smooth(grad_norms, smooth_window)
    ax.plot(steps, smoothed_gn, linewidth=1.8, color="#9467bd", label=f"RL Token MA({smooth_window})")
    if vla_grad_norm is not None:
        ax.plot(steps, vla_grad_norm, linewidth=1.0, color="#17becf", alpha=0.5, label="VLA")
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norm")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Time / LR ---
    ax = ax_flat[3] if len(ax_flat) > 3 else ax_flat[1]
    if has_timing:
        ax.plot(steps, times, linewidth=1.0, color="#8c564b", alpha=0.7)
        mean_t = np.mean(times)
        ax.axhline(y=mean_t, color="#e377c2", linestyle="--", linewidth=1.5, label=f"Mean: {mean_t:.1f}s")
        ax.set_ylabel("Time (s)")
        ax.set_title("Training Speed")
        ax.legend()
    else:
        ax.plot(steps, lrs * 1e6, linewidth=1.5, color="#2ca02c")
        ax.set_ylabel("LR (×10⁻⁶)")
        ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)

    # --- Combined loss (joint only, 5th plot) ---
    if l_vla is not None and len(ax_flat) > 4:
        ax = ax_flat[4]
        ax.plot(steps, losses, linewidth=1.0, color="#bcbd22", alpha=0.5, label="Total")
        smoothed_total = smooth(losses, smooth_window)
        ax.plot(steps, smoothed_total, linewidth=1.8, color="#d62728", label=f"Total MA({smooth_window})")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Total Loss (L_ro + α·L_vla)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    # -- LR (6th plot for joint) --
    if l_vla is not None and len(ax_flat) > 5:
        ax = ax_flat[5]
        ax.plot(steps, lrs * 1e6, linewidth=1.5, color="#2ca02c")
        ax.set_xlabel("Step")
        ax.set_ylabel("LR (×10⁻⁶)")
        ax.set_title("Learning Rate Schedule")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n图表已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="从 train.log 提取训练指标并绘制曲线")
    parser.add_argument("log", help="train.log 文件路径")
    parser.add_argument("-o", "--output", default=None, help="输出图片路径（默认与日志同目录的 training_metrics.png）")
    parser.add_argument("-w", "--smooth-window", type=int, default=5, help="滑动平均窗口大小 (默认: 5)")
    parser.add_argument("--no-plot", action="store_true", help="只打印汇总，不生成图片")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"错误: 文件不存在: {log_path}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_path = args.output
    else:
        output_path = str(log_path.parent / "training_metrics.png")

    data = parse_log(str(log_path))
    print_summary(data)

    if not args.no_plot:
        plot_metrics(data, output_path, args.smooth_window)


if __name__ == "__main__":
    main()
