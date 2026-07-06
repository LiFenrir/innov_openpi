# Stage 2 在线 RL 训练操作文档

## 概述

Stage 2 在线 RL 在 Stage 1（RL Token 训练）产出的信息瓶颈编码器基础上，使用 TD3 算法（双 Q-critic + BC 正则化）在线优化 Actor 策略。训练循环在 Training PC（有 GPU）运行，机器人执行在 Robot PC（连接硬件）进行，两台机器通过 WebSocket 通信。

## 架构总览

```
Robot PC (边缘)                     Training PC (本机, GPU)
────────────────                     ──────────────────────
rl_robot_bridge.py                    Stage 2 训练循环
├─ robodeploy Robot 硬件              ├─ VLA (冻结) → a_tilde 参考动作
├─ WebSocket Client ─────msgpack──→   ├─ RLTokenModel (冻结) → z_rl
├─ 键盘: s=子任务完成 f=失败          ├─ Actor (可训练) → a
│                                     ├─ TwinQCritic (可训练) → Q(s,a)
│                                     ├─ ReplayBuffer (CPU)
│                                     └─ RemoteWebSocketEnv (端口5556)
```

---

## 前置条件

### 1. SFT 微调模型（Stage 0）

```
checkpoints/bi_s1_pi05_sft_shifted/bi_s1_sft_shifted_run/24000/
├── model.safetensors    (7.5 GB)
├── optimizer.pt
├── metadata.pt
└── assets/
    └── bi_s1_sft_shifted/
        └── norm_stats/  (归一化统计量)
```

### 2. Stage 1 RL Token 模型

```bash
# 训练命令
CUDA_VISIBLE_DEVICES=1 python scripts/train_rl_token.py \
    --config configs/rlt/stage1_rl_token_bi_s1_sft_shifted.yaml

# 输出
checkpoints/rl_token/bi_s1_sft_shifted_stage1/
├── rl_token_step1000.pt
├── rl_token_step2000.pt
├── ...
└── rl_token_step5000.pt  ← 用于 Stage 2
```

### 3. Stage 2 配置文件

```bash
# configs/rlt/stage2_online_rl.yaml
```

关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `action_dim` | 14 | bi_s1 动作维度：左右臂各 7 DOF |
| `chunk_length` | 10 | 每 chunk 的动作步数 |
| `embedding_dim` | 2048 | z_rl 维度 |
| `gamma` | 0.99 | 折扣因子 |
| `utd_ratio` | 5 | 每 episode 的更新次数 |
| `bc_regularizer_beta` | 0.5 | BC 正则化强度 |
| `batch_size` | 256 | TD3 采样批次大小 |
| `buffer_capacity` | 100k | ReplayBuffer 容量 |
| `max_env_steps` | 100k | 总环境步数 |
| `warmup_steps` | 1000 | 预热收集 chunk 数 |
| `save_every` | 50 | 每 N episode 保存检查点 |
| `actor_lr` / `critic_lr` | 3e-4 | 学习率 |

### 4. 网络连通性

- Training PC 开放端口 **5556**（RL WebSocket）
- Robot PC 能访问 Training PC 的 IP

---

## 操作流程

### Step 1: 启动 Training PC

mobaxterm 连接192.168.1.17 kemove 密码：123456
```bash
cd /home/kemove/VLA/innov_openpi
conda activate rise

CUDA_VISIBLE_DEVICES=1 python scripts/train_online_rl.py \
    --config configs/rlt/stage2_online_rl.yaml \
    --env-factory rlt.rollout.remote_env.make_remote_env \
    --vla-checkpoint-dir checkpoints/bi_s1_pi05_sft_shifted/bi_s1_sft_shifted_run/24000 \
    --vla-config-name configs/bi_s1/pi05_finetune_sft_shifted.yaml \
    --rl-token-checkpoint checkpoints/rl_token/bi_s1_sft_shifted_stage1/rl_token_step5000.pt \
    --action-dim 14 \
    --chunk-length 10 \
    --task-prompt "Grasp a single layer of the cloth with the gripper, then place the cloth onto the board"
```

启动后输出：
```
RemoteWebSocketEnv: port=5556 action_dim=14 chunk_length=10
Waiting for Robot PC to connect on port 5556...
```

**此时阻塞等待 Robot PC 连接。**

### Step 2: 启动 Robot PC

```bash
cd /home/kemove/VLA/robodeploy

# 单任务模式
python src/robodeploy/scripts/rl_robot_bridge.py \
    --robot.type=bi_s1_follower \
    --robot.left_arm_port=/dev/left_follower \
    --robot.right_arm_port=/dev/right_follower \
    --robot.cameras='{"front":{"type":"intelrealsense","serial_number_or_name":"135122077817","width":848,"height":480,"fps":30},"front_1":{"type":"intelrealsense","serial_number_or_name":"935422072733","width":848,"height":480,"fps":30},"left_wrist":{"type":"intelrealsense","serial_number_or_name":"409122273564","width":640,"height":480,"fps":30},"right_wrist":{"type":"intelrealsense","serial_number_or_name":"409122273228","width":640,"height":480,"fps":30}}' \
    --host=192.168.1.17 \
    --port=5556 \
    --action_dim=14 \
    --chunk_length=10 \
    --task="Grasp a single layer of the cloth with the gripper, then place the cloth onto the board" \
    --fps=15

# 多阶段任务模式（3个子任务）
python src/robodeploy/scripts/rl_robot_bridge.py \
    ... \
    --num_subtasks=3 \
    --task="grasp cloth → lift → place on board"
```

> **注意**：摄像头 key 名称需与 VLA 的 SFT 配置一致。bi_s1 默认 `front`、`left_wrist`、`right_wrist`。

### Step 3: 训练开始

Robot PC 连接成功后，Training PC 输出：
```
Robot PC connected on port 5556
Starting online RL training...
┌─ Warmup ──────────────────────────┐
│ Collecting 1000 warmup chunks...  │
│ VLA-only policy, exploring        │
└──────────────────────────────────────────────────┘
┌─ Online RL Loop ─────────────────┐
│ Episode 1: chunks=X, reward=Y    │
│ Episode 2: ...                   │
└──────────────────────────────────────────────────┘
```

---

## 训练过程详解

### Phase 1: Warmup（预热）

```
Warmup 1000 chunk (≈ 20-30 分钟)

每个 chunk:
  1. VLA 观测 → VLA 推理 → 参考动作 a_tilde
  2. 提取 z_rl + s_p → RL state x
  3. env.step(a_tilde) → Robot PC 执行
  4. 存入 buffer: (x, a_tilde, a_tilde, r, next_x, done)

完成后自动保存 warmup_buffer.pt
```

### Phase 2: 在线 RL 循环

```
while total_env_steps < 100,000:

  ┌─ Episode Collection ──────────────────────┐
  │ Actor(eval) 控制，探索噪声关闭             │
  │                                            │
  │ env.reset()  → Robot PC 归零               │
  │ while not done:                            │
  │   z_rl = RLToken.encode(VLA.embed(obs))    │
  │   a_tilde = VLA.reference_action()         │
  │   a = Actor(x, a_tilde)                    │
  │   next_obs, r, done = env.step(a)          │
  │   buffer.add(x, a, a_tilde, r, next_x, d)  │
  │                                            │
  │ stats = {reward, steps, success}            │
  └────────────────────────────────────────────┘

  ┌─ TD3 Update × 5 ──────────────────────────┐
  │ for g in range(5):                         │
  │   batch = buffer.sample(256)               │
  │                                            │
  │   [Critic] (每次都更新)                     │
  │   td = Σγᵏrₖ + γᶜ(1-d)min Q'(x',a')      │
  │   L = MSE(Q1,td) + MSE(Q2,td)              │
  │                                            │
  │   [Actor] (每2步1次)                        │
  │   L = -min Q(x, Actor(x))                  │
  │       + 0.5·MSE(Actor(x), a_tilde)         │
  │                                            │
  │   [Target] Polyak: θ' = 0.995θ' + 0.005θ  │
  └────────────────────────────────────────────┘
```

### Robot PC 按键操作

| 按键 | 单任务模式 | 多阶段模式 (`--num_subtasks=3`) |
|------|-----------|-------------------------------|
| `s` | 任务成功，episode 结束 | 当前子任务完成，进入下一阶段；最后阶段 → 成功结束 |
| `f` | 任务失败，episode 结束 | 任意阶段失败，episode 结束 |

---

## 检查点与恢复

### 自动保存

每 50 episode 自动保存到：
```
checkpoints/online_rl/<run_name>/
├── warmup_buffer.pt          (Warmup 完成后)
├── online_rl_ep50.pt         (ep50)
├── online_rl_ep100.pt        (ep100)
└── ...
```

每个检查点包含：
```python
{
    "actor": Actor.state_dict(),
    "critic": TwinQCritic.state_dict(),  # Q1, Q2, target ×2
    "actor_optimizer": Adam.state_dict(),
    "critic_optimizer": Adam.state_dict(),
    "replay_buffer": ReplayBuffer.state_dict(),  # 全部历史 transitions
    "total_env_steps": int,
    "total_updates": int,
    "total_episodes": int,
}
```

### 恢复训练

```bash
# 从 ep50 恢复，跳过 warmup
CUDA_VISIBLE_DEVICES=1 python scripts/train_online_rl.py \
    --config configs/rlt/stage2_online_rl.yaml \
    --env-factory rlt.rollout.remote_env.make_remote_env \
    --vla-checkpoint-dir checkpoints/bi_s1_pi05_sft_shifted/bi_s1_sft_shifted_run/24000 \
    --rl-token-checkpoint checkpoints/rl_token/bi_s1_sft_shifted_stage1/rl_token_step5000.pt \
    --resume-checkpoint checkpoints/online_rl/<run_name>/online_rl_ep50.pt

# 输出:
# Skipping warmup — replay buffer already has 1000 transitions
# Loaded checkpoint ... (episode 50, step 5000)
```

### 从预录 warmup 开始（跳过实时预热）

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_online_rl.py \
    ... \
    --warmup-buffer checkpoints/online_rl/<run_name>/warmup_buffer.pt
```

---

## 显存与硬件

| 组件 | 显存 |
|------|------|
| VLA（PaliGemma 2B + SigLIP + Gemma 300M） | ~5.4 GB（bfloat16 权重，冻结） |
| RLTokenModel | ~0.1 GB（冻结） |
| Actor + TwinQCritic | < 0.01 GB（可训练） |
| VLA 推理激活值 | ~5-7 GB |
| **总计** | **≈ 11-14 GB** |

RTX 4090 (24 GB) 单卡足够，建议 `CUDA_VISIBLE_DEVICES=1`。

ReplayBuffer 存在 CPU 内存（100k transitions ≈ 580 MB），不占显存。

---

## 耗时估算

| 阶段 | 时间 |
|------|------|
| 模型加载 | ~30 秒 |
| Warmup 1000 chunk | 20-30 分钟 |
| 纯训练 100k 步 | 4-5 小时（连续运行） |
| **实际总耗时**（含人工重置场景） | **10-15 小时**，建议分 3-4 天 |

---

## 常见问题

### Q: 训练中断后怎么恢复？

```bash
# 查找最新检查点
ls -t checkpoints/online_rl/<run_name>/online_rl_ep*.pt | head -1

# 添加 --resume-checkpoint 重新启动
```

### Q: Robot PC 断连怎么办？

Training PC 的 `RemoteWebSocketEnv` 超时（30s）后会抛异常。重新启动 Robot PC 的 `rl_robot_bridge.py`，然后重启 Training PC 训练脚本（加上 `--resume-checkpoint`）。

### Q: 如何调整探索强度？

修改 YAML 配置或 CLI 覆盖：
```bash
--actor-noise-sigma 0.2    # 增大探索噪声 (默认0.1)
```

### Q: Actor 输出动作异常怎么办？

Actor 最后一层零初始化，初始时完全复现 VLA 参考动作。如果长期不改进，降低 BC 正则化强度：
```bash
--bc-regularizer-beta 0.1  # 降低约束 (默认0.5)
```

### Q: 如何评估训练效果？

```bash
python scripts/evaluate.py \
    --env-factory rlt.rollout.remote_env.make_remote_env \
    --checkpoint checkpoints/online_rl/<run_name>/online_rl_ep200.pt \
    --num-episodes 10
```
