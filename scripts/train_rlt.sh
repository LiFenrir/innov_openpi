CUDA_VISIBLE_DEVICES=1 python scripts/train_online_rl.py \
    --config configs/rlt/stage2_online_rl.yaml \
    --env-factory rlt.rollout.remote_env.make_remote_env \
    --vla-checkpoint-dir checkpoints/bi_s1_pi05_sft_shifted/bi_s1_sft_shifted_run/24000 \
    --vla-config-name configs/bi_s1/pi05_finetune_sft_shifted.yaml \
    --rl-token-checkpoint checkpoints/rl_token/bi_s1_frozen_20k/rl_token_step20000.pt \
    --action-dim 14 \
    --chunk-length 10 \
    --task-prompt "Grasp a single layer of the cloth with the gripper, then place the cloth onto the board"