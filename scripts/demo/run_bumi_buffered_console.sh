#!/usr/bin/env bash
# BUMI 整首生成控制台：复用用户当前 s190000 CUDA ONNX、DDIM 20 配置。
# 输入音乐后先收齐全部 qpos，再提交给 buffered 桥；不改变模型、采样或实时入口。
set -euo pipefail
bumi_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$bumi_repo_root"
exec .venv/bin/python -u scripts/demo/demo_music_bumi_buffered_console.py \
  --backend onnx \
  --checkpoint inputs/checkpoints/bumi_5set_robot_retargeter_pass_v2_qpos30_contact_s190000_20260904/s190000.ckpt \
  --onnx outputs/onnx/bumi_music/rr_pass_v2_5set_s190000_20260904/bumi_music_denoiser_s190000_t120_qpos30_contact.onnx \
  --kinematics inputs/checkpoints/bumi_5set_robot_retargeter_pass_v2_qpos30_contact_s190000_20260904/assets/bumi_kinematics_robot_retargeter_fe934_v1.json \
  --stats inputs/checkpoints/bumi_5set_robot_retargeter_pass_v2_qpos30_contact_s190000_20260904/assets/bumi_qpos30_stats_train_5set_pass_v2_mine_fe934_v2.json \
  --onnx-provider cuda --ddim-steps 20 "$@"
