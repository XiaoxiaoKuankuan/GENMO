#!/usr/bin/env bash
# BUMI 整首生成控制台：统一使用 v5 s200000 的 checkpoint、ONNX、运动学与统计资产。
# 保留 CUDA ONNX、DDIM 20；输入音乐后先收齐全部 qpos，再提交给 buffered 桥。
# 仅更新独立入口的模型默认路径，不改变采样/足锁/播放逻辑或实时入口；后置参数仍可覆盖。
set -euo pipefail
bumi_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$bumi_repo_root"
exec .venv/bin/python -u scripts/demo/demo_music_bumi_buffered_console.py \
  --backend onnx \
  --checkpoint inputs/checkpoints/bumi_5set_robot_retargeter_pass_v2_qpos30_contact_v5_s200000_20260907/s200000.ckpt \
  --onnx outputs/onnx/bumi_music/rr_pass_v2_5set_v5_s200000_20260907/bumi_music_denoiser_v5_s200000_t120_qpos30_contact.onnx \
  --kinematics inputs/checkpoints/bumi_5set_robot_retargeter_pass_v2_qpos30_contact_v5_s200000_20260907/assets/bumi_kinematics_robot_retargeter_fe934_v1.json \
  --stats inputs/checkpoints/bumi_5set_robot_retargeter_pass_v2_qpos30_contact_v5_s200000_20260907/assets/bumi_qpos30_stats_train_5set_pass_v2_mine_fe934_v2.json \
  --onnx-provider cuda --ddim-steps 20 "$@"
