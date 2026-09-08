#!/usr/bin/env bash
# BUMI 整段缓存桥：运动学资产与独立 Console 的 v5 s200000 归档保持一致。
# GMT 跟踪策略仍为 model_135000_stage2，不随舞蹈生成模型升级而替换。
# 只启动独立的 buffered 桥，不启动 Gazebo 或实机；附加参数可覆盖端口等运行选项。
set -euo pipefail
bumi_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$bumi_repo_root"
exec .venv/bin/python -u scripts/demo/demo_bumi_gmt_buffered_bridge.py \
  --kinematics inputs/checkpoints/bumi_5set_robot_retargeter_pass_v2_qpos30_contact_v5_s200000_20260907/assets/bumi_kinematics_robot_retargeter_fe934_v1.json \
  --gmt-policy /home/weili/docker_projects/bumi_GMT_deployment_listao/bumi_GMT_deployment_obs/src/legged_rl/rl_controller/rl_controllers/policy/bumi/model_135000_stage2.onnx \
  "$@"
