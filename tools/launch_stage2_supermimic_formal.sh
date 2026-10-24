#!/usr/bin/env bash
# 中文用途：服务器2正式GENMO×SuperMimic启动入口，转交Python完成真实资产与验收门禁、
# 空闲GPU/磁盘检查、唯一目录创建和tmux启动。仅检查通过才启动；失败保存阻塞证据。
# 数据流：已标定配置和真实报告→固定正式配置→训练进程→独立物理评价/完整checkpoint。
# 边界：不修改已有训练环境、不停止其他进程、不更新SM；恢复只使用完整checkpoint，
# 不伪恢复物理求解器。GENMO_SM_PYTHON可指定独立运行环境，所有参数原样转交。
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${GENMO_SM_PYTHON:-python}" "${script_dir}/launch_stage2_supermimic_formal.py" "$@"
