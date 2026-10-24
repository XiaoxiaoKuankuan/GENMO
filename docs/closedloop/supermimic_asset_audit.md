# GENMO × SuperMimic P0 资产与实施边界

2026-10-24，用户明确要求本轮仅编写 P0–P7 代码，不执行测试、模型加载、推理、仿真或训练。下面是文件系统、Git 与源码的静态审读记录，不是模型验收报告。

| 项目 | 实际位置 / 状态 |
|---|---|
| GENMO 基线 | `feature/bumi-continuous-rolling`，本地及远端 `85d9cb191b9fbb209227b983dfc2ff89c4201fdc` |
| GENMO 独立工作树 | `/home/weili/bumi-closedloop-worktrees/supermimic/GENMO`，`feature/stage2-supermimic-worldmodel` |
| SM 基线 | `/home/weili/supermimic_vibed`，main，本地及远端 `bf833cf9856dbf54d173124cd7e77220e6c1aaba` |
| SM 独立工作树 | `/home/weili/bumi-closedloop-worktrees/supermimic/SuperMimic`，`feature/genmo-worldmodel-bridge` |
| SM 候选权重 | `/home/weili/supermimic_vibed/logs/bumi3_seven_filtered_v1/ckpt_024000.pt`，文件存在，68108549 字节；未加载、未验证内容或效果 |
| SM 派生机器人资产 | `/home/weili/supermimic_vibed/data/bumi3_4340/robot/` 存在，独立工作树不自带未跟踪资产；运行配置须显式指定该只读目录 |
| Stage1 s595000 | 当前本机文件名搜索未找到；已有配置记载服务器资产路径，不将其当成本机存在。运行前必须提供真正的 Stage1 checkpoint |
| Stage1 stats | `/home/weili/bumi-closedloop-worktrees/stage1_server2_scratch600k_20260930/assets/qpos30_train_stats.json` 存在；与待提供 checkpoint 的配套关系未核验 |
| Stage1 FK | `configs/bumi/bumi_kinematics_robot_retargeter_fe934_v1.json`，仍使用原 codec/关节名称 |
| 原配对数据 | 不新建、不改写；运行前提供原 Stage1 数据配置和可访问数据路径 |
| 正式进程 | 只读观察到七库 `supermimic/train.py` PID 3550702，以及既有 evaluate watcher；未操作进程、容器或 GPU |

源码契约为 state55、obs75、history10、goal570、action21、xi25、WM 输入101/输出54。编号 checkpoint 保存代码包含同一个快照的 `pi`、`wm`、`args`、`robot`、`motion`；本轮未反序列化候选文件，不能声称张量已经核验。

冻结网络的运行依赖为 PyTorch；FK 按需加载原生 MuJoCo 读取模型；真实物理仅使用 SM 的 IsaacLab3 / Newton / MuJoCo-Warp（50Hz 控制、4 子步、零控制延迟、自碰撞关闭）。不会导入 `supermimic/train.py`。新 Python 环境的实际可用性、CUDA 和物理启动均未检查。

GENMO 与 GMT 固定共享目录在开始时均为约定分支且干净；rolling 与 SM main 也干净。版本仅追溯，不设置源码 hash 相等门禁。冻结权重、归一化、物理参数、关节名称和数据契约仍需运行时校验。

后续交付状态与命令见 `supermimic_stage2.md`；所有测试条目在本轮均为“未执行”。
