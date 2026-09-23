# GENMO：BUMI 音乐生成分支

本分支 `feature/bumi-music-only` 专注于 **音乐 → BUMI qpos30 + contact2**：
保留当前音乐数据生产、训练、离线评估、ONNX/TensorRT 导出和 GMT 部署。
人体生成、文本、Webcam、多模态服务和旧 SMPL/GMR 专用部署入口已从本分支移除。
这不是 closed-loop GENMO 分支，不在这里修改历史观测输入或开展第 4 步网络改造。

代码基于 NVIDIA GEM/GENMO；原项目作者与引用保留在本文末尾。
清理只改变本分支的 Git 源码，不删除生产数据、模型、统计量或历史评测报告。
详细边界与迁移说明见 [BUMI 音乐分支维护范围](docs/BUMI_MUSIC_BRANCH_SCOPE.md)。

## 当前实验

`configs/exp/` 仅保留两个显式入口：

| 实验 | 用途 |
|---|---|
| `gem_bumi_music_only_umr70_mine_scratch_350k` | 当前默认：UMR70 + Mine 四来源，qpos30/contact2、v5 损失、8 卡每卡 256、350k 从头训练 |
| `gem_bumi_music_only_5set_robot_retargeter_pass_v2_qpos30_contact_latest` | 保留的五库 v5 兼容实验，显式 checkpoint 权重续训；不是当前默认 |

通用训练配置以及音乐生成、ONNX 导出、checkpoint 选择、ONNX/TensorRT 验证工具的
默认实验均已对齐到 UMR70 + Mine。实验名称不能替代资产身份核验：
使用时仍须指定匹配的 checkpoint、kinematics 与训练集 stats，不自动创建或覆盖它们。

正式数据、统计量指纹和服务器启动入口见
[UMR70 + Mine 训练交付](docs/BUMI_UMR70_MINE_TRAINING.md)。
该文中的进程、训练进度和环境信息均是对应日期的历史记录，不代表当前在线状态。
现有训练入口为 `scripts/train_bumi_music.sh`；执行前应自行确认数据、设备和输出路径。
本次清理没有启动训练或同步训练服务器。

## 动作与条件

- 音乐条件：EDGE35，120 帧 @ 30 Hz。
- 网络输出：`qpos30` 与独立 `contact2`，不输出 51D 或额外 joint velocity。
- qpos30 由根部 heading 坐标 XY 位移、根高偏移、6D 根旋转和 21 维关节角构成。
- 现有 codec、根高兼容、FK、接触标签、足锁、30→50 Hz 与速度派生实现保持不变。
- 训练、ONNX 数值对照和离线运动学测试不等于 GMT 动力学或真机安全证明。

参阅 [qpos30/contact 契约](docs/BUMI_QPOS30_CONTACT_V3.md)、
[根高兼容](docs/BUMI_ROOT_HEIGHT.md) 和 [方法说明](docs/BUMI_GENMO_METHOD_CN.md)。

## 生成、评估与部署

`scripts/demo/` 仅保留以下九个 BUMI 入口：

| 入口 | 用途 |
|---|---|
| `demo_music_bumi.py` | PyTorch checkpoint 音乐生成 |
| `generate_bumi_music_npz_batch.py` | 批量音乐生成与 NPZ 输出 |
| `demo_music_bumi_console.py` | 常驻音乐生成控制台 |
| `demo_bumi_gmt_bridge.py` | 实时 GMT 桥 |
| `demo_bumi_onnx_gmt.py` | 离线生成与 GMT 播放 |
| `demo_music_bumi_buffered_console.py` | 整首预生成的仿真控制台 |
| `demo_bumi_gmt_buffered_bridge.py` | 按 GMT 仿真策略步推进的缓存桥 |
| `check_bumi_deployment.py` | 部署包与环境检查 |
| `run_bumi_deployment.py` | 配置式部署入口 |

各入口的参数可用 `python scripts/demo/<入口名>.py --help` 查看。
部署主文档为 [BUMI 音乐部署手册](docs/BUMI_MUSIC_DEPLOYMENT.md)，
接收端契约见 [GMT 对接](docs/BUMI_GMT_GENMO_INTERFACE.md) 与
[integrations/gmt](integrations/gmt/README.md)。
缓存模式保留，但旧硬编码 s200000 Shell 包装已删除，改用 Python 入口显式指定匹配资产；
[缓存模式说明](docs/bumi_buffered_gmt_playback.md) 已给出当前替代方式。

`tools/eval/` 保留 BUMI 音乐指标、渲染、模型选择、ONNX/TensorRT 对照与部署验证；
`tools/export/` 保留 BUMI 导出、engine 构建、部署打包与部署分支准备。
`scripts/validate_bumi_hq_music_full.py` 与
`scripts/build_bumi_hq_original_comparison.py` 保留完整音乐生成/原动作对比能力。

## 数据生产与测试边界

当前 UMR/robot_retargeter、Mine CSV、EDGE35、音乐与动作配对、清单与统计量校验、
公开音乐源数据转换和人工质量审核继续保留。
因此目录里仍可出现 SMPL-X、SONIC 等名称：源动作打包、具名关节重排或共享重采样
不等于保留独立的人体生成/文本部署入口，不能按名称一概删除。

`tests/bumi/` 覆盖当前配置、数据、编解码、损失、采样、导出和部署；
`tests/` 根目录保留音乐源数据、EDGE35、审核、定时器与进度报告等公共回归。
共享的 DDIM、音乐条件、GMT 轨迹、重采样、质量区间和定时器用例已经从旧混合测试拆出。

运行测试时应禁用字节码和 pytest 缓存，将 `--basetemp`、第三方缓存及日志都放在独立
系统临时目录，并在成功或失败后清理；完整约定见 [AGENTS.md](AGENTS.md)。
外部 GMT/CUDA 集成测试需要显式条件，未启用时不会自动启动外部控制器。
清理详情、实际验证和已知边界以 [记录文本.md](记录文本.md) 为准。

## 原项目与许可证

上游项目：[NVIDIA GEM/GENMO](https://research.nvidia.com/labs/dair/gem/)。
原作者：Jiefeng Li、Jinkun Cao、Haotian Zhang、Davis Rempe、Jan Kautz、Umar Iqbal、Ye Yuan。
本分支的范围收敛不改变上游归属与许可证。

## 📖 引用

```bibtex
@inproceedings{genmo2025,
  title     = {GENMO: A GENeralist Model for Human MOtion},
  author    = {Li, Jiefeng and Cao, Jinkun and Zhang, Haotian and Rempe, Davis and Kautz, Jan and Iqbal, Umar and Yuan, Ye},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025}
}
```

---

## 📄 许可证

本项目采用 NVIDIA OneWay Noncommercial License，详情请参阅 [LICENSE](LICENSE)。第三方组件遵循各自许可证，具体信息请参阅 [ATTRIBUTIONS.md](ATTRIBUTIONS.md)。
