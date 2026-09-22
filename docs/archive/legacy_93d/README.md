# BUMI 93D/482138 历史文档归档

本目录保存已经退役的 BUMI 93D 表示、482138 机器人资产、legacy pickle 质量筛选和旧版
ONNX/TensorRT/GMT 部署记录。文档正文为审计与历史复现保留，其中的路径、配置名称、
checkpoint、stats 和命令不再代表当前仓库状态，**不得复制后直接执行**。需要复现旧结果时，
应同时检出对应历史提交，并核对当时的资产指纹、表示版本和完整运行环境。

## 归档内容

- [旧版 BUMI ONNX 到 GMT 部署链路](BUMI_ONNX_GMT_DEPLOYMENT.md)
- [旧版 BUMI-native Music-only GENMO 工程记录](bumi_native_music_genmo.md)
- [旧版 BUMI-GENMO 93D 方法说明](BUMI_GENMO_METHOD_CN.md)
- [旧版 482138/GMR 动作质量筛选方案](bumi_motion_quality_filter.md)
- [本地 93D/482138 资产退役清单](LOCAL_ARTIFACT_RETIREMENT_20260922.md)：记录删除前的
  精确路径、大小、表示契约和明确保留的 qpos30/v5 资产；不保存模型或运行产物本体。

## 现行 qpos30/v5 入口

- [BUMI qpos30、FK 接触与足底锁定 v3](../../BUMI_QPOS30_CONTACT_V3.md)：当前动作表示、
  contact head、FK 与后处理契约。
- [BUMI UMR70 + Mine 训练说明](../../BUMI_UMR70_MINE_TRAINING.md)：当前正式训练入口、
  数据集合约和运行边界。
- [BUMI 音乐部署说明](../../BUMI_MUSIC_DEPLOYMENT.md)：当前 qpos30/v5 导出与部署入口。
- [BUMI 根高度说明](../../BUMI_ROOT_HEIGHT.md)：当前根高度语义与兼容边界。
- [闭环 Stage 1 静态契约](../../closedloop/stage1_contract_v1.md)：闭环条件接口定义；该文档
  不代表闭环训练实现已经完成。
- [已退役的 BUMI 资产与表示](../../RETIRED_BUMI_ASSETS.md)：源代码、配置与资产退役范围。
