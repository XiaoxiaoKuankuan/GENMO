# BUMI 文本生成动作

本分支 `feature/bumi-text-only` 维护 BUMI 四库文本训练、数据准备、评测与本地推理。
唯一实验入口是 `configs/exp/gem_bumi_text_fullseq.yaml`，名称为兼容保留，实际训练为
`gem_bumi_text_crop120`：源动作完整保存，30 Hz，训练随机120帧，验证固定窗口。
机器人采用 fe934 资产和 MuJoCo 原生21关节顺序，网络30D表示经固定编解码生成qpos28；
默认根高参考为0.48120910 m，真实姿态的根高仍由动作决定。

## 当前文档

- [四库训练、采样与恢复](docs/BUMI_TEXT_CROP120.md)
- [UMR文本动作筛选与数据准备](docs/bumi_text_umr_preprocessing.md)
- [KIT-ML源动作准备](docs/kitml_amass_preparation.md)
- [独立推理部署包](docs/BUMI_TEXT_DEPLOY.md)
- [本地文本网页](docs/text_motion_web.md)
- [根高定义与旧模型兼容](docs/BUMI_ROOT_HEIGHT.md)
- [本次分支整理与功能边界](docs/TEXT_BRANCH_CLEANUP.md)
- [测试范围与历史源码说明](tests/README.md)

## 安装与入口

使用与当前训练项目匹配的 PyTorch/CUDA 环境，在仓库根目录安装 `pip install -e '.[train,web,dev]'`。
独立部署包使用其自带的安装器；训练环境与独立部署环境不是同一依赖集合。

```bash
# 查看当前合成配置，不启动训练
python -B scripts/train.py --cfg job
# 数据准备、筛选、统计的具体参数见 UMR 文档
python -B tools/data/bumi/prepare_bumi_text.py --help
# 导出、校验和打包
python -B tools/export/bumi_text.py --help
# 本地网页，只发现并接受 BUMI 文本模型
python -B scripts/demo/demo_bumi_text_web.py --port 8767
```

模型、T5和正式数据由配置指定的外部路径提供，不随代码仓库发布。训练数据必须满足
当前资产、帧率、文本时间范围和统计量契约；旧93D权重及旧机器人资产不能直接复用。

## 历史边界与保留内容

SMPL人体、音乐、视频重建、多模态服务及旧数据构建在对应功能分支维护，本分支已移除
这些运行入口。公共质量统计、重采样、T5编码、TensorRT和训练基类已有独立模块。
历史full300文本模型仍按自身契约读取，不能用crop120配置跨契约完整恢复。

历史文档集中在 `docs/archive/`；历史测试源码保留原位置，默认只执行 `pytest.ini`
列出的当前测试。`assets/bumi_viewer/`、`outputs/bumi_umr_binary_latest/` 与
`docs/reports/` 的正式资产及报告保留原内容。当前代码测试不等于模型语义或动力学验收。

项目来源与许可证见 [ATTRIBUTIONS.md](ATTRIBUTIONS.md) 和 [LICENSE](LICENSE)。
