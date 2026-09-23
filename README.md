# GENMO / BUMI music closed-loop

当前分支：`feature/bumi-music-closedloop`。本分支面向音乐驱动的 BUMI 动作生成，以及
“音乐 + 实际状态历史 + 已承诺参考前缀”的 closed-loop Stage 1。

模型输出保持 **qpos30 + 独立 contact2**，不预测 joint velocity，不使用 51D 动作表示。
Stage 2 的 Frozen GMT、动力学、Critic / DPPO 不由本页的离线入口启动。

## 当前代码与契约

- [Stage 1 契约](docs/closedloop/stage1_contract_v1.md)：120 帧 @ 30 Hz 的未来窗口、
  48D proprio、50 Hz 因果历史、逐坐标 known mask，以及下游 GMT 契约。
- [Stage 1 Actor、损失与小规模验证](docs/closedloop/stage1_model_v1.md)：
  当前独立网络/训练入口的配置、权重迁移、验证范围与证据边界。
- [Stage 1 四库数据配置](configs/closedloop/stage1_dataset_server1_fourset_v1.yaml)：
  历史长度、前缀配置和来源抽样比例；不改写生产数据或原统计量。
- [BUMI 音乐导出与部署](docs/BUMI_MUSIC_DEPLOYMENT.md)：现行 music-only 基线的导出、
  数值对照、发布与控制器接入。该文档的基线部署性能不等于 closed-loop 的部署验证。
- [修改记录](记录文本.md)：按时间记录实现、测试和未验证事项。

默认历史为 50 个采样点 @ 50 Hz，形状 `[B,50,48]`，首尾间隔 0.98 秒。
未来为 120 个采样点 @ 30 Hz，形状 `[B,120,30]`，从首点到末点为 119/30 秒。
已承诺前缀包含在这 120 帧内；前缀长度可以为零，不额外增加窗口。

## 保留的 music-only 实验

`configs/exp/` 只保留两个 BUMI 实验：

- `gem_bumi_music_only_5set_robot_retargeter_pass_v2_qpos30_contact_v5_scratch_s350000`：
  五库 v5 frozen 基线，也是现行 BUMI 生成/选权重/导出/parity 工具的默认实验。
- `gem_bumi_music_only_umr70_mine_scratch_350k`：UMR70 + Mine 实验。

`scripts/train.py` 要求显式选择 `exp`，不自动选用某个实验。closed-loop Stage 1 使用
独立的 `configs/closedloop/` 与 `tools/train_closedloop_stage1.py`，不能把旧 music-only
checkpoint 当作已经训练完成的 closed-loop Actor。

查看参数，不启动训练或部署：

```bash
python tools/train_closedloop_stage1.py --help
python scripts/demo/demo_music_bumi.py --help
python tools/export/export_bumi_music_onnx.py --help
python tools/eval/select_bumi_checkpoints.py --help
```

正式执行前须核对本机环境、数据划分、checkpoint、stats、kinematics 和对应指纹。
不要照搬历史文档中的服务器路径；本 README 不会下载模型、启动正式训练或同步服务器。

## 保留的 8 个 Demo / 部署入口

| 入口（均位于 `scripts/demo/`） | 用途 |
|---|---|
| `demo_music_bumi.py` | 音乐生成 BUMI 动作 |
| `generate_bumi_music_npz_batch.py` | 批量生成 BUMI NPZ |
| `demo_music_bumi_console.py` | 音乐生成控制台 |
| `demo_bumi_gmt_bridge.py` | 既有 BUMI → GMT 桥 |
| `demo_music_bumi_buffered_console.py` | 完整生成后缓存播放控制台 |
| `demo_bumi_gmt_buffered_bridge.py` | 缓存播放桥 |
| `check_bumi_deployment.py` | 部署检查 |
| `run_bumi_deployment.py` | 部署启动入口 |

buffered Python 入口仍在，绑定旧资产路径的两个 Shell 包装已删除。请通过 Python 入口
显式提供匹配的发布清单/模型与资产，先查看 `--help`。buffered 仿真播放不等于实机支持。

## 数据生产与验证边界

保留当前 BUMI producer、配对源数据审核、质量筛选、评估、导出和部署测试。
源音乐数据的 SMPL/SMPL-X 审核并不等于保留人体生成 Demo：AIOZ、FineDance、CoMPAS3D
审核需要的共享渲染已移到
[`tools/data/music_dance/render_utils.py`](tools/data/music_dance/render_utils.py)。

[四库人工筛选文档](docs/MOTION_CURATION_4SET.md)中的源数据审核仍可使用；
其验证工具只检查动作/音乐配对、划分与审核结果，不再构建已删除的 SMPL 实验。

旧人体、文本、Webcam、多模态 Demo、9 个 SMPL 实验及其专属直接调用方已退出本分支。
混合测试中的数据裁剪、采样、collate、条件/扩散数学和 BUMI 协议断言保留。
独立数据工具、底层共享数学和按需依赖没有因名称包含 SMPL 而一概删除。
历史文档中的退役命令仅用于追溯，不能作为当前分支的执行指南；源码可通过 Git 历史恢复。

## 上游项目

本分支基于 NVIDIA GEM / GENMO。上游作者：Jiefeng Li、Jinkun Cao、Haotian Zhang、
Davis Rempe、Jan Kautz、Umar Iqbal、Ye Yuan。
[上游项目页](https://research.nvidia.com/labs/dair/gem/)。

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
