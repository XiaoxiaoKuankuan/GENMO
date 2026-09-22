# 本地 93D/482138 资产退役清单（2026-09-22）

本文记录从本机 `/home/weili/GENMO` 工作副本清理旧 BUMI 93D/482138 模型与运行产物前的只读盘点证据。清理目标均被 Git 忽略，不属于版本化源码；历史算法、配置和命令仍可从对应 Git 提交以及本目录的归档文档恢复，但本文不保存大模型、视频或训练日志本体。

## 删除状态

2026-09-22 已完成本机清理。删除前再次逐项确认 10 个路径的 realpath、普通目录类型、
Git ignored/untracked 状态、内部符号链接、打开文件、进程命令行引用，以及与正式 v5
s350000、当前 ONNX/TensorRT、fe934 配置之间的路径隔离；检查结果均通过。随后只对下表
10 个精确绝对路径执行删除，未使用通配符或删除其父目录。删除后 10 个路径均不存在，
“明确保留的现行资产”仍全部存在。

这些本地二进制、视频和运行结果未进入 Git，删除不可通过当前工作树恢复；Git 历史只保留
曾版本化的算法、配置和文档。需要旧运行产物本体时只能从另行保存的外部备份恢复。

## 退役原因

当前 BUMI 音乐生成主链使用 `genmo.bumi_motion_features.qpos30.v3`、30 维 qpos、独立 2 维足接触输出、fe934 运动学资产和 v5 物理损失。下列资产则绑定旧 `genmo.bumi_motion_features.v2` 或 93 维网络输出，并使用 SHA256 为 `226306c217a39643a32ccc641b63e72c8ce0bf307af1fd6781b739fb694bcda0` 的 482138 阶段运动学。两套表示、统计量、checkpoint、ONNX、TensorRT 和运动学资产不得交叉组合。

## 删除前清单

以下大小来自 2026-09-22 删除前的 `du --apparent-size`；精确合计为
`10,541,691,826` bytes（约 9.82 GiB），共 1,200 个普通文件：

| 绝对路径 | 删除前大小 | 退役证据 |
| --- | ---: | --- |
| `/home/weili/GENMO/inputs/assets/bumi_manual_q1_v3` | 26 MiB | 含 482138 kinematics、`genmo.bumi_stats.v2` 93D stats 和 `genmo.bumi_hq5_release.v1` 发布记录 |
| `/home/weili/GENMO/inputs/checkpoints/bumi_5set_manual_q1_v3` | 3.2 GiB | 训练快照声明 `observed_motion_3d_dim=93`、`output_dim=93`、`feat_dim=93` |
| `/home/weili/GENMO/outputs/checkpoints/server2_bumi_random_v1` | 2.4 GiB | 旧 s430000 checkpoint 及 `genmo.bumi_stats.v1` 93D stats |
| `/home/weili/GENMO/outputs/onnx/bumi_music/s430000` | 1.4 GiB | `genmo.bumi_music_guided_denoiser_step.v1`，输出为 93D；checkpoint SHA256 为 `a61e4e4a7629304ed06e00f5479fcd059337bc7a9478b05bb8f7a04d5f501b58` |
| `/home/weili/GENMO/outputs/onnx/bumi_music/s350000_hq50` | 892 MiB | `genmo.bumi_music_guided_denoiser_step.v2` + `genmo.bumi_motion_features.v2`；旧 checkpoint SHA256 为 `a3a8bf58b430a9404116ac3ab0d44074842c362044472fb0f98c508270fbac55` |
| `/home/weili/GENMO/outputs/onnx/bumi_music/s350k_s50k` | 1.1 GiB | 同为 93D representation v2；旧微调 checkpoint SHA256 为 `5d3acfa5463c27040e50ddb3c869e7c935bcdc958ef9115140c39681bb516660` |
| `/home/weili/GENMO/outputs/tensorrt/bumi/s430000` | 414 MiB | `genmo.bumi_music_tensorrt_parity.v1`；旧 engine SHA256 为 `41dea596316e2d85f346ad3cb349efaff7083f8c297e15cd649e1543d54f1466` |
| `/home/weili/GENMO/outputs/bumi_onnx_gmt` | 399 KiB | 旧 s430000 诊断计划与运行记录 |
| `/home/weili/GENMO/outputs/server_music_wav_4set_hq50_s350k_s50k_20260825_videos` | 397 MiB | 旧 s350k+s50k 93D 模型验证视频 |
| `/home/weili/GENMO/outputs/training_analysis/bumi_5set_manual_q1_v3_s350000` | 213 MiB | 旧 manual-q1 v3 训练曲线、日志和分析结果 |

## 明确保留的现行资产

本次退役不删除下列现行基线，也不把旧资产的 SHA 或文件名当成现行验收证据：

- `/home/weili/GENMO/inputs/checkpoints/bumi_5set_robot_retargeter_pass_v2_qpos30_contact_v5_scratch_s350000_20260914`
- 当前 qpos30 v5 的 checkpoint、stats、fe934 kinematics、ONNX、TensorRT、部署 manifest 和最终验证结果
- `configs/bumi/bumi_kinematics_robot_retargeter_fe934_v1.json`
- `configs/bumi/sole_proxies_robot_retargeter_fe934_v1.json`

本次也不删除 `data/motions`、`data/motions_npz_bumi3_smooth_q1` 等历史源数据。它们是否继续保留属于数据归档决策，不应与代码入口退役或模型产物清理混为一谈。

## 证据边界

该清单证明的是文件路径、大小、嵌入契约和关键产物指纹。删除旧文件不会证明现行 qpos30 模型的训练质量、ONNX/TensorRT 数值一致性、GMT 动力学稳定性或实机安全；这些仍须使用现行资产单独验收。
