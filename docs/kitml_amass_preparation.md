# KIT-ML 与 AMASS 的来源对齐和 30 Hz 数据

所属分支：`feature/bumi-text-only`。入口：`tools/data/kitml/prepare_kitml_amass.py`。

## 当前来源选择

用户于 2026-09-20 确认复用本地
`/home/weili/GENMO/inputs/AMASS/hmr4d_support/smplxpose_v2.pth`。
此文件含 18,086 条 GENMO 预处理记录，来源模型明确为 SMPL-X，保存身体姿态
`pose[T,66]`、平移 `trans[T,3]`、形状 `beta[10]` 和性别。
帧率依据现有 `gem/datasets/pure_motion/amass.py` 的 30 Hz 契约；容器没有原始帧率字段。
本次不重新拟合人体，不补造手部/面部参数，也不把该容器声明为原始 SMPL+H G。

官方 KIT-ML 2017-06-22 包有 3,911 条动作和 6,353 条文本，保留原始 ZIP。
参考项目：[AMASS-Annotation-Unifier](https://github.com/Mathux/AMASS-Annotation-Unifier)，
固定提交 `a0c031ad6af69a1f64c1dd4d4d199ff6974d1708`，外部源码保留原 MIT LICENSE。
其已发布映射覆盖 3,701 个 ID：KIT 2,092、CMU 1,566、EKUT 43；另有 210 个未匹配。
本地容器精确命中其中 3,699 个，缺 `KIT/572/dance_waltz 01_poses.npz` 和
`KIT/572/dance_waltz 02_poses.npz`。最终准入数量以服务器报告为准，不能把映射命中等同对齐成功。

## 服务器2的路径与命令

```text
/data0/user/liwei/datasets/KIT-ML/
  raw/2017-06-22.zip
  sources/AMASS-Annotation-Unifier/
  genmo_30hz/
    metadata.json
    metadata_ready.json
    motions_30hz/kitml_<ID>_poses.npz
/data0/user/liwei/datasets/AMASS/hmr4d_support/smplxpose_v2.pth
/data0/user/liwei/dataset_reports/kitml_amass_latest/
/data0/user/liwei/logs/kitml_amass/
```

```bash
cd /home/user/liwei/GENMO-bumi-text
PYTHONDONTWRITEBYTECODE=1 /data0/user/liwei/envs/GENMO-cu128/bin/python \
  tools/data/kitml/prepare_kitml_amass.py \
  --kitml-root /data0/user/liwei/datasets/KIT-ML/raw/2017-06-22.zip \
  --mapping-root /data0/user/liwei/datasets/KIT-ML/sources/AMASS-Annotation-Unifier/kitml_process \
  --unifier-commit a0c031ad6af69a1f64c1dd4d4d199ff6974d1708 \
  --amass-genmo-file /data0/user/liwei/datasets/AMASS/hmr4d_support/smplxpose_v2.pth \
  --output-root /data0/user/liwei/datasets/KIT-ML/genmo_30hz \
  --report-root /data0/user/liwei/dataset_reports/kitml_amass_latest
```

如果将来获得原始 AMASS `SMPL+H G`，将 `--amass-genmo-file` 换为
`--amass-root /path/to/amass`，该根目录必须包含 KIT/CMU/EKUT 对应层级。
官方 AMASS 下载页需要账户访问；本脚本不处理密码、登录或访问许可。

## 产物与对齐边界

- `metadata.json` 保留全部官方 ID、原始 texts、逐条 caption、来源 metadata、映射方法及状态。
- `metadata_ready.json` 只纳入通过身份、有限性、形状与时长检查且成功写出 NPZ 的记录。
- `amass_path` 是 Unifier 的逻辑 SMPL+H 来源路径；复用 GENMO 容器时，该路径不表示已经下载了同名原始 NPZ。
  真实输入由 `source_container`、`source_key` 和 `source_sha256` 标识；真实输出为 `motion_path`。
- 原 KIT-ML caption 描述整个动作。`start/end` 和 `annotations[].start_time/end_time` 是输出动作的秒数区间，
  end 为排他的 `num_frames / fps`；`source_start_time/source_end_time` 保存输入时长。
  未获得有效动作的 end 为 null，不借 MMM 时长冒充 AMASS 时长。
- GENMO 来源只删除固定容器前缀和重复的子集目录，将 `_stageii.npz` 对应到 `_poses.npz`，
  子集、被试和完整动作名必须相同，不模糊匹配 basename、不自动修正空格。
- GENMO 来源用 MMM 的真实 timestamps 对照 `T/30`，只允许一个 MMM 源帧加一个 30 Hz 帧的量化差；
  原始 SMPL+H 来源另要求原始帧数相等，并检查帧率对应时长。错配写入 `alignment_mismatch`。
- NPZ 保留 `poses[T,66]`（根轴角 3 + 身体轴角 63）、`trans[T,3]`、`betas[10]`、gender、
  `mocap_framerate=30`、SMPL-X 模型身份、缺失手脸字段清单。沿用原 AMASS/GENMO 世界坐标，
  `coordinate_transform=identity`，不二次执行 Y-up/Z-up 变换或接地平移。
- 原始 SMPL+H 分支保存 156 维身体和手部参数、完整 betas 和可选 DMPL，复用已有音乐数据的
  SLERP/线性重采样，不做轴角逐元素插值；原始形状/性别保留。
- `summary.json` 汇总全部状态、caption、时长、映射指纹与相同 AMASS 源动作组。
  后续划分 train/val/test 必须以 AMASS 来源组去重；还需核对与 HumanML3D 的 AMASS 来源交集。
- 本交付属于重定向前的人体数据。没有执行 BUMI 重定向、机器人质量筛选、T5 embedding、
  训练格式构建或训练；`training_ready=false` 明确记录这一边界。

## 验证

`tests/test_prepare_kitml_amass.py` 验证缺文件、坏映射、帧数/时长错配、NaN、模型维度、
跨 ±π 最短路径 SLERP、手部/DMPL/形状保留、ZIP 输入、文本时间范围及已有 GENMO 容器身份。
测试写入系统临时目录并关闭 Python/pytest 缓存；真实全量结果、下载完整性与跨机 SHA
记录在服务器报告及根目录 `记录文本.md`，不以测试通过代替全量交付。
