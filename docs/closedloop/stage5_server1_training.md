# 服务器 1：BUMI closed-loop Stage1 训练与恢复

本说明对应 `feature/bumi-music-closedloop`，以
[`configs/closedloop/stage1_server1_8gpu.yaml`](../../configs/closedloop/stage1_server1_8gpu.yaml)
为正式配置。下面的命令仅供用户手动启动，不会自动执行、创建后台任务或提交定时任务。
首次正式训练与本轮短程验收使用不同目录和初始化来源。

**2026-09-23 真实 GPU 短程验收通过：单卡 10 步 → 8 卡 DDP 10 步 → 完整恢复再 5 步。**
使用四库真实数据和完整原音乐权重初始化模型，没有使用缩小网络。每卡 batch 32、
梯度累计 1、有效全局 batch 256 已实际通过 8 卡训练与恢复；50,000 步为正式训练配置，
本轮没有执行。机器可读证据见[验收报告](stage5_server1_acceptance.json)。

| 验收段 | 实际优化步 | 每卡 / 全局 batch | loss 最小–最大 | 裁剪前梯度 norm 最小–最大 | 峰值 allocated / reserved GiB | 首步后平均秒/步 |
|---|---:|---:|---:|---:|---:|---:|
| 单卡 | 0→10，共 10 步 | 8 / 8 | 0.10060–1.58396 | 2.45894–50.61327 | 4.019 / 4.178 | 0.12650 |
| 8 卡 DDP | 0→10，共 10 步 | 32 / 256 | 0.31660–0.54866 | 1.31375–3.98856 | 7.180 / 7.543 | 0.20671 |
| 8 卡 resume | 10→15，共 5 步 | 32 / 256 | 0.29774–0.39858 | 1.02305–1.75465 | 7.182 / 7.559 | 0.19611 |

显存取所有 rank 的 PyTorch 峰值；梯度值为 clip=1.0 前的 norm。step 计时包含取 batch、
前反传与更新，不包含步后日志、验证和 checkpoint；每段首步分别为 0.829、1.118、
1.143 秒。含段尾验证/checkpoint 的训练循环总时间分别为 6.051、7.105、6.140 秒，
不含启动、数据索引与权重加载，不能据此承诺完整 50,000 步工期或更大 batch 的吞吐。

两次 8 卡结束时各 rank 的完整模型哈希一致。直接读取 checkpoint 验证 Adam 的 320 组
参数状态 step 从 10→15、scheduler.last_epoch 从 10→15，八个采样游标从 320→480；
随机状态也按 rank 恢复。BF16 不使用 GradScaler，保存值为 null。warm start 加载 312
个原有 state 项、新增条件 10 项，缺失、非预期和形状冲突均为 0；旧 global_step 和
optimizer 均未恢复。checkpoint 含 221,736,736 个 state 元素，维持完整 1024/16/8 主干。

每段结束固定四库各 1 个 val 样本，纯条件 DDIM 10 步、CFG 2.5；12 次采样均输出
`[1,120,30]` qpos30 与 `[1,120,2]` contact，输出有限，最终物理前缀与每步 normalized
已知坐标最大误差均为 **0**。这些固定样本的已知坐标数分别为 AIST 58、AIOZ 178、
FineDance 298、Mine 268，覆盖前缀末端 XY 位移未知的逐坐标约束。固定 AIST 样本的
生成监督 loss 仍为约 3.34–4.31，短测只证明执行正确，不说明动作质量已经合格。

短测显式覆盖 `train.max_steps` 为 10/15、scheduler warmup 为 0、日志每步、验证每库
1 个样本/10 步；正式配置恢复 warmup 500、每库 2 个样本/20 步，其余模型和损失一致。
日志中的 NCCL 初始 barrier 设备提示与已有 AMP API 弃用提示未阻止验收；设备实际由
LOCAL_RANK 显式绑定，未出现 NCCL 失败、OOM 或非有限损失。未升级服务器依赖。

## 模型、数据与初始化

保持完整的 GENMO 扩散 Transformer：latent 1024、16 层、8 头，沿用音乐编码、主干和
qpos30/contact2 输出 head。新增条件继续使用 GRU 历史编码与 MLP 前缀编码，历史 H=50、
前缀 P∈[0,24]；动作布局始终为 120 帧 @ 30 Hz。没有改成 51D，没有增加 joint velocity
预测，没有接入 GMT、Critic 或 DPPO，也没有修改 30→50 Hz 速度派生链。

| 项目 | 服务器 1 路径 |
|---|---|
| 代码工作树 | `/home/user/liwei/GENMO-bumi-closedloop` |
| Python 解释器 | `/home/user/liwei/GENMO/.venv/bin/python` |
| 四库数据根目录 | `/data0/user/liwei/datasets/bumi_music_umr70_mine_pass_90505_v1` |
| 数据配置 | `/home/user/liwei/GENMO-bumi-closedloop/configs/closedloop/stage1_dataset_server1_fourset_90505_v1.yaml` |
| 正式训练配置 | `/home/user/liwei/GENMO-bumi-closedloop/configs/closedloop/stage1_server1_8gpu.yaml` |
| 正式输出目录 | `/data0/user/liwei/GENMO_outputs/bumi_closedloop_stage1_90505_warm_s350000_8gpu_v1` |
| 原音乐初始化权重 | `/data0/user/liwei/GENMO_outputs/bumi_music_umr70_mine/bumi_umr70_mine_s350k_20260918_170256_6ac1e6d/version_0/checkpoints/s350000.ckpt` |
| 与初始化权重匹配的 stats | `/data0/user/liwei/datasets/bumi_music_umr70_mine_pass_v1/stats/qpos30_train_stats.json` |
| 运动学 | `/home/user/liwei/GENMO-bumi-closedloop/configs/bumi/bumi_kinematics_robot_retargeter_fe934_v1.json` |

配置固定了上述权重、stats 和运动学的 SHA256 以及关节顺序；加载器严格核验表示、
参数形状和来源配置。正式运行有意使用原音乐模型的 stats，保持其 normalized x0
数值空间；数据清单仍使用新 90/5/5 版本。新划分重新计算的 train stats 保留在新数据
目录中，不在本次 warm-start 实验中替换原模型 stats。

四库为 AIST++、AIOZ-GDANCE、FineDance、Mine，训练集内部的抽样概率依次为
20%、35%、25%、20%。这是 train 内的采样概率，与动作按组分出的 90%/5%/5% 不同。
现有 Dataset、validator、qpos30 表示、coordinate mask、anchor、时间和来源记录保持不变。

Mine 保留 `legacy_body_origin_min_zero` 语义，在完整源序列上复用原接触标签器的 FK
足底分位估地。结果缓存并写入既有 `meta.ground_supervision`，只用于接触高度和穿地等
损失，不进入 Actor 条件；不改变源动作 Root Z、已存接触标签或坐标。其他三库继续使用
原 floor-zero 地面。实现细节见[模型说明](stage1_model_v1.md)。

## 首次初始化与真正恢复

**首次正式训练使用 weights-only warm start。** 从指定原音乐 s350000 权重加载匹配的
音乐编码、Transformer 和原输出 head；新增条件分支独立初始化。旧 optimizer、旧
scheduler、旧 global_step 不恢复，Stage1 从自己的 step 0 开始。配置要求明确的
warm-start 来源，来源缺失或身份不匹配会报错，不会静默改成从零训练。参数迁移详情写入
`weight_loading_report.json`，列出已加载、新增、缺失、形状冲突和非预期参数。

**续训使用 full-state Stage1 checkpoint。** 保存并恢复模型、AdamW optimizer、
scheduler、AMP scaler（使用 FP16 时）、已完成的 global_step，以及每个 rank 的随机
状态和采样 epoch/消费位置。默认 BF16 不使用 GradScaler，checkpoint 明确记录其禁用
状态。恢复校验训练配置、精度、world size、batch、累计步数、数据指纹和原 warm-start
来源；不允许把普通 Actor 权重或原音乐 checkpoint 当作可恢复训练状态。

8 卡训练使用每 rank 独立设备、加权采样位置分片和 DDP 梯度同步。加权采样允许同一动作
被多次抽到，但各 rank 消费的全局抽样位置不同。仅 rank0 写公共日志、TensorBoard、
验证结果和 checkpoint。验证时其他 rank 等待，随后一起继续优化。

## 当前正式训练参数

以下 batch 是本轮实际验证后的保守建议，没有根据 85GB 显存推算或搜索最大 batch。

| 参数 | 当前配置 |
|---|---:|
| GPU 数 | 8 |
| 每卡 batch | 32（已通过 8 卡真实训练与恢复） |
| 梯度累计 | 1 |
| 有效全局 batch | `8 × 32 × 1 = 256` |
| 训练终点 | 50,000 个 optimizer 更新步 |
| 精度 | BF16；物理损失内部 FP32 |
| AdamW 学习率 | `1e-5` |
| weight decay | `0.01` |
| 梯度裁剪 norm | `1.0` |
| scheduler | 前 500 步 warmup，固定 50,000 步 cosine，末端比例 `0.1` |
| 每卡 DataLoader worker | 2 |
| 每 rank 每个采样 epoch 的样本数 | 8,192 |
| 控制台日志间隔 | 10 步；JSONL 每步写入 |
| 验证间隔 | 500 步，并在本次运行终点验证 |
| checkpoint 间隔 | 1,000 步，并在本次运行终点保存 |

`train.max_steps` 表示绝对优化步终点：例如从 step 10 恢复且设置为 15，实际执行 5 步。
短程测试保持正式 scheduler 的总周期 50,000，并将 warmup 临时设为 0。完整恢复要求保持采样、精度和
batch 配置一致，不在恢复时悄悄改变有效全局 batch。

## 正式 8 卡启动命令

在服务器 1 的终端中运行。启动前查看 `nvidia-smi`，仅在八张所需 GPU 空闲时启动；
本命令不停止或抢占已有任务。正式目录必须不存在或为空，入口会拒绝非空的新实验目录，
不要删除已有实验来绕过检查，也不要把短程测试 checkpoint 填入本命令。

```bash
set -euo pipefail
cd /home/user/liwei/GENMO-bumi-closedloop
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NCCL_CUMEM_HOST_ENABLE=0 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export BUMI_CLOSEDLOOP_90505_ROOT=/data0/user/liwei/datasets/bumi_music_umr70_mine_pass_90505_v1
export BUMI_KINEMATICS_PATH=/home/user/liwei/GENMO-bumi-closedloop/configs/bumi/bumi_kinematics_robot_retargeter_fe934_v1.json
/home/user/liwei/GENMO/.venv/bin/python -B -m torch.distributed.run \
  --standalone --nproc_per_node=8 --max_restarts=0 \
  tools/train_closedloop_stage1.py \
  --config configs/closedloop/stage1_server1_8gpu.yaml \
  --output-dir /data0/user/liwei/GENMO_outputs/bumi_closedloop_stage1_90505_warm_s350000_8gpu_v1
```

此命令在前台执行，不包含 `nohup`、`&`、tmux 自动创建或自动重启。不要预先将 shell
输出重定向到正式输出目录内部，以免启动前就把目录变成非空。

## 从本正式实验的最后一个完整 checkpoint 恢复

以下命令读取本正式目录的 `latest.json`，验证其 checkpoint 确实位于本实验的
`checkpoints/` 内，再以同一配置和同一目录恢复。它不引用本轮临时短训权重。
`latest.json` 只在编号 checkpoint 完整发布后更新；恢复报告写入 `reports/`。

```bash
set -euo pipefail
cd /home/user/liwei/GENMO-bumi-closedloop
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NCCL_CUMEM_HOST_ENABLE=0 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export BUMI_CLOSEDLOOP_90505_ROOT=/data0/user/liwei/datasets/bumi_music_umr70_mine_pass_90505_v1
export BUMI_KINEMATICS_PATH=/home/user/liwei/GENMO-bumi-closedloop/configs/bumi/bumi_kinematics_robot_retargeter_fe934_v1.json
STAGE1_RESUME_CHECKPOINT="$(/home/user/liwei/GENMO/.venv/bin/python -B - <<'PY'
import json
from pathlib import Path
experiment = Path('/data0/user/liwei/GENMO_outputs/bumi_closedloop_stage1_90505_warm_s350000_8gpu_v1').resolve()
record = json.loads((experiment / 'latest.json').read_text())
checkpoint = Path(record['checkpoint']).resolve()
if checkpoint.parent != experiment / 'checkpoints' or not checkpoint.is_file():
    raise SystemExit('latest.json does not identify a complete checkpoint in this experiment')
print(checkpoint)
PY
)"
/home/user/liwei/GENMO/.venv/bin/python -B -m torch.distributed.run \
  --standalone --nproc_per_node=8 --max_restarts=0 \
  tools/train_closedloop_stage1.py \
  --config configs/closedloop/stage1_server1_8gpu.yaml \
  --resume-checkpoint "$STAGE1_RESUME_CHECKPOINT" \
  --output-dir /data0/user/liwei/GENMO_outputs/bumi_closedloop_stage1_90505_warm_s350000_8gpu_v1
```

`--resume-checkpoint` 显式关闭配置中的 weights-only 初始化选择，恢复时仍核验原音乐
初始化权重的 lineage。训练已达到 step 50,000 时无需再次恢复；入口要求新运行终点
严格大于恢复步数。入口拒绝回退覆盖更晚的完整 checkpoint；若恢复后在下次保存前再次
中断，可重复同一恢复命令，新尝试使用独立 attempt 后缀，保留前次配置、日志和验证报告。

## 日志、checkpoint 与验证结果

查看逐步训练记录：

```bash
tail -n 20 -f /data0/user/liwei/GENMO_outputs/bumi_closedloop_stage1_90505_warm_s350000_8gpu_v1/train_metrics.jsonl
```

JSONL 包含 step、loss 及各损失分项、学习率、裁剪前梯度 norm、step 耗时和各 rank 的
峰值 allocated/reserved 显存。每次运行的摘要及恢复核验保存在 `reports/`；采样验证
保存在 `validation/`；`run_contract.json` 和 `config_from_*.yaml` 记录实际契约与配置；
`checkpoints/sXXXXXX.pt` 为完整可恢复权重，`latest.json` 指向最后完整保存的一个。

在服务器 1 的另一个终端运行 TensorBoard：

```bash
/home/user/liwei/GENMO/.venv/bin/python -B -m tensorboard.main \
  --logdir /data0/user/liwei/GENMO_outputs/bumi_closedloop_stage1_90505_warm_s350000_8gpu_v1/tensorboard \
  --host 127.0.0.1 --port 6006
```

在本地终端建立 SSH 隧道，再用本地浏览器打开 `http://127.0.0.1:16006`：

```bash
ssh -p 50030 -N -L 16006:127.0.0.1:6006 user@112.65.216.193
```

## 评估边界

周期验证固定每库 2 个 val 样本，四库共 8 个，使用固定 seed、DDIM 20 步、CFG 2.5。
采样只传条件，生成完成后才调用监督损失；同时检查物理空间已知前缀逐坐标保持、扩散
中间步骤约束、输出形状与有限值。这个小型固定集合用于监控训练，不是全量无偏评测，
也不能替代最终对完整 val/test 的质量评估。

新 90/5/5 清单包含一部分原 train 动作；原音乐 s350000 模型及匹配 stats 已见过其中
部分数据。因此本实验 val/test 衡量闭环适配变化，不能宣称为整个预训练模型的严格未见
音乐或未见动作泛化评估。训练损失有限、DDP 同步成功和前缀保持也不构成收敛质量、
动力学、部署或实机安全证明。

本轮临时短训目录 `/data0/user/liwei/tmp/genmo-stage5-acceptance-e81pvcyt` 已精确清理，
包含 3 个验收 checkpoint 在内共 27 个文件、7,739,639,675 bytes；本地证据收集临时目录
也已删除。上述机器可读报告保留各步 loss/梯度/耗时、各 rank 显存和权重摘要、恢复状态
及纯条件采样检查。结束时八卡均为 0 MiB / 0%，无训练进程，正式输出目录仍不存在。
本地 `tests/closedloop` 最终 141 项通过；重复恢复修复发生在真实 GPU 验收之后，仅改变
输出会话命名和保留方式，未修改训练/CUDA/DDP 数学路径。
