# 根Tracking加权、H64与Critic八卡正式对照

2026-10-29，GENMO feature/stage2-supermimic-adapter；配套SM feature/genmo-worldmodel-bridge。

## 方法与奖励

用户指定根位置Tracking求和权重从0.5改为1.2，根姿态从0.5改为1.0。误差尺度仍为0.2m、0.3rad，身体位置/姿态尺度仍0.1m/0.15rad，身体四项权重仍1。RewardCfg默认仍为旧SM的0.5/0.5，新配置仅覆盖本运行实例。最大每步Tracking从0.1变为0.124；后续比较必须与本轮同配置零Adapter配对，不能拿不同权重的原始Tracking直接比较。

GENMO、Actor和Actor归一化冻结；网络2654→512→512→256→270、十帧独立有界补偿、根XY关闭、根姿态开启。仍无QP/SQP/IPM或导数投影。音乐1e-4、四项精度整体0.4467147963、根姿态附加角速度/重叠连续性软成本0.002/0.001；软关节限位、额外身体角速度Huber、世界根位置非饱和附加成本均保持关闭。真实根高度差0.2m及重力投影z差0.6终止规则不变。

WM使用原预训练结构/参数初始化，真实H8递归状态监督，10次更新/轮，batch2048，LR3e-5，统计继承原RunningNorm。Adapter更新完整H16/H32/H64闭环BPTT，无候选物理门禁、PPO或Critic-off的终端bootstrap。

## Critic实验

Critic-on沿用SM的2835维特权输入、256/128隐藏层与RunningNorm。用户要求启用，因此删除10月28日临时禁止Critic的入口异常，并把terminal_value接到实际开关。Critic参数在Adapter阶段冻结，终端价值对状态/Adapter保留梯度。

每步总奖励不超过0.124，所以gamma=0.99的无限时域上界是12.4。输出V=U-softplus(U-raw)，U=12.4；负值不设未经证明的下界，未使用detach或直通梯度。末层零权重配合解析bias使初始价值为0。该上界只防数值超过物理回报上限，不能冒充价值校准或可靠方向证明。

Critic以H+1想象轨迹GAE/lambda-return为主目标，并加入最新2～3个真实向量步的TD(0)锚定：y_real=r_real+gamma*(1-done_real)*V(next_real)。起点/末态由采集器reset前保存，实际动作和奖励完全一致，不拼接跨reset数据。真实失败和音乐结束不bootstrap；TD含自举，不叫完整MC。相对loss权重1，Critic batch2048、3 epochs、LR1e-4。100轮WM预热期间开启组同时用真实TD预热Critic，关闭组不执行Critic优化。两个数据源的loss/target/value上限接近比例分别记录，不设阻塞启动的校准门槛。

## 正式矩阵与预算

|GPU|BPTT|Critic|Seed|Adapter LR|目标更新|
|---|---:|---|---:|---:|---:|
|0|16|关|42|1e-5|2000|
|1|16|开|42|1e-5|2000|
|2|32|关|42|1e-5|2000|
|3|32|开|42|1e-5|2000|
|4|64|关|42|1e-5|2000|
|5|64|开|42|1e-5|2000|
|6|64|关|43|1e-5|2000|
|7|64|开|43|1e-5|2000|

每组2048真实物理环境，Adapter batch1024，microbatch256，WM batch2048/window8/每轮10次，真实动作噪声0，gamma0.99、lambda0.95。同seed的初始零Adapter和原WM一致，后续模型、优化器、Replay、RNG各组独立；Critic优化消耗随机数，因此不声称开启/关闭组的所有后续采样完全相同。训练集仍64首冻结GENMO生成缓存。

所有组从零Adapter/原WM新建，不恢复旧退化权重，也不复用旧奖励Replay。0/5/10次短片段，20/50/100/200/400/700/1000/1500/2000次整歌validation与evaluation，使用独立4环境、同初态、无噪声零模型对照；只用validation选best。完整保存每100次及评价点，保留zero/best/latest两份和所有编号轻量模型。

沿用现有连续两次整歌validation明显退化停止标准：根RMSE相对+25%、jerk+20%、Tracking相对-2%或完成率下降0.25；不新增逐更新物理回滚。2000为上限，触发退化或非有限保护时单组可提前结束，其他组独立继续。不能承诺所有组必然完成2000，也不能把训练次数当效果成功。

## 验证与启动

针对根权重/旧默认隔离、dt一次、输入梯度、H64有界Critic BPTT、真实终止TD及Critic-off路径，CPU定向回归17 passed / 10.88s。完整回归与Server-1实际启动、首次Adapter更新、物理评价及checkpoint恢复结果完成后追加。

完整CPU回归：150 passed / 50.05s，临时产物清理完成。

## Server-1实际启动与恢复记录

2026-10-29 16:19（北京时间）已实际启动全部八卡，无排队或跳过。运行源码固定为GENMO `0e23094dd062b828ee5b8a51a9660cb32a3d78b4`、SuperMimic `44cc31b77bc2dd1f998cec93933f812540dbd4d6`。输出根为 `/data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/formal_rootweight_critic_20261029_1619`，各组目录为 `gpu0` 至 `gpu7`，独立配置为根目录的 `config_gpuN.yaml`，完整命令见 `launch_manifest.json`。冻结运行工作树为该根目录的 `source_genmo` 与 `source_supermimic`；后续文档提交不改变已运行源码。

执行的启动命令（Server-1，adapter/GENMO开发树内）：

```bash
PYTHONDONTWRITEBYTECODE=1 /data0/user/liwei/envs/genmo_adapter_runtime_20261026/bin/python -B tools/launch_stage2_alternating.py \
  --root /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/formal_rootweight_critic_20261029_1619 \
  --session-prefix rootweight_critic_20261029_1619 \
  --config-dir configs/closedloop/stage2_rootweight_critic_20261029 --wait-for-free-gpus
```

|GPU|训练PID|tmux后缀（公共前缀rootweight_critic_20261029_1619_）|16:24已完成Adapter更新|真实采集转移|
|---|---:|---|---:|---:|
|0|418357|g0|18|739328|
|1|418382|g1|16|729088|
|2|418424|g2|10|694272|
|3|418509|g3|10|694272|
|4|418580|g4|7|684032|
|5|418625|g5|5|673792|
|6|421299|g6|6|677888|
|7|422338|g7|5|673792|

上述是采样时刻的运行快照，不是最终训练结果。八组首次Adapter更新前均完成647168条真实采集和1010次WM更新；首次参数L2变化均约0.00253，控制动作变化非零。开Critic的GPU1/3/5/7首次正式迭代分别执行24/48/96/96次Critic小批量优化，折扣终端价值非零；关闭组为0。日志中的Adapter计数不包含WM/Critic优化。GPU可见设备、PID argv/cwd、tmux、GPU占用和初始资产均已核验，见 `launch_verification.json`；无关tmux `genmo-rolling-20261022_182720`保持原状，旧实验未覆盖。

初期完整交替轮次中位数：H16约4.9～5.2s、H32约9.0～9.1s、H64无Critic约16.5～16.8s、有Critic约17.9～18.6s。这个统计包含采集、10次WM及Adapter/Critic计算，不含周期性checkpoint和独立评价；H64不是单次终态预测。16:22实测GPU进程占用约8.6～12.8GiB，各组仍运行；CUDA allocated峰值约7.2～8.3GiB，两者口径不同。

启动前64环境H64+Critic真实测试完成两次更新后恢复到第三次；保存/恢复前后优化器step分别为Adapter 2→3、WM 3→4、Critic 33→49，Replay恢复1472条、最终采集1664条，Critic上界/归一化、RNG和三网络/优化器均包含在完整checkpoint中。物理隐藏求解器状态仍按既有合同新开真实episode，不宣称逐位恢复接触状态。独立临时目录 `/tmp/genmo_rootweight_critic_gpu_20261029_1617` 已清理，摘要保留在正式运行根的 `startup_validation.json`。

## 第5次更新的初期物理对照边界

八组均完成独立4环境、相同歌曲/初态、无探索噪声、256控制步（5.12秒）的validation。下表是候选相对本组零Adapter的百分比变化；Tracking越高越好，其余误差越低越好。不是整歌结果，不用该短前缀的完成率判断整歌表现。

|GPU|Tracking|关节q误差|身体姿态误差|根RMSE|jerk|
|---|---:|---:|---:|---:|---:|
|0|+0.305%|-2.639%|-1.463%|-8.955%|+1.006%|
|1|-0.052%|+1.071%|+0.726%|-0.797%|+0.329%|
|2|+0.262%|-2.665%|-1.796%|-2.504%|+0.097%|
|3|-0.107%|+1.197%|+0.854%|+2.230%|+0.504%|
|4|+0.156%|-1.482%|-1.306%|+1.371%|+1.228%|
|5|-0.009%|+0.921%|+0.832%|-2.523%|+2.337%|
|6|+0.192%|-0.812%|-0.627%|-4.912%|+0.751%|
|7|+0.042%|-0.207%|+0.307%|-4.905%|+0.495%|

零Adapter此短validation的Tracking约0.114233/步、关节q RMSE约0.090652rad、qd RMSE约0.544726rad/s、身体姿态RMSE约0.101801rad、根RMSE约0.035854m。新权重下Tracking上限0.124/步，不能仍按旧满分0.1换算。

目前只能确认八组训练链路和首次物理评价正常；不能据五次更新宣布最终提升，也没有证据表明加Critic已经优于不加。Critic虽未超过12.4上界，部分估值已接近上界，且终端项及梯度较大，仍需持续看真实配对评价；本轮不把想象EV或上界本身称为校准成功。后续按既定20/50/100/200等更新点自动整歌评价与保存，validation选best，evaluation只报告。
