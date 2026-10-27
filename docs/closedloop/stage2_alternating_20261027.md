# Stage2-A 原SuperMimic交替正式训练

用户2026-10-27最新授权替代旧物理方向放行流程。本入口是正式独立实验，不表示旧WM反向问题已消失或真实质量已改善。旧Oracle、逐候选真实信任域和WM响应研究代码保留，不在本入口调用。

## 固定合同

GENMO缓存与SuperMimic Actor及Actor normalizer冻结；独立Adapter保持2654→512→512→256→270、三层SiLU、十帧SO(3)残差、根XY关闭。采用现有已运行的灵活参考投影，关节速度/加速度1rad/s、10rad/s²，踝0.5rad/s、5rad/s²，根高度0.12m/s、1.2m/s²，根旋转0.3rad/s、3rad/s²。上述是附加预算，不是硬件极限。

原SM六项及action-rate不变；music=1e-4，软关节限位关闭，不加body_w Huber或接触损失。原q、qd、身体姿态和局部根位移精度及参考正则保持；八组统一precision weight=0.446714796317961，其余权重见配置。训练只用64首真实GENMO生成缓存，validation选best，evaluation仅报告。

## 每轮算法

先采64个向量步填充真实Replay。随后每轮累计1/UTD=2.5，交替采2/3个真实向量步。Actor动作加标准差0.1高斯探索后clip[-20,20]；该实际动作同时进入物理、奖励、Replay和下一观测的上一动作。评价无探索。

独立WM从原预训练权重初始化。只有它的in_norm/out_norm替换回原RunningNorm，逐张量继承mean/var/count/eps/max_count，不清空统计、不修改冻结bundle。每次真实采集后按原SM inputs/targets更新，达到原max_count则沿用原冻结行为。

每轮WM做10次batch2048、H8真实动作递归监督：

    L_WM = (1/B) sum_b sum_t mask_bt * mean_d [(error(s_hat_bt,s_real_bt)/std_d)^2]
    std = maximum(original_out_norm.std(), original_SM_error_floor)

第一步真实状态，后七步使用预测状态，八步全部监督；日志wm_loss=L_WM/8，反传使用时间求和原式。没有一步额外损失、长窗口额外损失、切向或奖励响应损失。WM grad norm clip=1。

前100轮只采集并训练WM；第101轮开始，每轮从真实Replay取1024起点，以32 microbatch做H8/H16/H32完整BPTT。每步重算Adapter、投影、Actor和单步WM；内部历史、状态、参考和音乐图保留。Actor/WM/Critic参数冻结但输入梯度保留。使用非重入梯度检查点，累计原SM软终止存活率，不额外叠加想象硬终止门禁；歌曲结束仍不bootstrap。

    J = sum_t gamma^t * c_t * r_t + enabled * gamma^H * c_H * V(x_H)
    c_(t+1) = c_t * (1 - soft_termination_t)

Adapter Adam LR3e-5、betas=(0.9,0.95)，原SM norm clip=1及value clip=0.1。每个有限Adam完成就增加optimizer计数；参数/Goal/动作变化只报告，不作候选物理门禁。

Critic开启时，复用2835维特权输入并使用原SM Critic/RunningNorm；每次Adapter更新后，对H+1想象输入更新norm，重新计算V及lambda-return：

    R_H = V_H
    R_t = r_t + gamma*(1-d_t)*[(1-lambda)*V_(t+1)+lambda*R_(t+1)]

gamma=.99、lambda=.95；Critic LR1e-4，原版5 epochs×4 minibatches，MSE拟合stop-gradient标签。Critic指标明确为想象标签，不冒充真实MC校准。关闭时无终端项、无Critic优化。

## 八组与记录

|GPU|BPTT|Critic|Seed|
|---|---:|---|---:|
|0|8|关|42|
|1|8|开|42|
|2|16|关|42|
|3|16|开|42|
|4|32|关|42|
|5|32|开|42|
|6|32|关|43|
|7|32|开|43|

每组2048真实环境、1000次Adapter更新；不同GPU没有通信，共享只读资产，在线模型/优化器/Replay/RNG独立。启动器实时检查占用，仅启动空闲卡，不终止其他任务。

每轮training.jsonl记录真实采集数/奖励、WM更新和策略更新；首个及每10次Adapter更新记录固定多歌曲Goal/动作变化。第50/100/200/400/700/1000次进行固定真实整歌validation和evaluation，报告六项、精度、根RMSE、jerk、真实接触脚滑、完成率和音乐；失败前缀不能冒充完整歌曲。评价结果不阻塞下一次Adam。

每50次、首次Adapter更新和WM预热结束保存完整checkpoint，含三网络/优化器、WM/Critic归一化、动态Replay、音乐/残差状态、时钟及RNG。为约束服务器磁盘，保留最近两份完整及validation best，全部编号轻量权重和报告保留。latest/best使用JSON指针，不覆盖旧实验。恢复须新输出目录、同算法配置；物理从新episode开始，不声称恢复求解器隐藏状态。

运行时NaN/Inf、OOM、仿真异常或投影不可行保存现场并退出对应组；不做物理方向审计、候选参数缩放、物理回滚或WM修复循环。

## 实际验证与启动记录

- CPU完整回归80 passed/31.55秒，新6项回归覆盖原式H8 loss与梯度、统计继承/冻结隔离、探索动作一致、可选Critic/GAE、soft termination。
- 配置矩阵、八组共享奖励/预算、AST、git diff --check通过。未运行新的方向研究或pilot。
- 实际服务器根目录、源码、PID及首个有效Adam更新见下方交付快照，不把WM预热称为Adapter更新。
- 首次正式启动八组均完成1次Adam后，发现全局TF32破坏SO(3)小差分和QP精化：GPU0保存的2048提案24个证书失败，改回严格FP32重投影全部通过，最大角加速度超量从1.4161降至0.0001588，原容差0.0025未改变。修正正式入口全局TF32并从本组完整checkpoint恢复；未进行WM方向研究或放宽约束。首次失败根目录与checkpoint保留。
- r1恢复后的非零投影触发PyTorch `torch.func.jacrev` 与非重入检查点的saved_tensor_hooks冲突。现将投影留在检查点外保留原图，Actor与WM/奖励独立检查点；不改变隐式梯度或时域。新增真实SM活跃SO(3)分支回归，含Critic终端的有/无检查点参数梯度一致，7项定向回归4.97秒通过。全部r1完整状态仍保留，后续按本组恢复。

Server-1启动：

```bash
cd /home/user/liwei/bumi-closedloop-worktrees/adapter/GENMO
/data0/user/liwei/envs/genmo_adapter_runtime_20261026/bin/python -B tools/launch_stage2_alternating.py \
  --root /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/alternating_时间戳 \
  --session-prefix smalt_时间戳
```

完整恢复到新目录，使用本组自己的配置及latest_checkpoint.json中的路径：

```bash
CUDA_VISIBLE_DEVICES=对应GPU ISAACLAB_PATH=/data0/user/liwei/GENMO_assets/adapter_20261026/IsaacLab \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
/data0/user/liwei/envs/genmo_adapter_runtime_20261026/bin/python -B -u tools/train_stage2_alternating.py \
  --config 本组配置.yaml --output 新目录 --resume 本组完整checkpoint.pt --visualizer none --deterministic
```

## 八卡正式运行交付快照：2026-10-27 22:53:35 北京时间

八个进程、八个tmux会话及GPU compute PID逐一匹配，均已跨过100轮WM-only预热，持续执行真实采集、WM、Adapter及可选Critic更新；当前目录没有failure报告。每卡2048环境，目标各1000次Adapter Adam。以下为实际计数，包含本次实验兼容修复前的1～2次已完成Adam，不包含历史退化Adapter的10563次更新。

|GPU|BPTT|Critic|Seed|PID|Adapter Adam|WM更新|真实训练转移|Torch已分配峰值MiB|当时GPU占用MiB|
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
|0|8|关|42|262724|8|1100|692224|10629|18235|
|1|8|开|42|263025|8|1100|692224|8393|16181|
|2|16|关|42|266187|5|1070|677888|10956|16760|
|3|16|开|42|266228|5|1070|677888|10194|13807|
|4|32|关|42|266281|5|1070|677888|7362|12927|
|5|32|开|42|266367|4|1060|671744|7538|14169|
|6|32|关|43|266619|4|1060|671744|7362|11979|
|7|32|开|43|266896|4|1060|671744|8342|16765|

GPU瞬时利用率37%～43%。Torch峰值统计不含全部驱动、物理及缓存开销，不能与nvidia-smi瞬时占用混用。`/data0`当时剩余491GB。原`genmo-rolling-20261022_182720`会话保持不变。

当前根目录：`/data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/alternating_20261027_2226_r2`。
GPU N的独立运行目录为根目录下`gpuN/`，配置为`config_gpuN.yaml`，完整实际命令为`run_gpuN.sh`，日志为`gpuN.log`，tmux为`smalt_20261027_2226_r2_gN`（N=0～7）。`launch_manifest.json`保存所有完整命令；`delivery_runtime.json`保存本表原始状态、进程命令、cwd、GPU及tmux证据。

实际运行GENMO源码快照为`9e59b1e55ab37930ccea0097f0d1402a5100e329`，配套SM为`a0911e4bbbffe96e1d6bc842b29f4b01c2ebe7a6`。运行使用独立detached快照；交付文档后续提交不会热修改运行源码。恢复state中的`source=28d8c00...`是本次实验初始来源，当前运行源码应以`source_genmo`的Git HEAD和启动清单为准。

当前进程的实际启动命令（已经执行，不是待执行计划）：

```bash
cd /home/user/liwei/bumi-closedloop-worktrees/adapter/GENMO
/data0/user/liwei/envs/genmo_adapter_runtime_20261026/bin/python -B tools/launch_stage2_alternating.py \
  --root /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/alternating_20261027_2226_r2 \
  --session-prefix smalt_20261027_2226_r2 \
  --resume-root /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/alternating_20261027_2226_r1
```

最初八组从零Adapter和同一原预训练WM开始；`alternating_20261027_2226`及`_r1`分别保存上述TF32和检查点兼容故障的完整现场，修复后各自完整恢复，没有借用其他组在线数据或旧退化模型。八组加载后Adapter/WM参数SHA与对应保存值全部一致，三网络/三个优化器、WM/Critic统计、Replay动态字段及Python/NumPy/Torch/CUDA RNG齐全，详见`checkpoint_restore_evidence.json`。Critic关闭组优化器状态为空且没有执行价值优化，开启组每轮执行20次想象λ-return优化。

恢复后物理重新建立episode，旧Replay不跨reset拼接。当前各组`latest_checkpoint.json`明确指向实际已加载的本组完整恢复点，并标记`loaded_full_resume_checkpoint_current_weights_continue_in_memory`；它不声称覆盖随后尚未到50次保存点的内存更新。第50次保存会替换为本目录新完整checkpoint，首次validation评价后才产生best，当前未伪造best。

首次有效Adapter更新的GPU0证据：WM=1010，真实转移647168，参数变化L2=0.007569664，梯度范数0.026902754，128个多歌曲探针Goal RMS变化0.001833993、Actor动作RMS变化0.005618447，探针实际变化比例1。八组首次参数变化均非零，当前恢复后八组均又完成了新的Adam，不能把已有计数的加载当新更新。探索噪声与策略相对零Adapter的动作变化分别记录，避免把0.1探索当学习收益。

原WM统计count=200003584已达到原RunningNorm的max_count=2e8。代码按原版调用真实数据update，因此此次预训练统计保持原版封顶行为；未清空count、改变Actor归一化或擅自调整原SM规则。

最终完整CPU回归81 passed/33.35秒，真实SM权重未skip；活跃SO(3)约束加Critic终端的检查点梯度等价通过。已清理本轮精确测试临时目录，保留正式失败现场与训练证据。本地轻量交付证据：`/home/weili/bumi-closedloop-worktrees/adapter/alternating_20261027_2226_delivery/{initial,recovery1,current}`。

本次没有执行新的WM物理方向诊断或pilot。尚未达到50次更新，因此没有新的固定整歌评价结果，也没有证据宣称真实跟踪改善；后续按50/100/200/400/700/1000自动评价，validation选best，evaluation仅独立报告，效果好坏不作为逐更新门槛。
