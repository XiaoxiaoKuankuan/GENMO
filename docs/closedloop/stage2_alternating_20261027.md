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
- 实际服务器根目录、源码、PID及首个有效Adam更新将在启动核验后追加，不把WM预热称为Adapter更新。
- 首次正式启动八组均完成1次Adam后，发现全局TF32破坏SO(3)小差分和QP精化：GPU0保存的2048提案24个证书失败，改回严格FP32重投影全部通过，最大角加速度超量从1.4161降至0.0001588，原容差0.0025未改变。修正正式入口全局TF32并从本组完整checkpoint恢复；未进行WM方向研究或放宽约束。首次失败根目录与checkpoint保留。

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
