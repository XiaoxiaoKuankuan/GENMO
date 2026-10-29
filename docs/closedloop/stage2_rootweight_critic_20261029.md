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
