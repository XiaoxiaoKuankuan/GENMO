# 奖励 v2 的有限训练闭环与 Actor 学习率校准

本文记录奖励 v2 的 Critic、DPPO、checkpoint 和重启恢复验收。该实验与此前两轮
reward_v2_collect 是不同任务：保留原采集账本，创建独立训练验收账本；本轮所有
候选、对照和恢复共享该账本，不通过更换输出目录重置消耗。

## 本轮实现

- 同一64条真实执行转移生成固定回报目标；Critic仍独立训练20步，并新增全批
  更新前MSE和explained variance，避免把不同随机mini-batch的loss直接作前后比较。
- Actor的DPPO和原配对监督只反传一次，梯度裁剪到1。候选学习率
  `[1e-9,3e-9,1e-8]` 分别从完全相同的Actor参数、buffer和AdamW状态出发，复用
  同一固定梯度；不重新采集、不重算优势、不重复抽取BC样本。
- 每个候选实际optimizer.step之前预占一次iterations；最多三次，最终只保留
  满足原平均内部联合KL不超过0.02且参数实际发生变化的最大候选。
- 保存每候选的按去噪步KL、参数实际变化数量/比例、L2、最大变化和FP32 ULP
  诊断。每个完成候选立即原子落盘，失败时也保留已经得到的证据。
- 选中状态同时包含Actor和AdamW状态；未通过候选不累计进最终权重。所有候选
  失败或过程异常时恢复初始状态，不发布合格checkpoint。
- checkpoint保存selected_actor_lr和optimizer_attempts；配置保留起始actor_lr及
  候选规则，恢复时核对optimizer实际lr，不能把起始配置值误当恢复后的学习率。
- CFG和Critic步数/batch从已验证配置显式传入；本轮仍是有限单环境验收，未扩展
  为长期训练器，也未修改奖励、噪声、free mask、joint_sum、GMT或终止阈值。

## 为什么原先需要1e-9

旧v1两轮不同rollout中，lr=1e-6时平均内部联合KL约2964.814，lr=1e-9时约
0.0019376；二者不是严格同batch学习率曲线。第二轮约99.709%的KL总量来自最后
两个去噪步，其实际标准差均为0.001。这两步自身的平均KL约0.01896、0.01968；
不能把全20步的平均值误称每步都非常小。

方差固定时，高斯KL为 `0.5*sum_free((delta_mean/sigma)^2)`。标准差0.001、约
3000个自由坐标时，单步KL=0.02只容许约3.65e-6的均值变化RMS。约2.17亿参数的
网络可以把很小的参数变化累积成明显的输出变化，实际敏感性需同batch测量。

第一次AdamW更新（无weight decay）经偏差校正后约为
`delta_parameter=-lr*g/(abs(g)+epsilon)`。全局梯度裁剪同时缩放分子和分母，
不等于把最终参数步长按原梯度范数同比缩小。官方算法见
[PyTorch AdamW](https://docs.pytorch.org/docs/main/generated/torch.optim.AdamW.html)。

当前所有去噪样本反传结束后才执行一次optimizer.step；更新前ratio始终为1，
所以首步PPO clip不是参数位移硬约束。真正限制候选的是实际学习率和更新后解析KL。
优势均值约零会使更新前PPO loss数值接近零，不意味着其梯度为零。

也不能照搬开源DPPO配置中的学习率：其当前公开PPO实现对log-prob先裁剪、选取
reward_horizon后在动作/时间维取平均，并提供独立采样/概率标准差参数；本工作树
保持全部有效自由坐标的真实joint_sum及相同采样/重算标准差，两者的数值尺度不同。
此处只说明实现区别，不将公开实现的处理悄悄引入当前概率契约。来源：
[DPPO公开PPO源码](https://github.com/irom-princeton/dppo/blob/main/model/diffusion/diffusion_ppo.py)、
[官方配置说明](https://github.com/irom-princeton/dppo#key-configurations)。

FP32在参数0.1附近的间距约7.45e-9，在0.01附近约9.31e-10；1e-9的Adam步可能
让部分中大参数舍入不变。因此hash改变只证明部分参数变化，必须记录实际改变比例。
本轮ULP诊断针对真实FP32变化，不把它称为未舍入的理想Adam步。

## 解释边界

同批校准只回答本次固定梯度下的步长、KL及数值分辨率。最大合格候选不等于长期
最优学习率。不得通过改联合概率为均值、抬高仅重算时的sigma或缩小free mask来
让候选过门槛。未来若改变噪声、可训练模块或去噪组织，需重新采集和单独对照。

真实训练、恢复、独立审计与候选测量结果将在运行完成后追加，未运行阶段不预写通过。
