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

## 2026-09-30 服务器1真实闭环结果

训练和恢复均使用GENMO提交`34f8267`、GMT提交`e9349c8`；Actor由Stage1
`s350000.pt`仅权重初始化，新Critic与优化器从零建立。使用单张RTX6000D，GMT
ONNX和PhysX仍CPU执行。main/resume期间114个执行源码文件不变，源码清单SHA为
`997acf54ea59783111ef4fb3b7ed8fa44f88df500ec8b71dc5c9dd593b4c8d12`。

- 主训练64条上层转移/1600音乐控制步，Critic20步、最终保留Actor1步，BC计算一次
  batch=2、权重.1、loss=.38322849898。PPO-only梯度范数26379.86328，加入BC后的
  范数26379.67578，再裁剪至1；denoiser/history/prefix/music四模块均有PPO梯度。
- Critic全批MSE由600.74414降至446.57974，EV由-.0154505升至.0761981。
  预测值范围1.2617～6.7621、固定return范围.5938～30.4938，拟合仍弱，不称收敛。
- Critic更新时Actor不变；Actor更新时Critic不变；原资产、GMT策略、归一化及运行
  参数指纹不变。训练与恢复退出均正常，后端无强制关闭。
- checkpoint后新进程完整恢复Actor、Critic、两优化器、RNG及采样器；实际Actor lr
  为1e-9、Critic lr为1e-4，旧Buffer丢弃，新物理session为
  `76ad33cc-493b-453a-a8b1-dd1298e83e85`。物理环境重新reset，不恢复PhysX内部
  状态；恢复仅采16条新转移，不再次优化。
- 恢复16条包含425个音乐控制步，其中一次实际变长为50步，其余25步；记录按真实
  m累计，未硬编码成400步。主轮和恢复的零更新log-prob/ratio差均0，独立Gaussian
  密度核对最大差1.819e-12。
- 完整只读CLI不启用allow-incomplete，32项全部通过、stage9_passed=true。固定GAE
  与return独立重算最大误差3.695e-13。审计核对执行记录、奖励积分、固定targets、
  梯度/概率实测报告、候选选择、checkpoint实际lr、恢复身份及累计预算；不重新跑
  网络推理，也不把本次审计说成再次重算全部物理分项。

### 同批学习率实测

| lr | mean joint KL | P95 joint KL | 最大joint KL | 实际改变参数比例 | 非零梯度但参数未变比例 | 结果 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1e-9 | .002471652 | .02141519 | .07382346 | 53.0567% | 45.6615% | 选中 |
| 3e-9 | .030387565 | .26293872 | .94914966 | 82.9380% | 15.0582% | KL超限 |
| 1e-8 | .339809884 | 2.94981279 | 10.61263856 | 96.9312% | .7270% | KL超限 |

统计共有216,616,736参数元素，其中211,506,876个梯度非零。“实际改变参数比例”
以全部参数为分母，“非零梯度但参数未变比例”以非零梯度参数为分母。1e-9实际改变
114,929,671个元素，L2=1.52298e-5，最大实际变化1.86265e-9。按模块的改变比例为
denoiser52.4833%、music81.6054%、condition-presence76.2367%、history77.5210%、
prefix95.7071%。有梯度却未改变是实测现象，不能仅凭该计数逐元素区分Adam epsilon
抑制与FP32舍入；报告的ULP是实际已舍入变化，不是理论未舍入步长。

1e-9末两个去噪步的平均KL分别.02397075和.02532284，占全链KL的99.7179%；
每条去噪链20步KL之和再对样本平均为.04943304。当前.02门槛针对每内部转移自由坐标求和后的全批、
全去噪步平均，并不保证每条样本、每个去噪步或整条链都低于.02。

所以1e-9作为当前随机核下的一次保守验收值有实测依据，并非完全不更新；但它尚非
长期合理/最优学习率。3e-9和1e-8本批不合格，不能直接提高并宣称安全。下一次有限
实验应细化1.5e-9、2e-9附近候选并在新批次复核，再考虑多轮KL约束下的步长控制。
这是后续建议，本轮没有把未试验的中间值说成通过，也没有追加第四次尝试。

### 执行对照、预算和产物

A/B/C均同一AIST++样本、seed1729，每组4决策/100控制步/2秒音乐，均无执行终止
或拒绝。奖励分别为A原确定性5.84733030、B更新前随机5.85255513、C更新后随机
5.74403202。C低于B，因此只证明更新后仍可执行，不能证明舞蹈质量或收益改善。
本批仍是快速覆盖四库加长Mine片段的单环境验收，并非平衡训练集；实际前缀仍超
Stage1 P<=18的主要训练覆盖。未做长期、多环境、未见音乐、硬实时或实机验收。

完整主轮journal455次mutation/170次advance/2350控制/9400物理步，恢复93次mutation/
41次advance/625控制/2500物理步。共享预算累计108生成、2975控制、11900物理、3次
候选优化尝试；最终仅保留1次Actor更新。GPU退出检查8张均0MiB且无compute进程。

```text
服务器正式验收目录：
/data0/user/liwei/GENMO_outputs/closedloop_stage9/reward_v2_train_lr_20260930
共享账本：
/data0/user/liwei/GENMO_outputs/closedloop_stage9/reward_v2_train_lr_20260930_budget.json
完整checkpoint：
reward_v2_train_lr_20260930/checkpoints/stage9_000001.pt
```

checkpoint为2,602,828,034字节，SHA256为
`3a5e54da5a7e8fa14693d74cb65a50e956e1ef3be05101eec5a2e46b55b3ab3c`。
138个原始文件共3,208,671,330字节，artifact_manifest.json保存逐文件SHA；大权重、
去噪链、SQLite和rollout保留服务器；本地回传的22份JSON/YAML报告逐SHA核验全部
一致，另116份原始文件仅在服务器保存，详情见local_report_verification.json。正式
证据按本轮用户要求留存，不作为临时pytest数据清理。

本地完整集成269 passed；新增校准模块31项及入口/训练器最终联合53 passed（有
重叠不相加）。Ruff F/E9、diff通过；测试禁GPU/bytecode/cache且临时目录自动删除。

复核命令（输出必须使用新的文件名，避免覆盖正式审计）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/user/liwei/GENMO/.venv/bin/python -B tools/eval/audit_closedloop_dppo.py \
  --run-dir /data0/user/liwei/GENMO_outputs/closedloop_stage9/reward_v2_train_lr_20260930 \
  --budget-file /data0/user/liwei/GENMO_outputs/closedloop_stage9/reward_v2_train_lr_20260930_budget.json \
  --output /tmp/stage9_v2_independent_audit_new.json
```
