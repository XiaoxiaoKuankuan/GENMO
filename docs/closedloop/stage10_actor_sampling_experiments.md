# Stage10 Actor 与去噪步抽样实验

本页说明 Actor 计算重构、独立去噪步抽样目标和有限规模验收的使用边界。实现维护在
`feature/bumi-music-closedloop`，GMT 维护在 `feature/bumi-frozen-gmt-backend`。
数值诊断不能代替真实物理闭环，也不能用一次 KL 接受证明训练收益。

## 三类变化

| 类型 | 实现 | 验收边界 |
|---|---|---|
| 性能改动 | 联合权重梯度 GEMM、条件缓存直达、阶段设备断言、参考窗口增量更新、只读列式证据消费 | 对相同采集链验证损失、所有梯度、Adam、输出与 KL；完整奖励和执行证据保留 |
| 数值执行合同 | `fp32_fast`、`compiled_blocked64_fp32`、TF32/BF16 与强制 SDPA 后端 | 实际采样与重算必须自身一致；新合同重新采样，不改写旧概率，不透明恢复旧 run |
| 算法与采集合同 | PPO 每链抽样 4/8 步；`bounded_reference_wait.v1` | 抽样是目标估计，不能冒充完整 20 步计算；边界等待经用户授权，必须单列物理、奖励、预算和墙钟 |

## Actor 数据与梯度路径

`batch_execution.py::_SampleLinear.backward()` 的 `joint_gemm` 直接执行
`gradient.reshape(-1, O).T @ input.reshape(-1, I)`，不先生成 `[B,O,I]`。
`sample_bmm` 保留为参考；分块、局部 FP64、补偿 BF16 保留为显式诊断候选。
真实数据曾在 `denoiser.final_layer.fc2.weight` 发现相消梯度超差，所以提供精确模块名
`weight_reduction_overrides`。只有这个窄投影可以显式选 `joint_gemm_fp64`，其余骨干
继续联合 FP32；这不改变动作前向、old probability 或概率门槛。

`policy.prepare_conditions()` 在快速合同中批量编码音乐、历史和前缀。
`ConditionGraphCache.prime()` 对当前 optimizer minibatch 的唯一链建立条件图，20 个
内部转移只读取准备好的条件及约束张量。临时叶子上的条件梯度归并后回传原始编码图，
不能复用采样时的 detached 编码，也不能跨 optimizer step 复用旧条件。
GRU 的无效历史不更新状态，全空历史仍为零。原 `fp32_reference` 标量条件路径保留。

CFG 保持动作和 contact 两个头的 `2B` 合批。固定网络块只对内部尾行补零并裁掉，
有效行均来自真实输入；不复制环境、不把补零算作采样或梯度分母。
编译候选的采样和学习使用同一有梯度前向，再于无梯度边界 detach，以避免两种融合
归约使末端低方差概率漂移。BC 的 train/dropout/RNG 路径保持原 eager 前向。

## 独立的 20 / 4 / 8 步 PPO 对照

行为生成始终执行完整 `T=20` 步，每条链保存全部状态、old mean/std 和 old log-prob。
实验配置 `denoising_steps_per_chain=K` 只改变 PPO 对内部转移的无放回均匀抽样。
每个内部步入选概率为 `K/T`，因此完整目标

```text
(1 / (N*T)) * sum_chain sum_step loss(chain, step)
```

使用下面的无偏估计：

```text
(1 / (N*T)) * sum_chain sum_selected (T/K)*loss
= (1 / (N*K)) * sum_chain sum_selected loss
```

去噪折扣使用原始步号。全局归一化后梯度做 SUM，不能再乘除一次八卡数。
BC 频率、完整链 optimizer minibatch、epoch 和更新次数上限不随 K 改变。
种子来自可恢复的根 Actor RNG，再按全局链身份派生；保存选择 SHA 和步号直方图。
K=20 不消费额外抽样随机数。不同 K 的梯度差异属于估计方差，等价性对照应分别使用
同一 K、同一抽样计划的 B=1 参考。

PPO 当前前向得到的抽样 KL 只用于软停止估计。最终硬验收仍对全部有效链、全部
20 步重新计算联合 KL；不拼接不同参数版本的缓存，不使用抽样 KL 替代完整 KL。
拒绝时恢复 Actor、Critic、Adam、BC 和训练 RNG 状态，实际物理资源不退款。

## 世界级轮末等待

每卡固定 64 个环境，1024/2048/4096/8192 档分别让每环境采集 2/4/8/16 次决策。
异步终止会造成各环境不同步。经用户明确允许，已完成额度的环境可在共同边界前
继续执行现有参考，最多 100 个控制步，同时受剩余音乐和有效参考长度约束。
这段控制及四子步物理证据先持久化，再 ACK；奖励、执行时长和消耗并入末条转移。
到达等待或参考上限按行政截断记录 bootstrap，不能伪报物理终止。

状态身份、`fragment_tail`、边界等待奖励/控制/物理/墙钟和预算上界均显式记录。
部署模拟到达仍使用 `modeled_deployment.v3`，磁盘慢或 batch 变化只增加墙钟，
不修改模拟 P。断点禁止静默更换边界合同和等待上限。

## 服务器 1 有限验收入口

下面只运行一轮，并进行初始/结束固定评估及完整保存；不会启动长期训练。
仓库路径按实际隔离目录指定，输出必须为全新目录。默认 Actor LR 仍为 `5e-9`。

```bash
python -B tools/validate_stage10_scale.py \
  --config configs/closedloop/stage10_8gpu_server1_scale1024.yaml \
  --gmt-repo /path/to/legged_lab_gmt \
  --output /path/to/new_acceptance_directory \
  --precision-mode fp32_fast \
  --numerical-variant compiled_blocked64_fp32 \
  --sensitive-output-fp64 --rounds 1
```

独立实验显式添加 `--denoising-samples 4` 或 `8`；有限学习率对照显式传
`--actor-lr`，缺省不会扫描或自动降低。恢复使用同一目录、相同合同及
`--resume --rounds 2`，不得把不同目标或精度候选接到同一 run 上。

固定数据工具 `verify_stage10_gradient_reductions.py` 对不可变真实 rollout 验证完整
梯度和非空 Adam；显式精度模式会用同一真实条件重新生成该行为合同的诊断链，
报告必须与直接复用原链的结果区分。`benchmark_stage10_compute_blocks.py` 分别测
gen64、fb128 和 KL256，预热后报告最慢 rank 的 P50/P95。

## 结果解释

不得把编译首次启动、初始化固定评估或有限退出强制 checkpoint 混成普通轮。
同时保留完整墙钟和分阶段时间，嵌套计时不可重复相加。完整保存、归档队列背压
与退出清空也要报告，不能只展示前台 enqueue。

最终规模选择需要结合实际有效 Actor 更新数、覆盖率、每秒真实转移和固定任务奖励。
只使用更多显存、参数有变化或 loss 下降，都不能单独证明策略改善。
本页不将未通过的 BF16/TF32/融合候选标为生产可用；实际通过与失败结果写入
`记录文本.md` 及本次服务器报告。
