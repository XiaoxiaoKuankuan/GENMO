# Stage2-A：简化奖励与H32/H64真实复验（2026-10-27）

本报告对应 `feature/stage2-supermimic-adapter`，配套 SuperMimic `feature/genmo-worldmodel-bridge`。本轮只做Stage2-A；GENMO、SuperMimic Actor和normalizer保持冻结，没有PPO、额外控制器或扩散更新。

**结论：代码、长时反传、真实采集和恢复验证已完成，但当前WM的H32/H64真实方向仍不合格。新目标下有效Adapter更新为0，八组正式训练没有启动。** 三轮有明确区别的独立WM修复均按150次监督预算退出，没有把监督次数算作策略更新，也没有继续无限监督。不能宣称第二阶段已稳定优于零Adapter。

## 1. 已完成的代码修改

| 文件 | 修改与理由 |
| --- | --- |
| `precision.py` | 删除额外`body_angular_velocity` Huber奖励。保留身体姿态、q、qd和短时局部根位移，比例保持0.30/0.25/0.20/0.10。body_w仍用于诊断及WM监督；不改原SM六项。支持H32/H64训练集梯度尺度标定。 |
| `reference_response.py` | 按真实长度分桶存储H8/H16/H32/H64；不截8步、不补齐。新增根w及原SM FK body_w响应差，独立尺度/权重。指定H32必须有完整可靠的H32标签；真实连续mask与接触/噪声可靠mask分开。 |
| `stage2a_physics.py` | 当前Adapter经同一投影和冻结Actor，采集同起点基线/重复/正/负真实响应，混合持续1/8/32步干预；只用于采集，不是部署控制器。训练难例分数增加qd/body_w真实误差。 |
| `dynamics.py`、`vector_learning.py` | 原H8状态监督外加入H16/H32各0.25权重，H64只验证。响应监督不增加预测头。非重入梯度检查点保留整个状态、控制历史、残差和音乐记忆计算图；WM参数冻结时保留输入梯度。 |
| `vector_replay.py` | 按实际名义支持选长时真实起点，禁止跨reset或歌曲尾部；缺少H32支持时不默默回退H8。 |
| `long_horizon.py` | 同起点、同完整参数扰动的三路径诊断：真实状态单步WM、真实动作连续WM、Adapter/Actor/WM完整闭环。逐H1～H64记录q/qd/w/body_w、原六项、累计目标及真实接触。 |
| `parameter_audit.py`、`physical_trust.py` | 方向报告包含当前WM参数身份、实际奖励项和完整候选参数差；失败真实候选仍保存完整响应窗口。 |
| `stage2a_entry.py` | 初始正式BPTT32；Critic完全冻结且终端价值关闭。新增有界`wm-repair`模式，150次WM监督后完整保存退出；本模式不会执行Adapter优化。保存/恢复全部学习状态和真实响应桶。 |
| `tools/audit_stage2a_reference.py` | 新增三路径`trace`、独立WM checkpoint替换及H32/H64显存/梯度标定。替换WM仅用于同候选对照，不能当作断点续训。 |
| `tools/compare_stage2a_wm.py` | 对同一份已保存的独立真实256×64标签比较多个WM；记录标签SHA与各模型参数SHA，排除随机验证批次变化造成的假改善。 |
| `tests/closedloop/reference_adapter/test_long_horizon.py` | 长窗口、无额外body_w奖励、w/body_w监督梯度、H32/H64检查点完整参数梯度等价、终止后标签排除及无有效样本报告None的回归。 |
| `configs/closedloop/stage2a_*.yaml` | music=1e-4、软限位关闭、根XY关闭、H32优先、独立长窗监督/响应配置。八份配置使用新目标H32训练梯度标定，不代表已获物理放行。 |

核心提交：`f0b9359`（长时重构）、`15701ff`（持续干预与完整可靠窗口）、`bfb6cbb`（连续mask与2048环境覆盖）、`b715720`（固定标签比较与H32权重）。SM为`a0911e4`，只复用一次FK提供诊断，没有修改Actor权重或六项公式。

仍保持2654→512→512→256→270、三个SiLU隐藏层、10×27独立提案、SO(3)、50Hz控制/200Hz物理。没有向Adapter添加根世界位置、根速度、xi、接触标签或未来真实状态。

## 2. 监督公式与数据隔离

原WM已有w/qd输出。响应组为根局部位移、SO(3)姿态、q、qd、根w及SM原FK的body_w。对正负真实动作序列分别连续预测：

```text
Δreal_g(t) = g(s_real_plus(t)) - g(s_real_minus(t))
ΔWM_g(t)   = g(s_WM_plus(t))   - g(s_WM_minus(t))
L_response = Σ_g weight_g mean_valid[(ΔWM_g - Δreal_g) / scale_g / action_RMS]^2
L_WM = L_state_H8 + 0.25 L_state_H16 + 0.25 L_state_H32 + 0.1 L_response
```

action_RMS使用实际正负动作差的前缀RMS，下限1e-4；动作差过小、初态不一致、重复性不足和接触切换样本不充当可靠切向标签。此处尺度是学习归一化，不是验收门槛。

| 响应组 | 独立尺度 | 组权重 |
| --- | ---: | ---: |
| 根位移 | 0.05 m | 0.15 |
| 姿态 | 0.1 rad | 0.15 |
| q | 0.1 rad | 0.35 |
| qd | 2 rad/s | 0.35 |
| 根w | 0.5 rad/s | 0.10 |
| body_w | 1 rad/s | 0.10 |

76份真实GENMO生成缓存中64训练、4WM validation、4audit、4evaluation。FineDance189属于audit，因此只用于诊断，**没有混入WM训练**。难例由训练歌曲真实qd/body_w误差筛选，保留60%困难/40%均匀采样。H64没有进入本轮监督损失。

资产与歌曲SHA清单保存在证据根目录的`assets_64_sha256.json`和`tasks_64_generated.json`。原始动作数据只用于首帧初始化，未来参考均由冻结GENMO实际生成。

## 3. 实际执行的有限WM修复

服务器证据根目录（下文记作R）：

```text
/data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/stage2a_projection_20261027_01
```

| 独立运行 | 环境 | 真实转移 | 响应对 | 训练歌曲覆盖 | WM更新 | 新Adapter更新 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `long_WM_repair01`，单步冲击 | 128 | 190,464 | 480 | 56/64 | 150 | 0 |
| `long_WM_sustained02`，持续1/8/32步 | 128 | 190,464 | 480 | 56/64 | 150 | 0 |
| `long_WM_full2048_03`，持续干预及完整歌曲覆盖 | 2048 | 3,047,424 | 3,840 | 64/64 | 150 | 0 |

三轮分别从原始冻结资产复制独立WM；没有延长前一失败进程。第三轮按角色计数：train=2,670,592、train-safety=49,152、validation=327,680。3,840响应中完整可靠到H8/H16/H32/H64的数量为1,445/835/474/231。H32占12.34%，H64占6.02%，不能把总响应对数当作长时可靠标签数。

第三轮进程PID=3550315，GPU2，tmux=`stage2a_long_WM_full2048_03`，退出码0，最终状态`bounded_long_WM_repair_complete_zero_Adapter_updates`。此PID和会话是**已结束运行的历史身份**，不是仍在训练。

最后完整checkpoint：

```text
R/long_WM_full2048_03/supervised_WM_no_Adapter_update_000000_1793100188510272551.pt
```

其WM参数SHA256：`59fe490ca74079f0117b317321d3a0a07e3ec85f9a5b92023b898745e74b6f60`。Torch峰值6682.26MiB，日志记录GPU进程显存8720MiB。这是2048环境WM监督开销，不能冒充2048样本完整Adapter反向的显存。

## 4. H32错误从哪一步发生

统一使用旧pilot的同一个小幅完整Adapter参数扰动作为诊断探针，与精确零Adapter比较；每次仅换WM，重新执行真实物理。这个旧权重只用于同扰动对照，**没有作为新策略初始化，也不是本轮新H32 Adam候选**。因此结果证明该已知方向仍不可信，不能推导所有可能方向都失败。

最新WM、FineDance189、真实零策略前缀50步、H32：

| 路径 | H32累计完整目标变化 | 首次可信累计反向 |
| --- | ---: | ---: |
| 真实物理 | −0.000311375 | — |
| 每步输入真实状态、历史和动作的单步WM | +0.000717402 | 第4步 |
| 使用真实动作序列的连续WM | +0.000130177 | 第5步 |
| Adapter＋冻结Actor＋WM闭环 | +0.000821114 | 第3步 |

独立方向入口复跑的闭环值为+0.000821352，真实值−0.000311375；两个运行的微小数值差保留，不改写为完全相同。重复回报差4.768e-7，实际采用10倍噪声门槛4.768e-6，真实退化仍约为门槛65倍。

这里每步重置为真实输入的路径仍反向，说明**局部状态/动作响应偏差已经足以改变目标差分符号**，不是只有递归累积才出问题。长期状态RMSE小也不能证明小幅参数更新的收益差分可靠。

原SM项加action-rate的H32实际变化为−0.000770092，模型预测+0.000110626。新增q精度实际改善+0.000766939，但qd精度实际恶化−0.000279900，模型只预测−0.000076655；局部根位移实际恶化−0.000186330，模型只预测−0.000069423。奖励简化后这些真实损失仍存在。

原六项中body_angvel的累计折扣指数项变化：真实−0.0358321、闭环WM−0.00410928。该分项原权重为1，乘一次dt后分别为−0.000716642和−0.000082186；WM明显低估原SM身体角速度奖励的损失。删除额外body_w Huber并不会消除原SM合法速度奖励的预测偏差。

求解器接触集合首次变化发生在第4步；候选与零策略接触集合首次分歧在第8步，后续11/12/15～19步也有差异。这与速度误差尖峰及单步路径第4步反向相伴。**接触转换是需要继续查证的原因线索，不是已被证明的唯一原因**；布尔接触相同不保证接触力或求解器内部状态相同。当前可靠响应会排除这类切换，不能靠放宽mask把它们伪装成可靠导数标签。

完整H1～64曲线和各步六项见`long_trace_full2048_150/three_path_H1_H64.json`，真实状态/动作/接触见同目录`three_path_physical_tensors.pt`；提取后的接触与分项证据为`FineDance189_H32_diagnostic_details.json`。

## 5. 最新物理方向验收

4首audit歌曲、两个真实零策略前缀，共8个起点。每起点包含零策略、重复、完整候选、反向候选；noise floor=1e-6、重复状态最大差门槛=1e-4，其余状态门槛不变。

| 时域 | 高信噪比反向例 | 重复性 | 本轮结论 |
| --- | ---: | --- | --- |
| H8 | 2/8 | 通过 | 未通过，只是诊断 |
| H16 | 1/8 | 通过 | 未通过 |
| H32 | 1/8 | 通过 | 未通过；FineDance189/prefix50 |
| H64 | 1/8 | 未通过 | 未通过；AIOZ反向及另一起点重复性不足 |

H64的AIOZ/prefix50真实目标变化−0.002557278、预测+0.001072884。另一个AIOZ/prefix0重复状态最大差0.000258923，超过原1e-4门槛，未放宽。

H32八起点总收益虽为正，也不能覆盖可信反向。之前“关闭软限位但仍保留body_w Huber”的H8通过结论属于旧目标，**不沿用为本轮新目标通过**。

独立validation参考响应的32对中，完整可靠H32只有6对，H64只有3对。H32根姿态和根w分别仍有2个反向响应；H64样本太少且多个字段反向，不能宣称长期响应已经校准。

## 6. 完全相同真实标签下的预测能力

窗口来自独立validation Replay，共256×64；窗口SHA256为`6f4c6fb797544f125bf0f48f3b29713090fd873d68d1a6083785e743062b574b`。每个WM使用完全相同的状态、动作、xi及真实终态标签。位置/旋转是批内最大误差，q/qd/w/body_w是RMSE，**不同统计不能混称RMSE**。

| 时域/模型 | 根位置最大m | 根旋转最大rad | q RMSE rad | qd RMSE rad/s | 根w RMSE rad/s | body_w RMSE rad/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H32原冻结WM | 0.114961 | 0.383406 | 0.012078 | 0.121063 | 0.120558 | 0.144074 |
| H32旧短窗150步WM | 0.130027 | 0.455666 | 0.012361 | 0.133648 | 0.122479 | 0.158933 |
| H32新2048环境150步WM | 0.087539 | 0.333223 | 0.011327 | 0.126987 | 0.111273 | 0.150443 |
| H64原冻结WM | 0.649455 | 2.237559 | 0.051114 | 0.303693 | 0.619376 | 0.725652 |
| H64旧短窗150步WM | 0.580836 | 1.794174 | 0.058732 | 0.308863 | 0.696607 | 0.751583 |
| H64新2048环境150步WM | 0.688622 | 1.999698 | 0.045148 | 0.283024 | 0.608160 | 0.709840 |

H32位置/姿态/q有所改善，但qd、body_w相对原冻结WM略差；H64多个字段改善而根位置最大误差更大。结果不支持“已全面改善长时预测”。另两版长窗WM的完整同标签结果也保存在`fixed_long_WM_comparison_final.json`，没有只挑最好一列汇报。

## 7. 数值、显存和恢复验证

完整CPU回归：**72 passed，29.27秒，无真实SM资产skip**。覆盖零/非零Goal570、真实Actor、原奖励和梯度，SO(3)有限差分、接缝/halo、参考投影、音乐延迟、长响应Replay、完整H32/H64 BPTT及终止mask。精确临时目录已清理，用户要求的真实失败/诊断证据保留。

Server-1真实SM权重对齐：原SM奖励与原奖励梯度最大差均0。零Adapter Goal最大差4.63e-8、Actor动作4.77e-7；非零提案Goal差2.00e-6、Actor动作3.28e-7；相应梯度差在既有数值标准内，详见`numerical_original_SM_contract.json`。这些是合同验证，不是策略质量收益。

| 实测 | 结果 |
| --- | --- |
| 64环境长窗集成 | 30,208真实转移，WM1次、Adapter0，完整保存恢复通过 |
| 1024环境长窗集成 | 483,328真实转移，WM1次、Adapter0，Torch峰6240.80MiB，进程7706MiB |
| 完整150步checkpoint恢复 | 128环境新增8192真实转移，WM从150到151、Adapter0；三个模型/优化器、Replay及响应桶精确核验 |
| H32 GPU完整反传 | 64训练歌/128真实起点、microbatch32、梯度检查点；Torch峰988.53MiB，梯度有限且非零 |
| H64 GPU完整反传 | 同一采样规模；Torch峰1060.30MiB，梯度有限且非零 |

H32/H64反传检查没有执行Optimizer.step，也不包含完整2048环境并发开销。完整学习状态可恢复，但隐藏物理求解器状态不能保证复现，恢复后明确从新真实episode采集。**目前不能开始正式更新的原因是物理方向，不是H32/H64显存算不动。**

## 8. 配置与未完成边界

八份配置为`configs/closedloop/stage2a_gpu0_server1.yaml`至`stage2a_gpu7_server1.yaml`：四种预算/精度组合×seed42/43，num_envs=2048、Adapter/WM LR3e-5、actor batch1024、microbatch32、initial BPTT32、target200。

新目标H32标定低/高精度权重0.446714796/0.893429593，预定为SM梯度范数的0.5/1.0倍。标定只用64训练歌/128真实起点，记录标定WM身份`ac8a51e2...`；它没有为当前WM提供方向许可。原0.647/1.295为旧H8目标标定，已替换。

八组当前均未启动，无八个训练PID/会话、无首个有效更新。灵活预算有完整旧目标物理Oracle；保守预算还有未解决的实际投影数值反例。不得借另一预算Oracle或历史H8成功启动新H32正式组。

新目标下没有通过的Adapter，因此没有新模型整歌改善结果。原零Adapter四首独立整歌完成率100%，Tracking均值0.0965150、原始根RMSE0.0907677m、jerk718.926。历史旧目标单次H8更新结果不作为本轮长时训练成功证据。Critic按本次要求未训练、未提供终端价值；Stage2-B未实施。

## 9. 实际命令与复现

以下为Server-1宿主机命令，不是容器命令。原运行目录已经存在，应使用新输出目录复验，不能直接覆盖旧证据。完整历史启动命令、环境变量、输出重定向和退出码写入逻辑均在R下的脚本中：

```text
stage2a_long_WM_full2048_03.sh
stage2a_long_direction_full2048_150.sh
stage2a_long_trace_full2048_150.sh
stage2a_long_fixed_comparison_final.sh
```

对应实际执行入口：

```bash
# cwd=R/source_full_response_bfb6cbb；GPU2；完整命令见上面的run脚本。
/data0/user/liwei/envs/genmo_adapter_runtime_20261026/bin/python -B -u \
  tools/train_stage2a_reference.py \
  --config configs/closedloop/stage2a_wm_long_repair_2048_server1.yaml \
  --mode wm-repair --output R/long_WM_full2048_03 \
  --visualizer none --deterministic

# GPU4执行direction、GPU3执行trace；每次16个真实环境，4条件×4歌曲。
/data0/user/liwei/envs/genmo_adapter_runtime_20261026/bin/python -B -u \
  tools/audit_stage2a_reference.py \
  --config configs/closedloop/stage2a_pilot_server1.yaml \
  --mode direction --num-envs 16 \
  --checkpoint R/pilot_02/accepted_000001_1793091545760359804.pt \
  --wm-checkpoint R/long_WM_full2048_03/supervised_WM_no_Adapter_update_000000_1793100188510272551.pt \
  --output R/long_direction_full2048_150 --visualizer none --deterministic
```

上面`R/...`为便于阅读的路径缩写，实际脚本使用完整绝对路径；没有把占位命令当作实际启动证据。三路径只将`--mode direction`改成`trace`并使用独立目录。固定标签比较使用已提交的`tools/compare_stage2a_wm.py`，传入五个模型来源，其中原冻结WM自动加入。

CPU实际命令：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/weili/miniconda3/envs/env_isaaclab/bin/python -B -m pytest \
  tests/closedloop/reference_adapter -q -p no:cacheprovider \
  --basetemp=/tmp/genmo_stage2a_long_cpu_20261027_final
```

本地精选证据、完整实际启动脚本及曲线：

```text
/home/weili/bumi-closedloop-worktrees/adapter/stage2a_projection_20261027_01/long_response_evidence
```

当前必须保留的失败边界是接触变化附近的局部速度响应误判及H32/H64累计目标反向。已有有限修复预算用完，相关checkpoint、数据、日志和逐步证据全部保留；本次没有通过弱化阈值、增加更新次数或重新引入额外控制器强行放行。
