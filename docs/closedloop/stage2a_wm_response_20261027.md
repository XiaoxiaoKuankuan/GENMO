# Stage2-A世界模型真实一步与奖励响应修复：2026-10-27实测

本轮在Server-1独立工作树完成代码、两轮有限A/B/C/D监督、逐版本Newton审计和当前WM新Adam候选诊断。H32仍未通过，H64亦未通过；Adapter接受更新为0，没有启动新的正式Adapter训练。没有降低原噪声、重复性、状态或方向门槛。

任务于北京时间20:07开始，所有物理进程于21:06前结束；代码/报告同步在用户两小时预算内完成。原八组失败实验、既有checkpoint/log不覆盖，其它分支和会话不切换。

## 本次实际修改

|文件|具体变化及理由|
|---|---|
|`dynamics.py`|全部有效真实state/action/xi一次批量计算下一状态，六组物理误差归一化后真正反传WM.net；保留原targets诊断、输入输出维度及normalizer。|
|`reference_response.py`|取消接触变化否决；保留初态/数值/重复/动作可辨识/episode/时间/终止检查。保存四条件实际动作、投影残差、名义身份和奖励，新增噪声mask后的奖励差Huber。|
|`stage2a_physics.py`、`physical_trust.py`|随机参考以及完整Adam正反候选都经同一投影和冻结Actor采集；失败候选响应采用相同记录器。真实信任域公式和门槛不变。|
|`vector_learning.py`|显式将一步状态、原状态响应、奖励响应加入WM总损失，只执行WM optimizer；Adapter仍只使用WM+BPTT。|
|`stage2a_entry.py`|新增一次性真实数据准备，显式WM权重初始化；正式与集成入口读取固定训练梯度标定，不把权重初始化称完整续训。|
|`wm_ablation.py`、`tools/train_stage2a_wm_ablation.py`|同原WM、同标签A/B/C/D，最多150次，每25次独立真实审计；另区分固定已知探针与当前WM新H32 Adam提案。|
|两个`stage2a_wm_ablation*_server1.yaml`|首轮小权重与第二轮训练集标定配置；256环境一次采集、16环境专用审计，单GPU无DDP，seed42。|
|`test_wm_response_training.py`|真实SM数值夹具验证一步梯度/权重、奖励重放、接触保留、断续拒绝和WM专属梯度。|
|`tools/report_stage2a_wm_ablation.py`|从实际JSON只读生成所有版本表格、端点奖励误差和PNG/PDF；不自动放行训练。|

冻结GENMO、Actor、normalizer；Adapter 2654→512→512→256→270与十帧投影不变。原SM六项与权重不变，music=1e-4，软限位、额外body_w Huber、rootXY、Critic终端均关闭；H64不进入监督。

## 数据及覆盖变化

只准备一次153600真实转移：train 90112、train-safety 6144、validation 57344。64首train全部覆盖，4 validation、4 audit、4 evaluation保持分离；FineDance189仍是audit，未用于拟合。名义缓存来自冻结GENMO真实生成，仍用原资产manifest与时钟。

|同一新采集数据|采集对数|旧规则H32|取消接触筛选H32|旧规则H64|取消接触筛选H64|
|---|---:|---:|---:|---:|---:|
|随机参考：零/非零提案附近各64对|128|15|124|10|112|
|完整Adam参数正反响应|64|7|61|4|57|
|D组合使用|192|22|185|14|169|

“旧规则”是在同一新数据上重算接触mask，非不同采样批次的统计。跨接触数据只作有限幅度响应，不宣称平滑局部导数。真实四条件64步奖励重放最大误差为0。2048个普通训练窗口、2048个D训练窗口和256个验证窗口的相邻真实state与前一next_state差均为0，没有跨reset拼接。

初始WM参数SHA：`8551b2b8672e535d4a8f225876c5a353b24d2c2d5b32b00e8fd3140523318f5e`。固定标签SHA：`c8a8343ec6683d67793e262ac2d921c28815143ea94966170e47e8988caa45e0`。两轮所有模型均从此原WM独立初始化，未接续旧150版本。WM net发生实际变化；in/out norm的mean/var/count及error_floor逐张量完全不变；Adapter保存的基线始终精确为零。

## A/B/C/D定义及损失强度

- A：现有H8状态、H16/H32状态及参考响应损失组成。
- B：A + 全部真实时刻的一步状态监督，weight=1。
- C：B + 原SM六项及完整目标的累计奖励变化监督。
- D：C + 实际Adam/反向完整参数响应及对应真实连续转移。

接触取消适用于全部新组。第一轮使用Huber和最大1的动作幅度增益；校准发现该增益过度压低参考响应，第二轮改为1/sqrt(action_RMS²+0.01²)，上限100，仍远低于旧1e-4下限的一万倍。故A只代表当前损失组成，不冒充旧代码完全不变。两个轮次都保留，不把归一化变化归因成某一新增loss的效果。

首轮奖励权重0.01，在H8训练探针仅占状态梯度约0.018%。第二轮只用训练标签H8/16/32标定，完整状态+一步梯度8.260942；H32参考/奖励响应梯度7.299531/3.883056。固定参考权重0.0565854369、奖励权重0.1063716647，两项初始H32梯度各占5%；H8/H16奖励比例0.070%/0.901%。这只是初始化标定，不承诺后续梯度比例不变；没有用audit/evaluation选权重。

WM LR3e-5、clip1；512个短窗、64个长窗、16个响应对/次；响应时域8/16/32轮换，H16/H32状态权重各0.25；每组150次，0/25/50/75/100/125/150保存。各轮A-D配对seed42，数据源及基础物理一致，模型/优化器/输出独立。

## 固定validation的150次结果

六项及完整目标逐版本、H1/H4/H8/H16/H32/H64状态与响应完整数字见本地证据`report/summary.json`与`report/tables.md`。下面是第二轮最终值；一步项为所有16384真实转移的六组归一化均值，H32速度为固定256窗口端点RMSE，量纲rad/s。

|WM|真实一步损失|H32 qd|H32 根w|H32 body_w|H32完整奖励差RMSE|
|---|---:|---:|---:|---:|---:|
|原WM|0.0092559|0.126534|0.093064|0.137794|0.0010481|
|A|0.0091169|0.136477|0.109283|0.153287|0.0011582|
|B|0.0089760|0.136151|0.109629|0.152343|0.0013039|
|C|0.0090437|0.130353|0.105418|0.147198|0.0013461|
|D|0.0090889|0.144922|0.104744|0.151116|0.0014797|

新增一步损失有小幅一步误差改善，但没有得到速度响应与奖励方向同时稳定改善的WM。第二轮四个最终WM的H32根位置最大误差0.0500～0.0537m、根姿态最大误差0.1879～0.2144rad，长时状态也没有全面合格；不能只取容易的audit片段状态门槛通过来替代完整validation。H64最大根位置约0.53m，继续只作验证。

## Newton方向与新的完整Adam候选

两轮所有0/25/.../150版本固定参数探针H32均保留FineDance189/prefix50的可信反向。已知探针真实变化约−0.000311，预测仍约+0.0007～0.0009。H64还出现反向和重复状态差超过原1e-4；第二轮C150固定探针另有H32重复性失败。

为避免只检查旧方向，每个最终WM从同64首train真实起点重新生成H32 Adam候选（LR3e-5），再在独立audit歌曲重新执行baseline/repeat/candidate/reverse。下表均为FineDance189/prefix50；正数代表改善。

|组|真实完整目标变化|WM闭环预测变化|真实重复差|H32反向案例|接受更新|
|---|---:|---:|---:|---:|---:|
|A|-0.002265215|0.001099586|2.38e-07|1/8|0|
|B|-0.002816200|0.000961781|0|1/8|0|
|C|-0.002455711|0.001033545|4.77e-07|1/8|0|
|D|-0.002677917|0.000935555|0|1/8|0|

四组新候选H32重复性均通过，但真实方向失败。候选的参数L2变化约0.00760、Actor动作RMS变化0.00424～0.00433、Goal RMS变化0.00136～0.00143；64歌曲探针全部产生可测动作变化，Tanh饱和率0。它们仅作为未接受提案保存，未继续优化、未把提案数当有效更新。

## 具体原因及尚未解决的部分

已修正的代码问题：接触筛选丢掉大量长窗；一步预测以前只记录no_grad诊断；监督没有直接要求奖励变化正确；旧动作RMS倒数会放大下限附近噪声；参考数据未覆盖实际完整参数候选。上述问题修复后，数据可用性和工程正确性得到验证，WM物理方向仍然失败。

对D150新候选，同一起点H32三路径结果：

|路径|累计目标变化|
|---|---:|
|真实Newton|−0.002677917|
|每步喂真实状态的一步WM|−0.002049685|
|真实动作序列驱动WM连续预测|+0.002514124|
|Adapter+冻结Actor+WM完整闭环|+0.000934839|

新候选每步真实输入路径最终方向正确，但真实动作连续预测已经反向，说明该例的递归状态响应偏差是实质问题，不只是Actor闭环另选了动作。已知旧探针在每步真实输入下H32仍预测+0.0007045而真实−0.0003111；因此局部响应偏差也未消除，不能宣称只有长时累积一种原因。新候选三路径最早第2步出现累计反向；旧探针分别为第4/5/3步。

以D新候选的原SM六项为例，H32按原权重与一次dt累计的真实/闭环WM变化：

|原SM分项|真实Newton|WM闭环|
|---|---:|---:|
|body_pos|-0.000010865|+0.000023315|
|body_ori|+0.000190365|+0.000433912|
|body_linvel|-0.000257940|-0.000037315|
|body_angvel|-0.001009471|-0.000060566|
|root_pos|+0.000018075|-0.000003996|
|root_ori|-0.000330131|-0.000103979|

身体角速度的实际坏处约为WM预测的16.7倍；body_linvel也被低估，body_ori收益则高估。额外身体姿态精度项真实−0.0007328、WM+0.0002325，关节速度精度真实−0.0005030、WM−0.0000483。残差/参考正则约1e-6量级，音乐约5e-6，不能解释约0.0036的总方向差。奖励重放零差、实际动作有变化，已排除重复dt或Adapter未生效作为本轮直接解释。

当前证据不证明模型架构必然无解，也不证明接触切换是唯一原因；仅证明本轮有限数据和监督方式尚未学到可靠的长期参考响应。继续扩大Adapter更新会沿已知错误方向走，因此没有启动。下一轮应先用已有train片段检查真实单步响应和连续递归响应的差异，扩充可复现的训练难例；不能拿FineDance189这个audit样本反复拟合。

## 实际开销、进程与完整恢复

一次数据153600转移；两轮8个WM进程各7次独立审计，共159488；四个新Adam候选审计11392；两个三路径诊断5696；64环境集成26112。合计356288真实转移，没有重新采集300万条。两轮总WM监督1200次，集成另1次；策略接受0。

|运行|GPU|PID|tmux|环境数|WM次数|新增审计转移|Torch峰MiB|
|---|---:|---:|---|---:|---:|---:|---:|
|A01|0|1277701|wm_response_A01|16|150|19936|1597.32|
|B01|1|1277705|wm_response_B01|16|150|19936|1597.32|
|C01|2|1277708|wm_response_C01|16|150|19936|1597.32|
|D01|3|1277712|wm_response_D01|16|150|19936|1597.32|
|A02|0|1539434|wm_response_A02|16|150|19936|1597.32|
|B02|1|1539438|wm_response_B02|16|150|19936|1597.32|
|C02|2|1539442|wm_response_C02|16|150|19936|1597.32|
|D02|3|1539445|wm_response_D02|16|150|19936|1597.32|

四个新Adam诊断PID为1755905/1755909/1755913/1755916（对应GPU4/5/6/7，以各manifest/status为准）；它们同样只是诊断，未正式训练。数据准备PID1187246，256环境；集成PID1821943，64环境。所有本轮tmux进程均已按预算退出，17个主任务退出码均0。退出0表示任务正常执行结束，不表示方向通过。最终GPU0～7均无compute进程，原`genmo-rolling-20261022_182720`会话仍在，未触碰。

完整CPU回归74 passed/31.21秒；Server-1真实SM零/非零Goal、Actor和梯度合同通过，原奖励及奖励梯度最大差0。64环境实际26112转移、1次新组合WM loss更新；三个网络、三个优化器、真实Replay、新四条件参考响应Replay和RNG保存/加载精确一致。物理恢复明确新episode，不声称恢复Newton隐藏状态。

本轮未重新进行完整歌曲质量比较：没有合格的新Adapter可供比较。没有启动八组正式策略、没有恢复旧退化模型、没有实施Stage2-B，也没有把历史整歌基线冒充本轮新结果。

## 位置和复现命令

服务器根目录：`/data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/stage2a_wm_response_20261027_02`。各A01..D02、candidate_A02..D02、trace_D02_known/new、integration64_02目录独立；所有`.sh`保存完整env、命令、工作目录及退出码。

本地只读证据：`/home/weili/bumi-closedloop-worktrees/adapter/stage2a_wm_response_20261027_02`，含JSON/日志/完整命令和已视觉检查的`report/wm_ablation.png`、PDF；模型及真实大数据仍在服务器，未复制旧训练权重作为新策略。该目录按用户要求保留作为失败诊断和复现材料。

首轮运行源码1e67179，标定及第二轮82f355a，当前候选与三路径50e857f，完整集成b5dd98b；最终报告和汇总工具见当前提交。配套SuperMimic仍为a0911e4，没有修改SM。

以下命令在Server-1主机内执行；更换为不存在的output，禁止覆盖已完成目录：

```bash
cd /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/stage2a_wm_response_20261027_02/source_82f355a
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export ISAACLAB_PATH=/data0/user/liwei/GENMO_assets/adapter_20261026/IsaacLab
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=/data0/user/liwei/GENMO_assets/adapter_20261026/inductor_cache
/data0/user/liwei/envs/genmo_adapter_runtime_20261026/bin/python -B -u tools/train_stage2a_wm_ablation.py \
  --config configs/closedloop/stage2a_wm_ablation_calibrated_server1.yaml \
  --data /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/stage2a_wm_response_20261027_02/data01/fixed_ablation_data.pt \
  --group A --output /绝对路径/新的A组目录 --visualizer none --deterministic
```

其余组仅切换GPU、group、独立output。已有实际命令以`wm_response_A02.sh`至`wm_response_D02.sh`为准。新候选诊断加`--candidate-only --wm-initialize .../D02/wm_150.pt`，使用source_50e857f；它不会接受策略更新。

本地回归与只读报告：

```bash
cd /home/weili/bumi-closedloop-worktrees/adapter/GENMO
PYTHONDONTWRITEBYTECODE=1 /home/weili/miniconda3/envs/env_isaaclab/bin/python -m pytest -q -p no:cacheprovider tests/closedloop/reference_adapter
PYTHONDONTWRITEBYTECODE=1 /home/weili/miniconda3/envs/env_isaaclab/bin/python tools/report_stage2a_wm_ablation.py \
  --root /home/weili/bumi-closedloop-worktrees/adapter/stage2a_wm_response_20261027_02 \
  --output /绝对路径/新的汇总目录 --plot
```

当前没有符合验收条件的WM，不能将以上监督/诊断命令称为正式Adapter启动。
