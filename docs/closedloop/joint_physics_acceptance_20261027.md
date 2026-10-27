# 从零联合扩散与物理约束：2026-10-27实际交付

## 当前结论和版本范围

独立代码、真实缓存、梯度审计、恢复和有界运行已实施。没有启动长期八卡训练；没有
把有效梯度/更新次数当作完整音乐收益。本报告区分追加要求后的离线H32/H64方案，与
此前H8的100步工程对照；后者不能冒充新方案的有效训练结果。

- 仓库：`XiaoxiaoKuankuan/GENMO`。
- 基线：`4a005d14d7e523d43959c21d3a6364de79653462`，当时实际本地/远端最新有效HEAD。
- 分支：`feature/joint-diffusion-physics-from-scratch`。
- 本地：`/home/weili/bumi-closedloop-worktrees/joint-physics/GENMO`。
- 两台服务器独立树：`/home/user/liwei/bumi-closedloop-worktrees/joint-physics/GENMO`。
- 原共享GENMO/GMT目录未切换分支；SM源码只读复用，无新增SM分支或接口改动。
- 同一冻结SM/WM checkpoint SHA256：`172fbc720583f415fdf27560c91bb8c6f6efe8165738d961cf2455694789cb54`。
- SM源码追溯：Server-1为`d9164b0e15934806ff8d750ff70911949c7f5171`，Server-2为
  `51b2172288d645ad930edaac6eac50ec6b16b772`；源码提交仅记录，未增加源码相等门禁。
  实际逐文件核对显示差异只在reward的可选diagnostics返回；本流程用到的网络、观测、
  终止、参考插值和物理执行代码一致。该核对用于说明两端数学接口，并未作为源码门禁。

完整设计、文件职责、合同和四个CLI见[方法文档](joint_physics_from_scratch.md)。
分阶段中文commit和每批实际失败/修复记录见仓库[记录文本](../../记录文本.md)。

## 追加要求后的实现

`offline.py`把连续实测状态附属缓存与请求前瞻一次性离线生成并持久保存。训练依据
dataset/split/source ID、公共网格frame和前缀身份随机索引读取；分片采用mmap，历史与
预测状态使用紧凑偏移，音乐不按anchor重复保存。训练不运行Newton、不调用WM重做请求
前瞻、不自动回填缺失缓存。候选参考的可微SM/WM执行链保持在线反传。

新schema为`genmo.joint_physics.sidecar.v2.offline_requests`；旧v1必须通过显式
`prepare_joint_physics_data.py --import-sidecar`转换到新目录。不会覆盖或重新解释旧缓存。
前瞻只消费当时真实历史和已经公布的旧计划；未来实测状态用于独立起点/验证。

新执行损失软关节限位已关闭：`L_exec_safety=0`、`lambda_safety=0`，执行超限只作诊断。
原Stage1监督中的限位项仍保留；WM状态域、动作域、参考保护、数值和时间支持检查未放宽。
测试专门构造实际超限状态，确认诊断非零但新执行软限位Loss仍严格为零。

默认训练H32，审计H32/H64；两者分别为0.64/1.28秒控制。H64长窗可能受已知前缀与
60源帧右速度halo限制；审计可在固定音乐相位附近±60源帧内，在预测前只按时间支持
寻找合法窗口，记录原始/选中帧。不能根据Loss、梯度方向或WM误差挑选结果。

## 真实数据与测试

原配对库、四库权重、train/val/test划分、stats/FK和地面标签保持原合同。

| 项目 | 实际证据 |
|---|---|
| 连续训练sidecar | Mine `badboy` 2989步、`banana-chacha` 2314步，合5303控制步 |
| 连续采集 | 每episode仅1次reset；均以真实参考末端支持耗尽结束，无失败后重置拼接 |
| 原始采集开销 | 两首分别122.34/78.26秒，未包含全部进程启动开销 |
| v2离线请求 | 1067个请求，876个合法前瞻（496/380）；其余保留拒绝原因 |
| 离线预处理 | 67.095秒，完整两首v2约13,805,488字节；没有重新仿真 |
| 独立验证缓存 | AIST val `gLO_sBM_cAll_d13_mLO2_ch02`，467控制步；未进入训练 |
| Server-2真实测试 | 13 passed，3.88秒；使用真实原数据、v2缓存与固定SM/WM，全程CPU |
| 条件隔离 | 修改未知未来标签/contact/地面后，历史、旧计划、Future条件逐元素一致 |
| 无在线预处理 | 读取测试将WM.step_model替换为报错函数，仍通过真实缓存读取 |
| 纯监督一致性 | Future无效时与原60帧Stage1监督逐元素一致；前缀/mask正确 |
| 初始化 | 禁止旧GENMO加载函数的测试通过，同seed相同、不同seed主干不同 |
| 冻结 | SM/WM参数、normalizer和buffers逐张量不变；没有其优化器 |
| 模式恢复 | 采样前后各GENMO子模块train/eval状态恢复，冻结网络不随GENMO.train改变 |
| DDP | Gloo实测非均匀有效样本及全零rank不死锁，梯度等于全局有效样本均值 |

初始化配置始终`mode=random / pretrained_genmo=null`。构造只调用静态
`build_stage1_actor`；同路径下的历史`s595000.pt`不读取，允许读取原stats JSON。
新训练只有一个GENMO AdamW，global_step、scheduler、辅助warmup从0开始。

## 关闭软限位后的H32/H64真实梯度

审计使用本新流程随机起步、此前100步得到的学生checkpoint做固定诊断对象，并使用
当前离线v2条件和关闭软限位的新目标；不从旧GENMO/旧Stage2模型初始化。该学生此前
训练时域为H8，因此这里是新H32/H64局部物理验证，不是H32/H64已完成训练的结论。

每探针只用独立L_exec对学生x_hat及GENMO参数求导，不混入BC。有限差分使用同一
epsilon=0.002；Newton每次从episode初态重放原控制前缀，再做正负扰动。没有直接回写
中途state55冒充求解器状态恢复。阈值保持原配置，原始失败目录保留。

| 检查 | H32 | H64 |
|---|---:|---:|
| 合法探针 | 6 | 6 |
| autograd/有限差分 | 6/6 | 6/6 |
| Newton同方向 | 6/6 | 6/6 |
| 模型误差阈值 | 6/6 | 6/6 |
| 同起点重放最大差 | 0 | 0 |
| 冻结检查 | 通过 | 通过 |

H32独立执行梯度范数：x_hat为0.06075–0.08956、Future为0.03824–0.08158、主干为
1.88969–3.29104。WM相对Newton的最大窗口RMSE：根XY 0.00210m、根姿态0.00879rad、
关节位置0.00641rad、关节速度0.18787rad/s。每窗11步预览、21步中心消费，19–20个
源点得到梯度支持，占可修改源点42.2%–52.8%；不称完整60帧优化。

H64独立执行梯度范数：x_hat为0.05516–0.13244、Future为0.06110–0.11947、主干为
1.91205–4.40901。最大窗口RMSE：根XY 0.00416m、根姿态0.00471rad、关节位置0.00573rad、
关节速度0.13772rad/s。每窗11步预览、53步中心消费，39个源点得到梯度支持，占可修改
源点81.25%–95.12%。H32使用源帧261/870/1479；H64支持检查后为264/873/1479，属于
分别验证，不能当成完全相同锚点的H优劣比较。两份报告的因果前瞻对实测未来检查也通过。

H64最初固定锚点有4个探针因右halo不足被拒绝，只有2个完整探针，尽管其方向一致，
仍因样本数未达到预设minimum_probes=4而判不通过；该负结果未覆盖。

## 有界训练与完整恢复

追加要求前的H8工程对照使用完整1024维、16层原Transformer，A/B同随机seed、样本
ID/裁剪、噪声等级、dropout和历史来源逐步一致，各100次监督更新、400个监督样本。
A为0次物理更新；B为89次物理有效更新、146个有效窗口，有效率36.5%。监督和执行
对Future/主干的梯度分别记录。B支持域拒绝没有替换GT或放宽门禁。

A/B更新内墙钟分别39.95/102.61秒，额外62.66秒；该计时不含初始化与checkpoint写盘。
CUDA分配峰值约4.397/4.405GB。曲线文件为本地证据目录中的
`bounded_AB_H8_historical.png`，明确标注历史H8范围。

追加要求后，Server-2 CPU完成离线H32 A/B各8步、16监督样本，整次进程墙钟21.18/21.75秒。
B物理有效更新为0：随机起步预测仍被状态域/动作域等检查拒绝，BC正常保留。
这是监督起步与新缓存/门禁链路验证，不能报告H32联合训练成功。

B的16个样本拒绝分类：前缀不足2、SNR排除5、WM状态超域6、音乐dropout1、动作超域2。
该短测主动把物理课程起点设为0以检查随机坏预测的拒绝路径；正式默认物理start_step=1000。
当前v2流程完整checkpoint也已在Server-2实际从第8步恢复到第9步，data loss为
3.245849609375，累计物理更新仍为0；模型、日志保存到独立`offline_H32_B_resume9_cpu`。

完整恢复实测：原A不中断第3步与从第2步恢复，第3步loss同为2.4863522052764893；
327个Actor张量、AdamW、scheduler、rank RNG/采样游标、计数逐元素一致。原联合B分别
从第50步和第100步恢复至第101步，全部状态也逐元素一致；data loss均2.2559821605682373，
exec loss均0.19926729798316956，累计90次物理更新。仅接受新schema完整checkpoint；
切换到v2数据合同或关闭限位的新目标，不能把旧训练合同的恢复伪装成等价继续。

额外完成真实双GPU联合短测：8更新、64监督样本、21物理窗口、8次有效物理更新，
同步退出且冻结资产不变。其目标仍属此前H8合同，不外推H32/H64 DDP有效率。

## 三类评价与负结果

以下评价对象为同一对历史100步A/B checkpoint，同一独立val音乐、同一噪声种子，
不将两模型同seed视为相同生成舞蹈身份。生成与连续闭环结果不等于新H32训练收益。

| 模式 | A | B | 范围 |
|---|---:|---:|---|
| reconstruction监督均值 | 2.536113 | 2.535701 | 2合法窗口×5噪声等级，另1锚点前缀不足 |
| reconstruction真实执行均值 | 0.044879 | 0.044873 | H8；10窗口均未触发SM失败 |
| free_generation跨seed RMS | 约9.4e-5 | 约9.4e-5 | 完整20步DDIM，低多样性需进一步研究 |
| closed_loop完整曲目 | 0/1 | 0/1 | 9.6秒音乐均只执行1.74秒，0次新计划接管 |

连续评价冷启动和显式数值预热后都失败于启动参考耗尽。预热后实际首请求耗时
A3.0555秒/B3.0103秒，超过当前启动参考支持；其中因果前瞻约1.8秒、生成解码约1.2秒。
每次87个控制deadline超期。没有暂停物理等待生成，没有延长保护区把失败改成成功。
因此本轮没有完整音乐闭环成功、自由生成收益或实机稳定证据。

## 正式入口、恢复和证据位置

后续训练目标Server-2。本次准备时其八张GPU被原全量音乐评价占用，本会话没有修改旧
进程，Server-2新增工作仅用CPU。收尾再次查询时两台服务器GPU均已空闲；旧评价目录
仍标记未全量完成，不能把进程退出当作全库完成，本会话也未据此启动长期训练。
H32/H64 Newton测量在Server-1本轮旧任务自然结束、重新确认空闲后执行。

主配置：`configs/closedloop/joint_physics_server2.yaml`。以下提供正式命令，没有执行：

```bash
cd /home/user/liwei/bumi-closedloop-worktrees/joint-physics/GENMO
joint_py=/data0/user/liwei/envs/genmo_sm_runtime/bin/python
joint_cfg=configs/closedloop/joint_physics_server2.yaml

# 原train/val/test各自一次性准备，输出必须是独立新目录。
"$joint_py" tools/prepare_joint_physics_data.py --config "$joint_cfg" \
  --split train --output <full_train_sidecar_v2>

# 新目标长窗验证：明确新流程学生checkpoint，或显式--interface-probe作接口对照。
"$joint_py" tools/audit_joint_physics_gradients.py --config "$joint_cfg" \
  --checkpoint <joint_checkpoint> --set data.sidecar=<full_train_sidecar_v2> \
  --newton --output <new_h32_h64_validation>

JOINT_PYTHON="$joint_py" JOINT_GPUS=1 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/train_joint_physics.sh "$joint_cfg" --max-steps 600000 \
  --set data.sidecar=<full_train_sidecar_v2> \
  --set physics.validation_report=<matching_h32_validation.json> --output <new_run>

JOINT_PYTHON="$joint_py" JOINT_GPUS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/train_joint_physics.sh "$joint_cfg" --max-steps 600000 \
  --set data.sidecar=<full_train_sidecar_v2> \
  --set physics.validation_report=<matching_h32_validation.json> --output <new_ddp_run>

# 同一模型、数据、物理目标、采样和DDP布局，使用完整checkpoint；写新输出目录。
JOINT_PYTHON="$joint_py" JOINT_GPUS=1 bash scripts/train_joint_physics.sh <same_run_config> \
  --resume <joint_numbered_checkpoint> --max-steps 600000 --output <new_resume_directory>
```

当前真实缓存只有两首train和一首历史val，物理方向覆盖仅单曲3阶段/2特征，不能把本次
报告当全库验证。正式全库需准备对应v2缓存并扩大来源、动作、接触和种子覆盖。没有
扩大支持域、关闭门禁、减少DDIM步数或启动长期训练来掩盖这些缺项。

- Server-1原始日志/模型/失败目录：`/data0/user/liwei/GENMO_outputs/joint_physics_acceptance_20261027`。
- Server-2新v2、CPU A/B模型/日志：同上绝对路径，但主机为Server-2。
- 本机小证据、原始JSON/JSONL、曲线和v2副本：
  `/home/weili/bumi-closedloop-worktrees/joint-physics/evidence/acceptance_20261027`。
- 本机Server-2证据：`/home/weili/bumi-closedloop-worktrees/joint-physics/evidence/server2_20261027`。
- 实际失败与修复：Newton确定性枚举、固定参数梯度诊断、测试SM路径、真实缓存测试
  来源选择，以及H64右halo拒绝均保留记录；不覆盖旧checkpoint或负结果。

测试临时目录已按精确路径清理。真实验收数据/曲线/checkpoint依据用户要求保留。
