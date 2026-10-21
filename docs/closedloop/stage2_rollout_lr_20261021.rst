逐批学习率试探、正常结束剩余更新与取消偏航终止
================================================

已实施行为
----------

用户要求停止当前训练并修改更新控制，保持全部16层Transformer参与，取消转向
误差失败项。本批仍在原GENMO/GMT固定分支，没有启动新的正式训练。

actor_update进入新rollout时从配置maximum_lr重新试探，离开时恢复名义LR。
Adam一、二阶动量、18个参数组倍率完整保留；steps记录实际接受LR，名义值另记。
checkpoint不再将上一批额度耗尽后的微小临时LR带入下一批。

初始设置为共同名义LR1e-6，Transformer/其他模块倍率0.1，输出头倍率1；
stop_budget_fraction=0.95，minimum_trial_lr_ratio=0.0625。已接受状态用到
任一候选链KL额度95%时结束剩余小批次；试探低于名义LR的1/16时不再执行该
微小Adam尝试。数值为可配置候选，本批不宣称已通过物理收益标定。

前面已有接受更新时，后续全部为有限联合KL/active-clip拒绝，则只恢复该待接受
小批次的Actor/Adam及BC游标、控制器、RNG，保留之前步骤并正常结束剩余更新。
没有接受过任何一步或有数值异常时仍显式失败，由原整轮事务恢复。
实际尝试、拒绝、接受、跳过次数分开统计，丢弃步骤的BC不计入接受监督量。

完整rollout最终四步联合KL仍须通过；KL阈值、真实Gaussian噪声、完整链PPO、
奖励权重、Stage1 BC、GMT权重与物理时钟不变，没有恢复ESS硬门限。
正常提早结束不代表该轮所有数据都得到优化。

真实链目标和更新方向
--------------------

chain_diagnostics增加真实clipped目标、行为起点目标、目标增益和一阶score增益，
不改变loss。detailed_diagnostics捕获全局PPO/BC梯度，按实际Adam增量计算方向
内积；普通训练不开启额外整模型分量拷贝。固定数据入口增加--minibatch-chains，
支持同批旧数据、Stage1、冷Adam、BC/RNG比较2048/4096等有效批量，微批和概率
定义不变，每轮BC总量一致。物理收益仍需同协议完整音乐执行验证。

TensorBoard增加名义LR和整批PPO目标增益，两个坐标平均KL只留JSON辅助诊断；
面板标签数量保持70以内，既有JSON标签可读，没有新建额外run。

取消偏航失败
------------

stage2_transformer_groups、stage2_transformer_sensitive、stage2_full_chain三个
全参数配置的termination.yaw_error_rad设为null。GMT不注册yaw_anchor_ori，
实际终止管理器保留根高、非yaw倾斜、末端相对高度三项；运行清单记录null。
偏航误差仍计算/记录，八项跟踪奖励中的anchor_ori及其权重不变。

sim2sim统一加载器默认偏航阈值null；GENMO告警与GMT MuJoCo终止判定均跳过
null，避免float(None)，其他三项继续检查。历史显式数值阈值仍可重放原协议。
未改日常模型路径、旧评估记录或模型文件。

取消终止项会改变完成率和回报协议，后续须与同样关闭偏航终止的Stage1配对，
不能用旧3/32与新完成率直接比较后声称模型改善。

旧训练停止结果
--------------

2026-10-21 11:37向原session ed2c9524-cae0-4ef5-9931-a9421f96f402八个worker
发送由StopSignal处理的SIGTERM。旧程序第56轮先发生最大链KL回溯耗尽，未走到
轮末正常停止保存分支。约11:47异常证据保存、整体学习状态恢复后退出1；八个
worker和GPU计算进程全部清空。没有将异常退出写成正常退出。

第55轮曾完成434次累计Actor更新但未完整落盘；第56轮未接受。最新完整模型为
第54轮/426次Actor更新：

checkpoints/stage10_r000054_a000426_01792553325534125260_2ec4eb67fbb1_periodic.pt

SHA256：a65a573a95a965e1d6e480e1a0ea4c140809c979ae5c9084d53e2cf09c889971。
原模型、日志、失败rollout、冻结GMT资产全部保留，没有重新开启正式训练。

测试与证据
----------

服务器1隔离CPU：GENMO107 passed/2 skipped（17.89秒），GMT37 passed（0.75秒）。
包括非空分组Adam倍率/动量、新批次LR重试、有限拒绝保留先前步骤、BC/RNG恢复、
数值异常、真实小扩散网络、两进程CPU梯度归约、流式完整链概率/梯度、周期评估、
源码仅记录、单run面板和sim2sim配置；GMT核对实际manager只保留三个终止项。
跳过GPU专用用例，未冒充新PhysX收益。

首轮GENMO4个失败来自测试误将接受一次等同只尝试一次Adam、旧BC替身缺状态
接口、面板总标签超约定；修正后全量107项通过，未放宽概率或梯度容差。

证据保留本地outputs/analysis/stage2_rollout_lr_20261021/以及服务器
/data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_rollout_lr_20261021_analysis/。
临时源码、pytest目录已删除，补丁包在本批结束前精确清理。新物理收益未验证。
