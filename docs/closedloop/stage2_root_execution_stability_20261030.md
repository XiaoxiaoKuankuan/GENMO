# Stage2根位置与执行稳定性四组双种子正式对照

## 依据和范围

2026-10-30的旧八组结果表明：H32关节/姿态在6000至9000更新间接近平台，真实
validation根轨迹与jerk没有同步改善；H64约两倍耗时，等4000更新没有持续优势。
根XY Huber从0.830增到1.491没有可靠收益，强参考连续性对根轨迹和脚滑较有帮助。
用户授权停止GPU0至4，并按效果判断其余GPU是否换组。本次八组全部保全后换组。

旧根目录保持不变：
`/data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/formal_root_continuity_20261029_2157`。
停止证据位于同父目录`replacement_audit_20261030_214608`，最终Adapter更新为
9550/9593/9545/9578/5146/5164/4813/5024。全部完整checkpoint已实际反序列化核对，
含Adapter/WM/Critic、三个优化器、Replay、因果历史及Python/NumPy/Torch/CUDA RNG。
旧入口SIGINT通过异常保存处理器退出，因此文件名failure和KeyboardInterrupt表示
用户换组，不伪装成新数值/物理失败。八个目标PID及GPU计算进程已全部退出。

## 保留的执行与评价合同

冻结Stage1 GENMO、SuperMimic Actor及normalizer；2654→512→512→256→270、三层
SiLU、十帧独立有界残差与SO(3)组合不变。根XY补偿关闭，根姿态补偿开启，原始
GENMO仍是Tracking目标。没有QP/SQP、接触分类网络、额外WM输出头或Critic终端值。
WM真实H8监督十次→H32闭环BPTT→下一轮真实采集，音乐权重1e-4。

八组共同使用原SM六项计分，根位置/根姿态报告权重为1.2/1.0，误差尺度为0.2/0.3，
body位置/姿态尺度为0.1/0.15；真实失败条件仍为高度误差>0.2m或倾斜z差>0.6。
增加根奖励通过独立`root_tracking_bonus`加入训练目标，不改共同Tracking报告，
防止奖励权重翻倍带来的分数增长被误报成真实跟踪提高。训练目标中根位置有效
权重分别为1.2或2.4，原名义根XY Huber仍用低档lambda=0.8300100327。

## 执行稳定性公式

实际执行jerk为`j(k)=(qd(k)-2*qd(k-1)+qd(k-2))/0.02²`，单位rad/s³。
成本为逐关节Huber(j/s_jerk)均值，标尺来自64首train零策略真实轨迹绝对jerk的P75，
下限100rad/s³。复位后的第一步不计；真实Replay保留前一实际qd和有效mask，想象
内部不detach历史。它不同于旧的参考速度/加速度成本，直接惩罚执行轨迹急变。

足端代理仅在名义支撑相位惩罚`Huber((v_actual_xy-v_nominal_xy)/0.1m/s)`。
相位权重为：

```
a = sigmoid((0.04-(z_nominal-z_floor))/0.01)
b = exp(-||v_nominal_xy||²/0.2²)
g = relu((a*b-0.8)/0.2)
```

`z_floor`是该歌曲名义足端高度P2，单次缓存。摆腿、腾空或快速名义足端不作为
支撑标签。g只依赖冻结参考，无法通过抬起预测的脚逃避惩罚；执行足速直接复用
原SM实际状态FK。该项明确是代理，正式脚滑指标继续使用Newton真实接触。
在train的64环境、零/微小非零Adapter各128步上独立核对精确率/召回率；仅当两种
分布精确率均>=0.85且召回率>=0.02时配置非零权重，否则明确关闭该可选项。
这个判定不阻止根位置/jerk正式训练，也不是WM方向门禁。

新增成本均仅乘一次dt=0.02；根bonus为`+0.02*bonus_weight*原root_pos指数项`。
新增执行成本初始参数梯度各标为原SM Tracking的10%，只用固定train真实起点和
原始预训练WM，不使用validation/evaluation调权。模型梯度标定不构成物理收益证明。

## 四组双种子矩阵

|组别|GPU|seed|有效根位置权重|执行jerk/名义支撑足端成本|
|---|---|---|---:|---|
|A 原目标对照|0/1|42/43|1.2|关闭|
|B 根奖励加强|2/3|42/43|2.4|关闭|
|C 执行稳定性|4/5|42/43|1.2|小权重，训练集固定标定|
|D 根奖励+稳定性|6/7|42/43|2.4|同C权重|

所有组统一H32、LR1e-5、2048环境、Adapter batch1024/micro256，WM LR3e-5、
H8/batch2048/每轮10次，gamma0.99、Critic关闭、真实动作噪声0，10000次实际Adapter
更新。固定强档参考连续性、原精度权重及其它奖励不变，不把所有参数一起改动。
各组从零Adapter和原始WM重新开始；同seed的四组初始权重/采样流配对，在线数据
及模型/优化器各自独立。不同seed不宣称隐藏层初始参数相同。

0/5/10更新短时真实配对；20/50/100/200/500及每1000次整歌validation/evaluation。
validation选best，evaluation只报告。评价退化不退出、不回滚；保留NaN/Inf、OOM、
物理异常保护。新增仅由人工创建的stop_requested.json，迭代边界保存完整状态后
正常退出，避免下次换组必须在Adam中途发送SIGINT。

## 实际验证与运行

本地完整Adapter回归164 passed，58.51秒，只有两条既有ONNX弃用warning。
默认本地supermimic_vibed生成MJCF于当日已变化，与旧冻结checkpoint不一致；没有
放宽资产检查，而是只读复制Server-1原始冻结MJCF到
`/home/weili/bumi-closedloop-worktrees/adapter/frozen_sm_assets_20261026/data/bumi3_4340/robot`，
明确传SM_ROBOT_ASSET_DIR后完成真实SM权重数值、梯度、Replay和checkpoint回归。
初次错误资产运行76 passed/88 setup errors如实保留为环境问题；不改变本地当前
机器人资产或冻结权重。测试临时目录自动精确清理。

真实标定权重、代理检查结果、正式运行路径/PID及首次有效更新将在实际完成后追加。
