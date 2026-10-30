2026-10-30：按新规则重新筛选音乐、文本及本地SuperMimic数据
======================================================

用户在规则完成后授权真实执行。筛选沿用
``configs/bumi/quality_filter_dance_binary_30hz_v1.yaml``，配置SHA256为
``4800509142f9bc0b109157167bba44909b267d7c30ea934c606abbcce59135e0``。
根高度、根倾角、地板和高动态仅诊断；数值、具名关节、四元数、机械位置范围、
30Hz完整时间线及原始来源身份仍验证。只输出PASS/REJECT。

实际筛选
--------

Server-1从原完整 ``/home/user/music7286_umr/out_umr/bumi3`` 建立6915条清单，
不再使用旧best70%。加上保留原CSV根高度的99条Mine，现有音乐输入7014条全部PASS。
7014组EDGE35逐条检查35维、有限数值及与动作等长，全部通过。

Server-2使用新独立adapter工作树，在da33b90版本上执行
``tools/data/bumi/rescreen_text_dance_binary.py``，16个低优先级CPUworker，
392.583秒完成四库717504条原始动作。完整旧SQLite inputs仅用于定位原始文件；
重新读取机器人NPZ、校验人体/机器人内容SHA并按新规则重算，不沿用旧状态。
没有使用旧60..300帧训练候选白名单、旧根倾角/接触筛选或最终环境筛选。

================= ======== ====== ==================
数据来源          PASS     REJECT 恢复旧非PASS数量
================= ======== ====== ==================
SOMA-BONES-SEED    131454   0      28384
HumanML3D         23242    0      4400
KIT-ML            2884     0      377
MotionMillion     559924   0      133078
AIOZ-GDANCE       5890     0      不与旧70%清单混比
AIST++            899      0      同上
FineDance         126      0      同上
Mine              99       0      同上
合计              724518   0      文本合计166239
================= ======== ====== ==================

所有PASS的 ``valid_intervals`` 均为完整 ``[0,T)``；
共120485975个30Hz帧，按SuperMimic原方式重采样为200082242个50Hz帧。
68061条触发旧“持续地板”启发式判据的动作都保留；不把该统计当舞种或真实接触标签。
原动作已由UMR约束过关节范围，因此新规则全通过不等于物理跟踪已通过。

覆盖缺口与条件资料
------------------

原音乐人体包7286条中371条没有现成机器人输出：AIST++121、AIOZ121、FineDance57、
CoMPAS3D72。没有将不存在的轨迹伪装为已筛过，缺口单独列出。
新结果覆盖所有现成机器人输出，不执行补充重定向或GPU训练。

BONES的36条及MotionMillion的2条没有可用caption，保留为SuperMimic纯动作参考，
没有伪造文本或将这些记录冒充GENMO文本训练样本。HumanML3D与KIT-ML文本完整。
原始文本CSV/JSON/SQLite无损压缩保留，并在本地对8份资料流式解压验证SHA。
MotionMillion原11,791,690条caption及split信息保存在原SQLite的压缩副本中。
音乐保留原配对索引，SuperMimic不增加音乐/文本输入，也不迁入音频特征作为模型条件。

报告与本地发布
--------------

* Server-1报告根：``/data0/user/liwei/datasets/bumi_dance_binary_20261030_v1/``。
* Server-2报告根：同名绝对目录，文本结果位于 ``text_quality/``。
* 本地目标：``/home/weili/supermimic_vibed/data/bumi3_4340/dance_binary_20261030_v1/``。
* 源数据：``source30/manifests/pass.jsonl``、原始NPZ/PT及99条无损规范化NPZ。
* 来源摘要、传输、全量数据覆盖及条件资料：``reports/``。
* 实际50Hz缓存及读取验收：``cache50/metadata.json``、``reports/cache_validation.json``。

本地导入由 ``/home/weili/supermimic_vibed/tools/import_dance_binary_dataset.py`` 执行，
逐文件SHA验证724518条输入，122846个与旧库内容完全相同的NPZ使用不可变硬链接复用。
其余文件从Server-1/2迁回。实测后将本次复制从4路改为8路并断点继续，原4路日志和
清单保留于 ``reports/transfer_workers4/``；没有停止模型训练或评价。
本次源清单SHA256为
``b6c943b60f3d0ebfab65db9d07b9fbae1034ef5e7b99975e10c175161cbfc831``。
源文件完整验证后才原子发布，旧Bones、旧七库数据及其缓存、模型、日志保持。

不改变GENMO Stage1分片、统计量、模型、Stage2冻结资产或正在运行的训练任务。
没有以这次数据筛选或缓存构建宣称新的控制器收益，没有启动正式训练。
本地SuperMimic默认数据切换及恢复旧模型时的显式路径用法，见其
``docs/BUMI_DANCE_BINARY.zh-CN.rst`` 和本次 ``记录文本.md``。

验证
----

新文本入口与公共筛选回归29项通过；新SuperMimic导入及现有缓存回归9项通过。
覆盖旧REJECT动作重新评估后PASS、原文件内容变化拒绝、整段保留，以及真实BUMI
运动学下侧卧PT到NPZ逐值保留、50Hz完整缓存和Goal570读取。
正式全量文件、缓存和默认路径的实际结果在本地报告及记录中追加，不能用这些CPU
回归代替完整数据发布，更不能代替物理执行、模型质量或实机验收。

本地替换已完成
--------------

完整缓存已成功构建并发布：228980820224字节，213.26GiB，包含命令启动和输入清单
读取共3614.165秒。724518条原始文件逐条SHA核验完成，所有缓存字段完整生成。
独立GPU验收核对全库名称、长度、索引，从八来源选30条首帧对照，根位置、根姿态和
关节位置误差均0；身体FK重算最大差小于5e-7。4096组GPU Goal570全有限且不跨动作。

本地SuperMimic的 ``experiment.py`` 默认数据目录及 ``prepare_motion_cache.py``
默认参数已切换到新库，保留旧BONES常量供历史对照工具使用。
实际调用训练入口且不传数据路径，已读到724518条、200082242帧，32组Goal570有限。
旧Bones与旧七库目录均在，旧模型和日志未覆盖；现有两个评价进程未终止。
当前 ``READY.json`` 为PASS、default_training_dataset_switched=true，
本次模型优化器更新与物理rollout均0，没有启动新训练。

本地SuperMimic已有大量未提交的用户改动，全部保留，未将动力学/部署等变化混合提交。
本次默认路径的独立差异及切换前源码保存在新数据 ``reports/``，便于审阅和恢复。
