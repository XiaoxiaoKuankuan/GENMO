GENMO＋SuperMimic＋Adapter 音乐 MuJoCo 演示
========================================

此入口接受一个音乐文件，先用冻结 Stage1 GENMO 生成整首固定名义参考，再由当前
十帧 Adapter 和冻结 SuperMimic Actor 控制 MuJoCo 实体机器人。蓝色半透明影子是
同一绝对时刻的**原始 GENMO 参考**；没有位置对齐、时间平移或根轨迹纠偏显示。
实体每 20ms 执行一次控制，每次实际推进四个 5ms MuJoCo 物理子步。

本机直接运行
------------

已验证的 Python 环境为 ``/home/weili/GENMO/.venv/bin/python``，无需启动 Isaac Lab。
在本工作树运行，把最后一个参数换成自己的音乐文件：

.. code-block:: bash

   cd /home/weili/bumi-closedloop-worktrees/adapter/GENMO
   /home/weili/GENMO/.venv/bin/python -B scripts/demo/run_genmo_sm_adapter_sim2sim.py \
     --audio "/完整路径/音乐.wav"

本机已有音乐的完整例子：

.. code-block:: bash

   /home/weili/GENMO/.venv/bin/python -B scripts/demo/run_genmo_sm_adapter_sim2sim.py \
     --audio "/home/weili/bumi-closedloop-worktrees/GENMO/inputs/evals/stage1_fourset_20260928/selected_data/Mine/audio/dance_3__BoomBoomPow.wav"

默认使用本机 CUDA:0；如需指定设备添加 ``--device cuda:1`` 或 ``--device cpu``。
不主动占用 Server-1 的训练 GPU。默认开启声音；不使用音频设备时添加 ``--mute``。
支持已有 EDGE35 提取器可解码的 WAV/MP3/FLAC。首次新歌先提取音乐特征、生成参考，
完成后打开窗口；终端打印进度。下次播放相同歌曲与生成参数会复用只读缓存。

窗口操作
--------

* 空格：暂停/继续，声音同步暂停。
* R：相同音乐、相同参考、相同初态重播。
* B：零 Adapter／已训练 Adapter 之间切换，并从相同初态重新开始；模式显示在终端。
* G：显示/隐藏蓝色参考影子。
* Esc：退出。鼠标操作沿用 MuJoCo 原生相机控制。

整曲结束后窗口保留，可以按 R/B 对照。若真实机器人摔倒，只记录首次失败时间，
不扶正或覆盖实际位置；可以按 R 重播。非有限动作、状态及求解器数值异常会终止本次
演示并保存已有记录。原有训练进程和 checkpoint 不受影响。

模型和配置
----------

``configs/closedloop/adapter_sim2sim_mujoco.yaml`` 绑定明确的本机资产路径。
默认 Adapter 是 GPU7 第2600次更新的固定快照，不自动读取训练中的 latest。
Stage1 为 s595000；SuperMimic 为 ckpt_024000，SHA256 为
``172fbc720583f415fdf27560c91bb8c6f6efe8165738d961cf2455694789cb54``，
与 Server-1 使用的冻结 Actor 权重一致。音乐状态使用训练时真实基线 s0，
不另设在线归一化。需要换模型可显式指定：

.. code-block:: bash

   /home/weili/GENMO/.venv/bin/python -B scripts/demo/run_genmo_sm_adapter_sim2sim.py \
     --audio "/完整路径/音乐.wav" --adapter-checkpoint "/完整路径/模型.pt"

支持当前无硬导数投影的 Adapter checkpoint 和 weights-only 快照；2654输入、
270输出、根XY开关、根姿态开关和SM奖励尺度沿用 checkpoint 配置。不会把旧投影模型
静默转换成另一个控制方法。运行中无优化器、无 BPTT；Actor 原 bundle 中的 WM 不执行。

从零 Adapter 开始、无窗口有界检查
---------------------------------

.. code-block:: bash

   /home/weili/GENMO/.venv/bin/python -B scripts/demo/run_genmo_sm_adapter_sim2sim.py \
     --audio "/完整路径/音乐.wav" --zero-adapter

   /home/weili/GENMO/.venv/bin/python -B scripts/demo/run_genmo_sm_adapter_sim2sim.py \
     --audio "/完整路径/音乐.wav" --headless --mute --fast --max-seconds 10

``--max-seconds`` 只限制物理展示长度，不裁剪生成时的原音乐；音频末尾的真实名义
导数 halo 仍由 GENMO 生成，音乐有效 mask 已结束的位置不会复制末帧音乐。
``--fast`` 仅允许无声无窗口验证，不会修改 50/200Hz 的仿真时间步长。

输出与方法边界
--------------

每次运行在 ``adapter/sim2sim_runs/日期_唯一标识`` 新建目录，终端打印实际路径。
``launch.json`` 是该次运行的配置和来源记录，不是启动命令。``summary.json`` 保存
各次重播的物理步数、误差、首次失败时间、控制计算耗时与音频时钟；
``episode_000.npz`` 等保存真实 state55、执行动作、十帧残差、实际 Goal570 和原参考。
显式 ``--output`` 只能使用尚不存在的目录，避免覆盖此前结果。

这是 Stage2-A 的**固定参考交互演示**。任意音乐采用已有 Stage1 自主生成方式：
前12帧站姿，120帧窗口、12帧已生成前缀，DDIM20、CFG2.5、固定seed42。
历史只来自过去已生成名义动作，不读取未来真实执行。它不等于训练时用 Newton
执行历史建立的64首缓存，不等于 Stage2-B 在线滚动生成，也不能作为同一训练歌曲的
原 Newton 配对评价替代。MuJoCo 与 Newton 的接触求解差异需要独立评估。

本入口不改变训练算法、奖励、Actor或Adapter的输入合同。PD、动作尺度、armature、
力矩上限及关节顺序直接读取 RobotSpec；不通过设置真实 qpos 追踪参考。
影子只在独立 MjData 中做运动学渲染。summary中的SM回报是六项Tracking和动作变化项
等实际SM定义的结果，不是包含所有训练辅助成本的训练总目标。

2026-10-30 实际验证
------------------

真实 SM 权重下的新增4项测试和相关合同/音乐/向量控制回归共25项通过；验证包括
state55往返、世界角速度转换、命名关节映射、力矩上限、实际动作影响物理、影子不改
实体、零/非零 Goal/Actor/SM奖励与历史对齐。实际打开了交互窗口并核对蓝色透明影子，
另执行音频设备播放与 B/G/暂停/继续的窗口回调验证。

本机 RTX4090 上对49秒 BoomBoomPow，从相同 GENMO 参考、相同初态执行两次完整
MuJoCo物理（2450控制步/次），两者均无首次失败。GPU7-u2600关节RMSE为0.06939rad，
零Adapter为0.09150rad；根位置RMSE分别为0.25237m和0.14823m。该例展示关节改善
伴随根漂移增大，不能声称Adapter全面更优，更不能外推为全部训练集结论。

Adapter控制＋物理＋逐步统计P50/P95为7.95/8.33ms；零Adapter为7.61/8.24ms。
这些是无窗口实测计算耗时，包含CUDA结果回传同步，不包括生成参考与窗口绘制；
不能当作所有机器、所有歌曲的20ms最坏情况保证。5秒窗口实测同项P50/P95为
8.91/10.61ms。首步或调度抖动可超过20ms，缓存生成本身也不属于50Hz控制循环。

单独导出 Adapter ONNX
---------------------

正式入口为 ``tools/export/export_reference_adapter_onnx.py``。默认读取上面的本机
配置及GPU7/u2600固定快照，使用CPU导出并自动核对PyTorch与ONNXRuntime：

.. code-block:: bash

   cd /home/weili/bumi-closedloop-worktrees/adapter/GENMO
   /home/weili/GENMO/.venv/bin/python -B tools/export/export_reference_adapter_onnx.py \
     --output /home/weili/bumi-closedloop-worktrees/adapter/exports/adapter_g7_u2600

此目录已于2026-10-30实际生成，可直接使用，不必重复导出。重跑时换一个新目录；
入口拒绝覆盖已有文件。导出其它权重时添加 ``--checkpoint /完整路径/模型.pt``，
换机器时用 ``--config`` 指定本机资产配置。

目录包含 ``adapter.onnx``、``metadata.json`` 和 ``parity_report.json``。
metadata逐字段保存2654输入的名称、shape、扁平start/stop（左闭右开）、归一化要求、
21关节顺序、幅值上限、根通道开关、来源checkpoint SHA及更新数。ONNX输入名为
``adapter_input``，float32、动态 ``[B,2654]``；输出名为 ``residual_plan``，
float32 ``[B,10,27]``，对应t+1到t+10，仍相对原始名义参考而非上轮残差增量。

图包含MLP、Tanh限幅、根XY关闭、根姿态开关和0.05rad旋转向量范数上限。
输入特征拼接/归一化、历史音乐状态、SO(3)右乘组合、当前锚点、导数halo、Goal570
及冻结SM Actor仍属于控制程序。Adapter ONNX不能直接接收音乐文件，也不直接输出
21维控制动作。旧checkpoint的外部投影不会被打包到该网络图；必须使用其匹配运行时。

部署不需要WM、Critic、Replay或优化器；需要复用匹配的GENMO、SuperMimic Actor、
其归一化及音乐基线资产。本页的MuJoCo演示目前仍走PyTorch控制链，新增导出工具
没有将它改为整链ONNXRuntime。实际GPU7/u2600导出在batch1/16、零输入/随机输入/
100倍输入共6组核对中最大绝对误差6.333e-8，所有输出有限且满足训练限幅；这仅证明
导出数值一致，不新增真实物理收益或硬件部署结论。
