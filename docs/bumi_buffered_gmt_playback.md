# BUMI 整首生成后按 GMT 仿真步播放

## 使用范围

这是与原实时模式并存的新模式：在 `bumi>` 输入音乐，先生成完整 30 Hz qpos，再按原有
SLERP/中心差分规则构建 50 Hz 轨迹，完整上传并缓存到 GMT；GMT 每执行一次 0.02 秒
仿真时间的策略更新，只消费一帧。生成计算本身尽快完成，不按任何 50 Hz 定时器限速。

此功能解决参考动作与慢仿真的时间轴错位，不提高 Gazebo 实时率，不意味着真实策略
频率已经达到 50 Hz。当前新模式默认且要求关闭自动音频播放；正常速度音乐在慢仿真中
仍无法同步。模型/checkpoint、CUDA ONNX EP、DDIM 20、CFG 2.5、seed 42、v5 CPU
规范初始噪声、足锁及 30 Hz 源时间边界保持不变。

## 启动（三个终端）

先关闭旧的仿真/Bridge/Console。不要在同一仿真中混用两个桥。

容器内启动 GMT：

```bash
cd /host/Documents/bumi_GMT_deployment_obs
bash ./simulation_buffered.sh
```

主机启动缓存桥（7023 端口，独立 Redis 键）：

```bash
cd /home/weili/GENMO
bash scripts/demo/run_bumi_buffered_bridge.sh
```

主机启动整首生成控制台：

```bash
cd /home/weili/GENMO
bash scripts/demo/run_bumi_buffered_console.sh
```

仍按原操作使仿真进入 GMT 模式。控制台输入：

```text
"/home/weili/GENMO/inputs/evals/bumi_s190000_mine10_20260904/mine_active/audio/dance_3__火力全开.wav"
```

同样支持 `play "路径" full --seed 42`、`stand`、`status`、`quit`。生成过程中只发布固定
站姿；看到“整段 1980 帧已生成，开始上传 GMT 缓存”才提交动作。GMT 应打印
`GMT BUFFERED: complete clip, one frame per simulation policy step` 和缓存帧数。
`status` 应显示 `playback_mode: buffered`、`playback_clock: gmt_policy_step`。
其中 `publish_hz` 仍是网络心跳频率，不是策略执行频率。

三个便捷脚本中的模型路径与用户当前 s190000 命令一致。也可以保留原全部参数，仅把
Python 入口替换为 `demo_bumi_gmt_buffered_bridge.py` 和
`demo_music_bumi_buffered_console.py`。缓存桥首次等待 GMT ACK 的默认期限为 300 秒。

2026-09-07配置更新：仿真入口现为500 Hz物理/基础控制、分频10，策略仍为50 Hz；
默认不启动GMT绘图和CSV共用节点。此轮只改配置，无需重新编译，重启仿真即可生效。
用户上一次2000 Hz运行的纯舞蹈约66秒、执行约110.68真实秒；详见
[基线记录与500 Hz调整](gmt_buffered_2000hz_baseline_20260907.md)。新配置的真实速度仍待运行测量。

## 接收端与编译

本机配套修改位于：

```text
/home/weili/docker_projects/bumi_GMT_deployment_listao/bumi_GMT_deployment_obs
```

修改 `GmtTrajectoryProtocol.h`、`MotionLoaderRedis.h`、`AcController.cpp`，新增
`simulation_buffered.sh`。接收端必须重新编译；仅重启 Python 不够。本次已经在 `noetic`
容器编译 `rl_controllers` 成功，之后需要重新启动仿真以加载新库。重新编译命令：

```bash
source /opt/ros/noetic/setup.bash
source /host/Documents/bumi_GMT_deployment_obs/devel/setup.bash
cmake --build /host/Documents/bumi_GMT_deployment_obs/build/rl_controllers --target rl_controllers -- -j2
```

接收端必须显式 `gmt_mode:=buffered`，要求 `/use_sim_time=true` 且控制频率/decimation
对应 0.02 秒策略步；拒绝直接用于实机。原 `simulation.sh` 和原实时 Python 命令行为不变。
部署工作树原本有大量既有修改；此次只局部修改上述文件，不提交或覆盖其他部署修改。

## 数据与时序

1. Console 完整生成后发送一个最终 qpos 块；不再在线高低水位节流，也不把续窗三秒
   门槛当作整首预生成失败条件。取消、身份/修订和安全校验仍然保留。
2. Bridge 上传 `OMGBF001` 完整缓存到 `gmt_buffered_frame_bumi:clip:<stream_id>`，含
   `[N,55]`、50 Hz、关节顺序 SHA256、CRC。轨迹包括缓入、动作、返回和尾部上下文。
3. 后续 `OMGBS001` 小窗口仅作为会话心跳；GMT 只在新 stream 首次读取一次完整缓存。
   所有 21×52 参考窗在 GMT 本地按当前帧索引提取，边界夹持，不能按发送次数跳帧。
4. 每次策略更新返回 `OMGBFA01` ACK，包含当前帧号和总帧数。Bridge 的显示、完成判断、
   返回衔接只跟随这个反馈，不使用真实时间 tick 推进。

新魔数使旧接收端无法把新模式误当实时轨迹。缓存为内存/Redis 临时数据，**不会自动
生成持久 NPZ 文件**；租约 60 秒、播放期间续租。每条源轨迹最多 39000 个 30 Hz 帧，
编码后的完整缓存不超过 65535 帧。仿真暂停时位置保持，网络心跳继续；Bridge 真正掉线
仍触发 GMT 原有数据过期保护。未进入 GMT、退出 GMT 或暂停时，帧号不会按墙钟前进。

## 已验证与边界

- 完整回归 `179 passed, 1 skipped`，包含既有实时链路；新测试覆盖生成收齐后才发送、完整缓存与 ACK 隔离、
  消费步/网络心跳解耦、重复包、暂停、末端夹持、错误模式、CRC 和过期保护。
- 使用真实 CUDA ONNX s190000、DDIM 20、CFG 2.5、seed 42 生成《火力全开》：22 窗，
  1980 源帧，3500 个含过渡和尾垫的 50 Hz 缓存帧；实际 C++ Redis 接收器完整逐帧比对
  21×52 参考窗，最大绝对误差 `3.36506301e-7`。
- 本次没有启动 Gazebo 闭环或实机；上述验证不代表策略动作输出、动态跟踪、正常速度
  音乐同步或硬件安全通过。也不声称与旧 CPU 离线 NPZ 完全相同：原有 CPU/CUDA 微小
  差异及 1980/1981 源帧时长边界仍存在。

测试均使用独立临时目录和专用 Redis 进程/键，不向生产 Redis 发送轨迹。
