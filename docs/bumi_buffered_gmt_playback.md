# BUMI 整首生成后按 GMT 仿真步播放

## 使用范围

这是与原实时模式并存的新模式：在 `bumi>` 输入音乐，先生成完整 30 Hz qpos，再按原有
SLERP/中心差分规则构建 50 Hz 轨迹，完整上传并缓存到 GMT；GMT 每执行一次 0.02 秒
仿真时间的策略更新，只消费一帧。生成计算本身尽快完成，不按任何 50 Hz 定时器限速。

此功能解决参考动作与慢仿真的时间轴错位，不提高 Gazebo 实时率，不意味着真实策略
频率已经达到 50 Hz。当前新模式默认且要求关闭自动音频播放；正常速度音乐在慢仿真中
仍无法同步。2026-09-08 起独立便捷入口统一使用 **v5 s200000** 配套文件；CUDA ONNX EP、
DDIM 20、CFG 2.5、seed 42、v5 CPU 规范初始噪声、足锁及 30 Hz 源时间边界保持不变。

## 启动（三个终端）

先关闭旧的仿真/Bridge/Console。不要在同一仿真中混用两个桥。

容器内启动 GMT（2026-09-08 起，这个独立入口默认 CUDA 设备 0；普通 `simulation.sh` 仍默认 CPU）：

```bash
cd /host/Documents/bumi_GMT_deployment_obs
bash ./simulation_buffered.sh
```

等效显式写法是 `bash ./simulation_buffered.sh gmt_onnx_provider:=cuda gmt_cuda_device_id:=0`。
如需回退，使用 `bash ./simulation_buffered.sh gmt_onnx_provider:=cpu` 后重新启动仿真。
只修改启动默认值，复用已编译的 GMT CUDA 支持，本机不需要重新编译；WALK 保持 CPU。
直接 `roslaunch rl_controllers ac_start.launch gmt_mode:=buffered` 不经过这个脚本，仍需显式
添加 `gmt_onnx_provider:=cuda gmt_cuda_device_id:=0`。现有运行进程不会随脚本修改热切换后端。

### 30 Hz 和 50 Hz 分别是什么

- Console 生成的源 qpos 是 **30 Hz 轨迹采样率**；Console 已使用 `--onnx-provider cuda`，
  不是每秒只允许模型调用 30 次。Bridge 按既有插值和中心差分构建 **50 Hz GMT 参考轨迹**。
- GMT 仍为 **2000 Hz 基础仿真控制 / 40 倍分频 = 每仿真秒 50 次策略更新**，没有配置成 30 Hz。
- 如果终端显示 `GMT_infer=30.00Hz`，表示每真实秒完成约 30 次 GMT 推理；仿真若只有约 0.6 倍
  实时率，就可能出现这个读数。仅凭这一读数不能确定瓶颈在模型、物理、通信还是调度。
- 修改前，本独立仿真脚本没有指定 GMT 后端，因此不带参数启动会继承 CPU 默认值。
  修改后，默认显式选择 CUDA 并通过原 launch 给 gzserver 优先加载隔离 GPU ORT。
  这不改变物理步长、PD 或缓存推进方式，也不保证整条链路达到真实 50 Hz。

启动应看到 `[GMT ONNX] requested=cuda`、RTX 4090 设备及 CUDA 注册/预热成功；进入 GMT 后，
频率行保留 `mode=GMT provider=cuda GMT_infer=...Hz status=...`。实际频率仍要看本轮真实运行日志，
不能由离线视频 30 fps 或独立推理测试推断。

### 继续启动 Bridge 与 Console

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

独立 Console 的 checkpoint、ONNX、kinematics、stats 已由 s190000 切为 v5 s200000，
Bridge 的运动学路径同步指向同一归档；GMT 跟踪策略仍是 `model_135000_stage2.onnx`。
本次只改两个便捷 Shell 脚本的模型路径，不改通用 Python 入口，也不会替换已有手写命令中的旧路径。
缓存桥首次等待 GMT ACK 的默认期限仍为 300 秒。

配套目录（相对 GENMO 根目录）：

```text
checkpoint: inputs/checkpoints/bumi_5set_robot_retargeter_pass_v2_qpos30_contact_v5_s200000_20260907/s200000.ckpt
ONNX:       outputs/onnx/bumi_music/rr_pass_v2_5set_v5_s200000_20260907/bumi_music_denoiser_v5_s200000_t120_qpos30_contact.onnx
kinematics: inputs/checkpoints/bumi_5set_robot_retargeter_pass_v2_qpos30_contact_v5_s200000_20260907/assets/bumi_kinematics_robot_retargeter_fe934_v1.json
stats:      inputs/checkpoints/bumi_5set_robot_retargeter_pass_v2_qpos30_contact_v5_s200000_20260907/assets/bumi_qpos30_stats_train_5set_pass_v2_mine_fe934_v2.json
```

Console 自动读取 ONNX 同名 `.onnx.json` 元数据并校验 checkpoint、kinematics、stats 指纹。
重启上述 Bridge 和 Console 两个便捷脚本即可加载新默认值，无需重新编译，也不因本次模型路径变更而要求
重启已运行的 GMT/Gazebo；若还需应用前一次 GMT CPU→CUDA 后端切换，则仍须单独重启仿真。
请在当前动作停止后切换，不在播放中混用新旧 Bridge。原始模型文件和用户已有轨迹均保留。

2026-09-07最新状态：用户要求恢复2000 Hz，现为0.0005秒物理步长、2000 Hz基础控制、
分频40，仍使用此前已恢复并编译的原显式PD；保留0.5秒姿态过渡，继续关闭绘图/CSV。
此次只改配置，本机无需再次编译，重启原仿真入口即可。三种仿真模式及SDF解析已核对，
实机仍500/10；本轮没有重跑WALK/GMT，不把恢复配置当成稳定性或实际50 Hz验收。
详见[显式PD与2000 Hz恢复记录](gmt_explicit_pd_restore_20260907.md)。旧隐式版的定姿指标和45.54 Hz
离线整首数据仅作[历史实验记录](gmt_500hz_servo_audit_20260907.md)，不是当前版本性能。
用户之前2000 Hz buffered纯舞蹈约110.68真实秒，见[历史基线记录](gmt_buffered_2000hz_baseline_20260907.md)；
两轮运行负载不同，不能直接作为严格提速对照。

## 接收端与编译

本机配套修改位于：

```text
/home/weili/docker_projects/bumi_GMT_deployment_listao/bumi_GMT_deployment_obs
```

缓存协议修改涉及 `GmtTrajectoryProtocol.h`、`MotionLoaderRedis.h`、`AcController.cpp`，
新增 `simulation_buffered.sh`；最新500 Hz底层修复还涉及`LeggedHWSim`、基础控制周期等。
仅重启 Python 不够。本机已在`noetic`容器编译两个目标成功，需要重新启动仿真以加载新库。
在其它构建环境重新编译的命令：

```bash
source /opt/ros/noetic/setup.bash
source /host/Documents/bumi_GMT_deployment_obs/devel/setup.bash
cmake --build /host/Documents/bumi_GMT_deployment_obs/build/legged_gazebo --target legged_hw_sim -- -j2
cmake --build /host/Documents/bumi_GMT_deployment_obs/build/rl_controllers --target rl_controllers -- -j2
```

接收端必须显式 `gmt_mode:=buffered`，要求 `/use_sim_time=true` 且控制频率/decimation
对应 0.02 秒策略步；拒绝直接用于实机。原实时 Python 命令及协议不变；`simulation.sh`
与缓存入口共用当前2000 Hz仿真底层，但原实时模式不会自动变成仿真步消费。
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

2026-09-08 独立生成模型升级到 v5 s200000：

- checkpoint、ONNX、同名元数据、kinematics、stats 的实际 SHA256 均与归档清单一致，
  ONNX 元数据中的 checkpoint/kinematics/stats 路径及指纹相互匹配，checkpoint 为 step 200000。
- 两个 Shell 入口 `bash -n` 和真实 `--help` 参数解析通过；新增3项默认模型/两端运动学一致/用户覆盖检查，
  加原3项伪模型及缓存步进测试共 `6 passed`，3项 C++/真实生成集成测试未执行。
- 此轮只更新默认路径，不重新导出、修改或生成模型文件，不发音乐请求，不运行真实模型推理或闭环控制。
  临时测试产物已清理；旧模型、既有轨迹及用户进程均未改动。

2026-09-08 独立入口默认 CUDA 的本轮验证：

- 六项回归通过：运行真实 Shell 参数链并拦截 roslaunch，仅用 XmlLoader 解析，验证默认 CUDA、显式 CPU、
  设备覆盖、重复参数最后值生效，以及原在线入口默认 CPU/显式 CUDA 不变；gzserver 库路径、2000/40、
  原显式 PD、绘图关闭均通过检查。永久测试位于部署仓库 `rl_controllers/test/test_buffered_launch.py`。
- 原生 C++ 探针在设备 0 / RTX 4090 加载隔离 ORT 1.19.0，20 次初始化预热通过；8 组固定合成输入、
  168 个输出与原 CPU ORT 对比，最大绝对误差 `2.384185791015625e-06`，按原 `1e-4 + 1e-4*abs(cpu)`
  阈值没有超限。此为小规模接通回归，不替代此前完整数值/性能验收。
- Profiling 中 Gemm、MatMul、FusedMatMul 共 952 个执行事件均在 CUDA EP；本次没有记录 CPU 算子事件。
  因而不是仅凭配置或显存占用判断 GPU 生效。
- 未启动 Gazebo、ROS 节点、Redis 或实物，未测本轮闭环真实频率；未重编译、替换模型或重启用户进程。
  临时探针报告和 Profile 已自动清理，仅在文档与记录中保留结论。

以下为原缓存协议实现阶段的历史验证，不是本轮 GPU 入口修改后重新执行：

- 完整回归 `179 passed, 1 skipped`，包含既有实时链路；新测试覆盖生成收齐后才发送、完整缓存与 ACK 隔离、
  消费步/网络心跳解耦、重复包、暂停、末端夹持、错误模式、CRC 和过期保护。
- 使用真实 CUDA ONNX s190000、DDIM 20、CFG 2.5、seed 42 生成《火力全开》：22 窗，
  1980 源帧，3500 个含过渡和尾垫的 50 Hz 缓存帧；实际 C++ Redis 接收器完整逐帧比对
  21×52 参考窗，最大绝对误差 `3.36506301e-7`。
- 上述缓存协议开发阶段没有启动 Gazebo 闭环或实机；这些验证不代表策略动作输出、动态跟踪、正常速度
  音乐同步或硬件安全通过。也不声称与旧 CPU 离线 NPZ 完全相同：原有 CPU/CUDA 微小
  差异及 1980/1981 源帧时长边界仍存在。

缓存协议测试使用独立临时目录和专用 Redis 进程/键，不向生产 Redis 发送轨迹。后续用户
buffered运行及最新500 Hz离线闭环测试分别记录在上面的基线、底层修复文档中，不能混为同一轮验收。
