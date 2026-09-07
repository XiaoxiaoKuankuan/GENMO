# Gazebo 500 Hz 底层振荡排查与隐式 PD 历史实验

日期：2026-09-07。实际部署目录：

```text
/home/weili/docker_projects/bumi_GMT_deployment_listao/bumi_GMT_deployment_obs
```

## 最新状态：隐式 PD 已撤回，频率已恢复2000 Hz

用户随后反馈 WALK 和 GMT 均异常，并要求恢复原 PD。生产插件现已移除隐式电机实现，
恢复原显式 PD；随后用户又要求恢复2000 Hz，当前为0.0005秒/2000 Hz/分频40。
显式插件本机已编译，此次频率恢复只改配置、重启加载。详见
[显式PD与2000 Hz恢复记录](gmt_explicit_pd_restore_20260907.md)。
下文完整保留当时的实验和判断，不是当前启动配置；定姿改善不能证明整条闭环根因，
旧版45.54 Hz数据不代表当前版本性能。两种策略的500 Hz适配问题未定位，本轮未重验2000 Hz效果。

## 1. 当时的实验结论

当前已重新配置为 **500 Hz 物理/基础控制、分频 10、策略按仿真时间 50 Hz**，并编译、运行了修改后的两个 C++ 库。这次不是只改 world 和分频，也不是退回 2000 Hz。

用户确认“开启控制或站立时，尚未进入 GMT”就开始转圈、发抖。隔离测试也在没有执行 GMT、没有 Bridge/Redis 的定姿阶段复现强振荡。主要证据指向：**原显式 PD 在放大四倍的物理步长下发生数值振荡**。另外确实漏改了按固定循环次数计时的姿态过渡。

同版本、同 500 Hz、同模型及原增益，仅切换显式/隐式 PD，固定机身定姿的关节速度 RMS 从 **10.6561 降为 0.07019 rad/s**。这是针对底层数值振荡的对照，不是对用户所有转圈现象的完整重放，也不等于全部运动质量已经与 2000 Hz 一致。

修改后完成一次原离线《火力全开》轨迹：**66.00 仿真秒用了 72.4569 真实秒，新策略输出平均 45.5443 Hz**。因此仍未达到真实 50 Hz；部署框架的串行推理、同步通信、诊断开销等优化需求仍然成立。

## 2. 之前遗漏了什么，这次怎么改

### 2.1 最主要的问题：不是频率数值没对齐，而是 PD 的离散计算方式不适应大步长

旧实现每步根据已经读到的位置、速度算一次力矩：

```text
力矩 = kp × (目标位置 - 当前位置) + kd × (目标速度 - 当前速度) + 前馈
```

之后把这个力矩保持整个物理步。2000 Hz 时每步 0.5 毫秒，500 Hz 时每步 2 毫秒。同一组较强的 PD 对小惯量关节作用四倍长的时间，可能先推过头，下一步再向反方向纠正，来回振荡并碰到力矩上限。**把策略分频从 40 改成 10，并不能修复这一底层数值问题。**

新实现仅在 BUMI3 Gazebo 中启用限力矩隐式 PD，让本步末位置、速度与物理约束一起求解。没有通过缩小 kp/kd、改模型或删保护来压抖动。实现公式如下，`h` 为实际物理步长：

```text
q_next = q + h × v_next
tau = kp × (q_des - q_next) + kd × (v_des - v_next) + ff
    = A - B × v_next
A = kp × (q_des - q) + kd × v_des + ff
B = kd + h × kp

ODE 软约束电机：target_velocity = A/B，CFM = 1/B，FMax = 允许力矩上限
```

采用独立 AMotor，而不是复用原 hinge 内的速度电机，避免电机与关节限位争用同一条约束行；原 hinge 继续负责位置约束。该表示及 CFM 单位已用真实 ODE 单关节测试核对，不仅是纸面推导。相关依据见 [Gazebo 内置 ODE 的关节电机/限位求解源码](https://github.com/gazebosim/gazebo-classic/blob/gazebo11/deps/opende/src/joints/joint.cpp)。

重要边界：**隐式与显式是不同的数值积分方式**。保留原 PD 参数和目标，不代表与旧 2000 Hz 的每一步响应完全相同。接触、延迟量化和整机跟踪仍须回归；这一实现不修改实机执行器。

### 2.2 保留的限制不能只剩一个最大力矩

原 ROS effort 限制器还处理位置越界、速度越界、软限位，某些状态下允许力矩是非对称区间，例如 `[-4, 0]`。本次继续调用原限制器，求得每帧的 `[lo, hi]`，转换为：

```text
bias = (lo + hi)/2
radius = (hi - lo)/2
总力矩 = bias + 电机力矩，电机范围为 [-radius, +radius]
```

这样保留原允许区间，而不是简单套一个对称最大力矩。急停时关闭独立电机，避免上一帧残余驱动；零反馈增益仍走原限幅前馈路径，无效数值拒绝执行并告警。

### 2.3 确实漏掉的固定循环计时

旧 `standDuration_ = 2 * 500` 固定为 1000 个基础循环：2000 Hz 时 0.5 秒，改成 500 Hz 后变成 2 秒。新代码以实际控制周期累计过渡进度，并显式配置持续秒数：

- 仿真：0.5 仿真秒，保持旧 2000 Hz 下的过渡时长。
- 实机：保留原 2 秒，不把仿真参数带到实机。
- 基础频率统一由 `RLControllerBase` 读取，移除 `AcController` 独立的旧默认值；备用 `GmtController` 的固定 `/500` 同步改为共享频率。当前启动的实际控制器是 `AcController`，备用路径不是此次振荡主因。

### 2.4 其它同步检查

- 首次速度差分从真实关节角初始化，避免父类默认 1 rad 引入初始尖峰。旋转关节速度差分和位置展开统一用最短角差，避免跨 ±π 的伪速度。父类初值见 [ROS 默认仿真硬件源码](https://github.com/ros-simulation/gazebo_ros_pkgs/blob/noetic-devel/gazebo_ros_control/src/default_robot_hw_sim.cpp)。同版本显式 PD 对照已经包含这些修正，仍有强振荡，因此它们不是主因解释的替代品。
- 新增启动检查：world 实际步长必须与 launch 的预期基础周期一致，混用 0.0005/0.002 时拒绝加载，防止以后只改一处。
- 实际 BUMI3 URDF 没有另设 `controlPeriod`，控制读写跟物理步进；P3D 状态发布原本已为 500 Hz，不改资产。ROS 插件的读/控制与写入调度见 [gazebo_ros_control 源码](https://github.com/ros-simulation/gazebo_ros_pkgs/blob/noetic-devel/gazebo_ros_control/src/gazebo_ros_control_plugin.cpp)。
- world 中历史 `<iters>2000</iters>` 不在有效求解器配置层级；运行时 `GetPhysicsProperties` 返回 `sor_pgs_iters=50`。它既不是 2000 Hz，也不是有效的 2000 次迭代，本次没有把它改成 500，未改接触求解器设置。
- `gazebo.delay=0.009` 保留，队列原本按时间戳而不是固定次数工作。2 毫秒网格下旧队列规则约选到 8 毫秒前的指令，0.5 毫秒网格下为 9 毫秒；离散步长变化不可能保证延迟逐帧完全等价，本次没有为提速删掉延迟。
- 试验过原生 `GetVelocity()` 替代位置差分，没有明确改善，已撤销该实验代码和参数；最终保留修正后的角差/周期。

## 3. 实际改动文件

以下路径均相对实际部署根目录，不涉及另一份同名 deployment checkout。

| 文件 | 本轮修改 |
| --- | --- |
| `simulation.sh` | 默认 500/10，继续关闭绘图/CSV，说明 C++ 编译要求 |
| `src/legged_rl/legged_base/legged_gazebo/worlds/empty_world.world` | 0.002 秒物理步长、500 目标更新率 |
| `src/legged_rl/legged_base/legged_gazebo/src/LeggedHWSim.cpp` 及对应头文件 | 限力矩隐式 PD、原限制器区间、急停、实际步长检查、初始角与角差修正 |
| `src/legged_rl/legged_base/legged_gazebo/include/legged_gazebo/ImplicitPd.h` | 新增不依赖 ROS 的隐式 PD 参数计算 |
| `src/legged_rl/legged_base/legged_gazebo/test/test_implicit_pd.cpp`、该包 `CMakeLists.txt` | ODE 数值、限幅、急停停驱动与生命周期回归 |
| `src/legged_rl/rl_controller/rl_controllers/src/RLControllerBase.cpp` 及对应头文件 | 统一频率、按秒推进姿态过渡 |
| `src/legged_rl/rl_controller/rl_controllers/src/AcController.cpp` 及对应头文件 | 使用共享频率，移除独立默认成员 |
| `src/legged_rl/rl_controller/rl_controllers/src/GmtController.cpp` | 备用路径周期除数改为共享频率 |
| `src/legged_rl/rl_controller/rl_controllers/include/rl_controllers/ControlTiming.h` | 新增按秒推进函数 |
| `src/legged_rl/rl_controller/rl_controllers/launch/ac_start.launch`、`load_ac_controller.launch` | 500/10、仿真 0.5 秒过渡、BUMI3 专用隐式开关、预期物理步长 |
| `src/legged_rl/rl_controller/rl_controllers/test/test_control_timing.cpp`、该包 `CMakeLists.txt` | 500/2000 Hz 秒制一致性、实机时长、非法周期回归 |
| `src/legged_rl/rl_controller/rl_controllers/test/probe_gazebo_servo.py` | 长期保留的隔离定姿/自由 WALK/离线 GMT 探针，默认不执行舞蹈 |

部署侧追加 `docs/CHANGELOG_SIMULATION.md`，GENMO 同步更新启动说明、历史基线文档、部署框架需求与根 `记录文本.md`。部署原有大量脏改动保留，不提交或覆盖整棵部署工作树。

## 4. 验证结果与没有通过的部分

### 4.1 编译、单元及配置

- `legged_hw_sim` 和 `rl_controllers` 两个生产目标编译通过，新进程实际加载修改后的库。旧依赖仍有 CMake/ODE 编译告警，不冒充零告警构建。
- 10 项单元测试通过：隐式解析式/CFM、力矩限幅、小惯量收敛、FMax 清零停驱动、非对称力矩区间、ODE world/motor 生命周期、非法参数，以及三项周期/时长测试。
- buffered/online/offline 实际 ROS launch 解析均为 500/10、过渡 0.5 秒、`implicit_pd=true`、预期步长 0.002；没有绘图/CSV/录包节点。实机仍 500/10、过渡 2 秒，不注入 Gazebo 隐式 PD 参数。
- SDF 解析、shell 语法、Python AST 和本次修改空白检查通过。Gazebo 两个历史 CRLF 文件保留原换行格式，使用识别 CRLF 的增量空白检查，未重排全文件。

### 4.2 最关键的同版本 500 Hz A/B

测试中仅在内存 URDF 固定机身、悬空排除落地接触；正式资产不改。不进入 GMT，进入 LIE 定姿后观测 7 仿真秒，统计末 3 秒的 21 个关节速度，1500 个采样时刻。两次均使用最终编译版本，仅切隐式开关。

| 指标 | 原显式 PD（开关关闭） | 新隐式 PD（开关打开） |
| --- | --- | --- |
| 关节速度 RMS | 10.65607357 rad/s | 0.07018656 rad/s |
| 最大绝对关节速度 | 31.37356799 rad/s | 0.32001097 rad/s |
| 观测 7 仿真秒所用真实时间 | 7.01367 秒 | 7.00373 秒 |
| ROS 会话 | `5dc0031a-aa95-11f1-9bdc-30560fa492c7` | `8edd916a-aa95-11f1-810a-30560fa492c7` |

RMS 降低约 99.34%，支持底层显式离散化为此复现条件下的主要振荡原因。仍有小幅残余数值波动，不称为零抖动。早期旧版本 2000/500 Hz 对照 RMS 分别为 1.0712/10.6449 rad/s；最终主证据以上面同版本、同 500 Hz 对照为准。

### 4.3 自由机身不是“固定住不抖就算成功”

测试初始化时临时使用零重力支持机器人，完成趴/站过渡，再进入 WALK 平衡策略，随后恢复正常重力。该支持仅用于测试夹具，不改正式启动/资产/保护。

- 2000 Hz 显式 WALK 短测：最低机身高度 0.4740 m，最大倾角 0.02418 rad，关节速度 RMS 0.2792 rad/s。
- 500 Hz 隐式 WALK 短测：最低机身高度 0.4452 m，最大倾角 0.06434 rad，关节速度 RMS 1.6578 rad/s；持续保持 WALK，但波动比该 2000 Hz 对照大，**不能宣称两者运动质量完全等效**。
- 曾在尚未进入 WALK、只有静态 STAND 时恢复重力，该测试倒地；因此没有把静态 PD 站姿当成自由平衡通过，也没有通过取消保护掩盖失败。之后采用有平衡策略的夹具，不能把不同初始化结果混为同一条件 A/B。
- 原生速度读数实验 RMS 1.9427 rad/s，没有清晰改善，未进入最终版本。

### 4.4 完整离线 GMT 与真实耗时

最终 ROS 会话：`f467678c-aa94-11f1-9aa2-30560fa492c7`。加载原文件：

```text
/home/weili/docker_projects/bumi_GMT_deployment_listao/bumi_GMT_deployment_obs/src/legged_rl/rl_controller/rl_controllers/motion_files/bumi/bumi/mine_bumi__dance_3__火力全开.npz
```

文件 3301 帧、50 Hz，首末帧跨度 `(3301-1)/50 = 66.00` 秒；loader 的 `3301/50 = 66.02` 是另一种时长口径，不能混用。

| 指标 | 实测 |
| --- | --- |
| 新策略帧间隔数 | 3300 |
| 仿真时间跨度 | 66.0000 秒 |
| 真实时间跨度 | 72.45692815 秒 |
| 按仿真时间的新策略频率 | 50.0000 Hz |
| 按真实时间的新策略频率 | 45.54429900 Hz |
| ROS 观测回调间隔 P99 / 最大值 | 33.633 / 77.191 毫秒 |
| 最低机身高度 | 0.36938 m |

没有触发探针的倒地终止条件（机身低于 0.25 m 或倾角大于 1 rad）。回调间隔包括 ROS 传输/调度，不是 ONNX `Run()` 耗时，也不是控制线程自身 deadline 的逐周期测量。

本次是 headless、无音乐生成/Bridge/Redis、关闭 GUI/rqt/手柄/绘图/CSV的离线运行，另有诊断订阅采样。**不是完整在线负载测试，不是 buffered 整链路本轮验收，也不是视频观感、足滑/跟踪误差或实机安全验收。**

用户此前 2000 Hz buffered 纯舞蹈约 110.68 真实秒，见[长期基线记录](gmt_buffered_2000hz_baseline_20260907.md)。两轮 GUI、通信、轨迹边界和采样负载不同，不能据此报告严格的同负载提速百分比。

## 5. 重启与复测

本机 `noetic` 容器已经编译完以下两个库，**现在只需关闭旧仿真并重新启动**。Python Bridge/Console 不需要为此次底层修复换模型或参数；旧运行进程不会自动加载新 `.so`。

在另一份构建环境复现时，需要编译：

```bash
source /opt/ros/noetic/setup.bash
source /host/Documents/bumi_GMT_deployment_obs/devel/setup.bash
cmake --build /host/Documents/bumi_GMT_deployment_obs/build/legged_gazebo --target legged_hw_sim -- -j2
cmake --build /host/Documents/bumi_GMT_deployment_obs/build/rl_controllers --target rl_controllers -- -j2
```

使用用户目前的独立缓存模式，容器内：

```bash
cd /host/Documents/bumi_GMT_deployment_obs
bash ./simulation_buffered.sh
```

原实时模式仍可使用 `bash ./simulation.sh`，底层同样应用新版 500 Hz；它不自动变成按仿真消费的 buffered 协议。缓存 Bridge/Console 命令见[独立模式说明](bumi_buffered_gmt_playback.md)。启动应看到 `implicit_pd=true, physics_dt=0.002000`、`posture_transition_duration=0.500s` 和控制频率 500、分频 10。

可复现探针支持 `--hz 500 --implicit-pd`；省略 `--implicit-pd` 是**测试用旧显式算法对照**，不代表生产默认关闭。必须为每次测试分配独立 `/tmp` 目录，使用私有端口，并在摘录结果后清理；不要把探针定姿模式当作实机控制入口。

## 6. 资产、构建指纹与清理边界

策略及 BUMI3 URDF 前后 SHA256 不变：

```text
model_135000_stage2.onnx
d2e176657d72d1b0efcb04406abd17145fc45a3a5755b62aae7b866e0a6e3d1b

BUMI3 URDF
041d843a30b16a6b93e2cce2ba99211f339ee2be84d36bd1ec1348cdcefb2592

本轮 liblegged_hw_sim.so
53527af0a95309de788d7ddcf227445a910ce7b32cf739ee69ca0f48730d29b9

本轮 librl_controllers.so
d6511ef9391bc1fb68a674bd921f3b32c2f62ce8cdd39074640a7ac57617524b
```

所有运行使用私有 ROS/Gazebo 11341/11342，没有连接生产 Redis，没有停止用户会话，也没有覆盖共享 URDF 导出。关键结果摘录入本文后，清理本轮主机 `/tmp/gmt_500hz_audit.aiSMv3` 与容器 `/tmp/gmt_500hz_audit.G3D1E4` 的临时代码备份、生成 world、日志/缓存和测试二进制；长期测试源码、生产编译库、用户原 CSV 和 2000 Hz 基线归档保留。清理结果在根 `记录文本.md` 记录。

后续应按[部署框架优化需求](gmt_real_50hz_optimization_requirements.md)分段测量、隔离推理/通信/诊断开销，并在同负载下验收真实 50 Hz 与运动质量，不能把这次底层数值修复当成整个性能问题已解决。
