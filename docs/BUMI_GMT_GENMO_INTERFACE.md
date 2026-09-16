# GMT 接入 GENMO：逐文件说明与移植清单

本次核对对象仅为：

```text
宿主机 /home/weili/docker_projects/bumi_GMT_deployment_listao/bumi_GMT_deployment_obs
容器内 /host/Documents/bumi_GMT_deployment_obs
```

下文文件相对于 `src/legged_rl/rl_controller/rl_controllers/`。这些是当前工作区已经存在、
GENMO在线链路依赖的实现；本次不修改GMT。移植到别人GMT时应逐项核对其已有能力，
按功能移植，不能覆盖对方整个AcController或把对方policy换成本项目旧文件。

环境和运行命令见 [部署手册](BUMI_MUSIC_DEPLOYMENT.md)，其他控制器包括SONIC的适配见
[通用适配指南](GENMO_CONTROLLER_ADAPTATION.md)。

## 1. policy仍由GMT选择

当前 `launch/load_ac_controller.launch` 在BUMI分支设置ROS参数 `gmtPolicyFile`；
`AcController.cpp:loadModel()`通过 `nh.getParam("/gmtPolicyFile", ...)` 读取它，
自行创建ORT session并读取joint_names、default_joint_pos、action_scale、PD等元数据。

GENMO Bridge的新默认行为只是读同一个ROS参数，通过实际容器挂载定位文件，再获取
关节顺序和默认姿态。Bridge不调用这个GMT模型做动作推理，也不写回参数。
正常GENMO命令没有`--gmt-policy`；GENMO资产清单v2不携带GMT权重。

这个自动发现方式没有新增GMT源码修改。若别人的GMT使用其他参数名、配置文件或不用ROS，
在适配器提供对应的只读contract provider即可。更通用的接口可由控制器发布已加载策略的
参考契约，避免生成端读取policy文件；这是后续设计，并非当前已有的GMT发布功能。

## 2. 新增完整轨迹协议，而不是只收一帧

**文件：`include/rl_controllers/GmtTrajectoryProtocol.h`**

主要入口为 `GmtTrajectoryV1::parse()`、`commandWindow()` 以及 `GmtTrajectoryAckV1`。
头注释明确说明GENMO与GMT的在线接口。移植时应把纯协议解析与ROS/硬件访问分离，使
解析器可以独立编译测试。

| 项目 | 当前trajectory_v1 |
|---|---|
| magic | `OMGBT001` |
| 帧率 | 50Hz |
| 窗口 | 110帧：过去10＋当前1＋未来99 |
| 每帧 | 55个float32 |
| 头部/总大小 | 104字节／24304字节 |
| 数值布局 | 根xyz3、根wxyz4、body线速度3、body角速度3、q21、dq21 |
| 时间/身份 | stream_id、sequence、发布时间、command_revision、plan_id、关节顺序SHA256 |
| 完整性 | 版本、header/frame/payload长度、FPS、CRC32、有限值和四元数有效性 |

解析顺序应为：验证整包外壳→核对尺寸和身份→验证payload→构造临时轨迹→全部成功后
更新可见快照。错误包不能先覆盖半个缓存，再试图回滚。

移植新关节数或字段时，应升级协议和测试，不能只改`55`常量而继续发送同一magic/版本。
完整布局以GENMO `gem/runtime/gmt_trajectory.py:GmtTrajectoryPacket`及C++解析器为共同依据。

## 3. Redis loader区分旧协议和完整轨迹

**文件：`include/rl_controllers/MotionLoaderRedis.h`**

需要保留/新增以下行为：

1. 在读取Redis值后识别`OMGBT001`，进入完整轨迹解析；保留对方原有单帧协议分支。
2. 用**当前控制器实际加载的policy关节名字**计算期待hash，传给协议解析器。
3. 整包通过后同时更新当前参考帧及过去/未来窗口。
4. 记录stream、sequence、revision、plan；同一流重复或倒序sequence不能覆盖新数据。
5. 新鲜度由有效新sequence到达的单调时间计算，不能由“Redis GET有返回值”刷新。
6. 在完整轨迹路径直接采用包中的q/dq和根速度；旧单帧差分/未来预测不能覆盖真实字段。
7. 只在合法新包接收后写ACK。

当前原有接口如 `update()`、`commandWindow()`、`hasFreshData()` 对AcController提供上述能力。
对方若使用共享内存、ROS消息或ZMQ，可换传输实现，保留同等轨迹/时间契约即可，不必引入Redis。

## 4. 从真实轨迹构造策略参考输入

**文件：`GmtTrajectoryProtocol.h`、`MotionLoaderRedis.h`、`AcController.h/.cpp`**

当前GMT的输入接口为：

```text
policy         [1,69]     当前机器人本体观测
history_obs    [1,690]    机器人状态/动作历史
command_window [1,1092]   21帧真实运动参考
```

它们不是三个都由GENMO生成。GENMO只提供运动参考，69/690仍由控制器采集机器人状态。
21帧命令窗口为当前前后各10帧，每个时间位置独立转换成：

```text
root_height1 + gravity_body3 + root_linear_velocity_body3
+ root_angular_velocity_body3 + joint_position21 + joint_velocity21 = 52
21 × 52 = 1092
```

根四元数用于把世界重力方向旋转到body系，不能将四元数4列直接代替gravity3列。
根线/角速度已经是约定body系，不要再次旋转；关节按名字对齐当前消费者顺序。
关节顺序hash是协议校验，不是自动修正机制，发送前必须已经重排。

`AcController.cpp:computeObservationGmt()`接入窗口；`handleGmtMode()`负责新鲜度和控制周期。
`AcController.h`保留相应loader、窗口就绪和模式状态。移植到另一模型时，以它训练时的
窗口/特征/展平顺序为准；1092不是所有GMT实现或所有控制器的固定输入。

## 5. 避免重复centered-delay

**文件：`AcController.cpp:handleGmtMode()`，以及loader的`commandWindow()`分流**

旧单帧输入需要先积累未来帧，当前代码用`gmt_online_centered_window`控制这一等待。
`trajectory_v1`已经携带真实未来，因此判断应按协议分流：

```cpp
// 逻辑示意：字段名参考当前源码，按对方控制器结构移植。
const bool wait_legacy_future = online_centered_window &&
                               protocol != TRAJECTORY_V1;
```

对于完整轨迹直接将窗口视为就绪，不能在已有未来参考之上再等待10帧。
本工作区实机launch即使默认centered=true，trajectory_v1仍走即时窗口路径。
验收探针比较同一轨迹在centered=true/false下的1092维结果，要求相等。

## 6. ACK只确认对应参考已接收

**文件：`GmtTrajectoryProtocol.h`、`MotionLoaderRedis.h`**

合法新轨迹接收后写回默认key `gmt_online_frame_bumi_ack`，magic=`OMGBTA01`，52字节，
包含stream、sequence、revision、plan和接收时间；TTL1000ms。

Bridge必须核对这些字段属于当前任务后再推进播放、启动音频。不能把上一首歌的ACK用于
新revision，不能把旧包反复ACK来掩盖发布端中断。ACK的时间戳和收到时间是通信诊断，
不是关节已经到位或动作已经消费完毕的证据。

如果对方需要严格按仿真步消费，应另提供消费游标/策略tick反馈及对应播放时钟；不能把
当前真实时间ACK语义直接改名为消费确认。当前`buffered`路径也是另一个显式协议，不能混包。

## 7. 断流与模式退出

**文件：`MotionLoaderRedis.h`、`AcController.cpp:handleGmtMode()`**

没有合法轨迹或trajectory_v1超过0.2秒无有效新序列时：

- 停止使用过期在线参考，切回控制器已有的DEFAULT逻辑。
- 重置首帧/窗口/观测初始化状态，防止下次上线沿用旧时间轴。
- 保留错误原因用于诊断；错误包不更新最后合法包到达时间。

不同控制器的DEFAULT/停止处理不同，应调用对方已有停止流程，不把切回某个模式当作
跨机器人通用保护保证。GENMO Bridge也会处理心跳、缓冲和返回站姿，两端的处理需配合。

## 8. launch和构建需要接上哪些参数

**文件：`launch/ac_start.launch`、`ac_start_real.launch`、`load_ac_controller.launch`**

上层入口应把参数传到真正初始化loader的地方，不能只在launch顶层声明：

```text
gmt_mode=online
gmt_redis_host=127.0.0.1
gmt_redis_port=6379
gmt_redis_db=0
gmt_redis_key=gmt_online_frame_bumi
gmt_redis_ack_key=gmt_online_frame_bumi_ack
gmt_redis_ack_ttl_ms=1000
gmt_redis_timeout=0.2
```

ROS参数实际使用GMT现有命名，如`gmtMotionMode/gmtRedisHost/gmtRedisKey/gmtRedisAckKey`。
保留对方policy配置和原有控制频率选择；确保参考/策略周期与消息50Hz契约一致。
当前仿真2000/40和实机500/10均得到50Hz，这不意味着所有控制器都需要2000Hz基础循环。

**文件：`CMakeLists.txt`**

Redis实现需探测并链接hiredis、为在线代码启用相应编译宏，协议独立测试依赖Eigen/gtest等。
移植时把依赖加到实际控制器target；只安装hiredis但没链接target，运行时仍没有在线路径。
对方使用其他传输则替换对应依赖，协议数学逻辑不应强耦合ROS。

在已配置的GMT容器工作区可以构建对应包，例如：

```bash
source /opt/ros/noetic/setup.bash
cd /host/Documents/bumi_GMT_deployment_obs
catkin build rl_controllers -j4 -p2 --no-status
source devel/setup.bash
```

此处是移植后的构建指令，不表示本次执行过GMT重编或仿真。修改前后都保留对方原有有效改动。

## 9. 测试哪些代码

**GMT文件：`test/test_gmt_trajectory_protocol.cpp`**

应覆盖：Python/C++关节hash一致，ACK字节布局，当前及21个真实参考帧，错误版本/长度/
CRC/FPS/关节/非有限值/四元数拒绝，重复/旧sequence处理，ACK和旧单帧协议兼容。

**GENMO文件：**

- `tests/test_gmt_trajectory.py`：Python端轨迹协议和重采样相关回归。
- `tests/bumi/test_bumi_online_deployment.py`：分块、帧序、revision、后处理和Bridge行为。
- `tests/bumi/test_gmt_policy_source.py`：读取GMT自己的ROS参数、模型选择变化、容器挂载和失败边界。
- `tests/fixtures/bumi_gmt_receiver_probe.cpp`：直接包含当前GMT头文件的独立接收探针。
- `tools/eval/validate_bumi_gmt_deployment.py`：临时Redis/ZMQ/XML-RPC参数夹具，真实模型→Bridge→真实C++接收器。

参数夹具只模拟ROS的只读getParam，并从当前obs的BUMI launch读取策略配置；不会启动ROS
控制器。它验证自动发现路径和通信，不能替代真实控制器加载、仿真跟踪或硬件验收。
本次GMT自身代码保持未修改；要把上述能力移植到别人工作区，应在对方分支独立验证。
