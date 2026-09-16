# 同款 GMT 首次接入 GENMO：复制文件与逐处修改

适用条件：机器人、policy 的输入语义、关节契约、50 Hz 策略频率和原有 GMT 控制流程
都与当前 BUMI GMT 一样，区别仅是目标 GMT 尚未接收 GENMO。**GENMO 推理及 Bridge
保持现状，工作集中在 GMT 接收端。** 此处不讨论更换控制器或改用 SONIC。

当前参照工作区为 `bumi_GMT_deployment_obs`，控制器包路径为：

```text
src/legged_rl/rl_controller/rl_controllers/
```

下面“目标包”均指你要接入的那份 GMT 中此目录。目标代码尚未提供，本包不自动覆盖目标
工作区；示例对应当前同款 GMT 的类/成员名称。先保存目标分支，再在编辑器合并这些位置。

## 1. 哪些文件直接复制

把本目录下的三个文件按相同相对位置复制进目标包：

| 本包文件 → 目标包相同相对路径 | 操作 | 功能 |
|---|---|---|
| `include/rl_controllers/GmtTrajectoryProtocol.h` | 新增 | 解码轨迹包、校验关节顺序/CRC、构造命令窗口、编码 ACK |
| `include/rl_controllers/MotionLoaderRedis.h` | 替换同款旧单帧 loader；先比较并保留对方有效改动 | Redis 接收、完整轨迹缓存、ACK、序列和过期检查 |
| `test/test_gmt_trajectory_protocol.cpp` | 新增测试 | 协议、真实窗口、错误包、ACK、旧协议回归 |

这两个头文件包含完整实现，**无需再新增同名 `.cpp`**。接收器支持旧单帧输入；本次使用
`trajectory_v1` 的实时路径，不启用另一个 buffered 播放模式。来源及 SHA256 见
[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)。MotionLoaderRedis 副本仅补了文件开头的中文说明。

测试文件还使用目标 GMT 原有的 `MotionLoaderNPZ.h` 和 `third_party/cnpy`，二者沿用目标版本。
无需复制 GENMO 权重、Python 环境、导出工具或整个 `AcController.cpp`。当前参照工作区的
AcController 还有其他功能修改，整文件覆盖会把它们一起带入；下面只列 GENMO 接收链路。

## 2. AcController.h：确认接收器成员

文件：`include/rl_controllers/AcController.h`。已有的声明保留，不重复添加。

```cpp
// 文件头：加载完整轨迹接收器。
#include "rl_controllers/MotionLoaderRedis.h"

// 类的成员区：旧 online 路径通常已有 motionRedisGmt_。
std::unique_ptr<MotionLoaderRedis> motionRedisGmt_;
bool gmtOnlineCenteredWindow_ = false;
bool gmtOnlineWindowReady_ = false;
```

同款 GMT 应已具有 `joint_names_gmt_`、`actionsSizeGmt_`、`commandHalfWindowGmt_`、
`gmtCommandFeatureDim_`、`gmtCommandWindowSize_`、`commandWindowGmt_`、`gmtFirstStep_`
和 `isfirstRecObs_`。本协议下分别应得到21个关节、半窗10、单帧52、窗口21，总计1092。
这些值来自控制器原来的策略接口，不需要改变 policy 或它的观测布局。

## 3. AcController.cpp：只改三个接入点

### 3.1 loadMotions()：让接收器知道关节顺序和 ACK key

定位原来的 `gmtMode == "online"` 分支。保留原 Redis 地址/端口/db/key/timeout 的读取，
在创建 `MotionLoaderRedis` 之前增加：

```cpp
std::string redisAckKey;
int redisAckTtlMs = 1000;
nh.param<std::string>("/gmtRedisAckKey", redisAckKey, redisKey + "_ack");
nh.param<int>("/gmtRedisAckTtlMs", redisAckTtlMs, 1000);
nh.param<bool>("/gmtOnlineCenteredWindow", gmtOnlineCenteredWindow_, false);
```

把原来只传六个参数的构造调用改为：

```cpp
motionRedisGmt_ = std::make_unique<MotionLoaderRedis>(
    redisHost, redisPort, redisDb, redisKey, actionsSizeGmt_, timeout,
    joint_names_gmt_, redisAckKey, redisAckTtlMs);
```

新增的三个参数是**当前 GMT policy 自己的关节顺序、ACK key、ACK 存活毫秒数**。
不能把 GENMO 的 native 关节顺序传进这里。缺少名字时无法完成对应顺序的包校验。

加载顺序必须是 `loadModel()` → `loadRLCfg()` → `loadMotions()`。参照工作区在
`RLControllerBase.cpp` 中就是此顺序，所以创建 loader 时名字和关节数已经可用。
同款目标按此核对即可；不要在 policy 元数据还未读取时用空列表初始化接收器。

### 3.2 handleGmtMode()：每个策略周期接收、检查过期、跳过重复延迟

以下逻辑放在策略周期分支内部，**接收 update 之后、computeObservationGmt 之前**。
原有 NPZ 分支和电机控制逻辑继续沿用。

```cpp
// 原有的 update 分支保留，每个策略周期调用一次。
if (motionLoaderGmt_)
  motionLoaderGmt_->update(gmtMotionTime_);
else if (motionRedisGmt_)
  motionRedisGmt_->update(gmtMotionTime_);

if (motionRedisGmt_) {
  const auto protocol = motionRedisGmt_->protocolKind();
  if (protocol == MotionRedisProtocolKind::NONE ||
      (protocol == MotionRedisProtocolKind::TRAJECTORY_V1 &&
       !motionRedisGmt_->hasFreshData())) {
    ROS_WARN_THROTTLE(1.0, "[GMT] no fresh GENMO reference; return to DEFAULT");
    mode_ = Mode::DEFAULT;
    gmtFirstStep_ = true;
    gmtOnlineWindowReady_ = false;
    isfirstRecObs_ = true;
    return;
  }
}

const bool useCenteredLegacyWindow = motionRedisGmt_ &&
    gmtOnlineCenteredWindow_ &&
    motionRedisGmt_->protocolKind() != MotionRedisProtocolKind::TRAJECTORY_V1;
if (useCenteredLegacyWindow) {
  gmtOnlineWindowReady_ = motionRedisGmt_->hasCenteredWindow(commandHalfWindowGmt_);
  if (!gmtOnlineWindowReady_) return;
} else {
  // GENMO 包已经有未来参考，不需要再等10帧。
  gmtOnlineWindowReady_ = true;
}
```

进入 GMT 模式的首帧初始化中，保留或补齐：

```cpp
gmtOnlineWindowReady_ = false;
if (motionLoaderGmt_)
  motionLoaderGmt_->reset(Eigen::Vector3f::Zero(), 0.0f);
else if (motionRedisGmt_)
  motionRedisGmt_->reset(Eigen::Vector3f::Zero(), 0.0f);
```

如果原实现等待窗口时仍会在外层基础控制循环下发旧 GMT 动作，在原有电机下发循环之前
保留当前参照代码的条件：

```cpp
if (motionRedisGmt_ && gmtOnlineCenteredWindow_ && !gmtOnlineWindowReady_) return;
```

`.hasFreshData()` 按有效新序列到达时间判断，默认0.2秒；反复读到同一个 Redis 包不续期。
同款控制器已有的 DEFAULT 状态处理照常执行，不新增别的动作模式。

### 3.3 computeObservationGmt()：command_window 使用真实过去和未来

原来的 `rootPosW/rootQuatWxyz/rootLinVelB/rootAngVelB/targetJointPos` 读取可继续使用。
在准备 `commandWindowGmt_` 的位置接入 loader：

```cpp
// 放在已有 NPZ commandWindow 分支之后。
else if (motionRedisGmt_ && gmtNeedsCommandWindow_) {
  Eigen::VectorXf commandWindow = motionRedisGmt_->commandWindow(
      commandHalfWindowGmt_, gmtCommandFeatureDim_, gmtOnlineCenteredWindow_);
  const int expectedSize = gmtCommandWindowSize_ * gmtCommandFeatureDim_;
  if (commandWindow.size() == expectedSize) {
    commandWindowGmt_.resize(commandWindow.size());
    for (int i = 0; i < commandWindow.size(); ++i)
      commandWindowGmt_[i] = static_cast<tensor_element_t>(commandWindow(i));
  } else {
    ROS_ERROR_THROTTLE(1.0, "[GMT] invalid online command window: got=%ld expected=%d",
                      static_cast<long>(commandWindow.size()), expectedSize);
  }
}
```

这段沿用当前参照实现；如果目标已有等价调用，复制新版 loader 后会自动走真实轨迹分支。
不能继续“把当前一帧重复21次”或使用旧的未来预测替换包中的参考。

保持 GMT 自己原来的推理输入绑定：`policy [1,69]` 和 `history_obs [1,690]` 来自机器人
真实状态；只有 `command_window [1,1092]` 接入这份轨迹。原有 action scale、默认角、PD
和执行器下发流程无需因接收 GENMO 而修改。窗口尺寸异常的处理仍以目标控制器既有
错误退出机制为准；本移植包不证明目标完整控制器的故障处理已经通过验收。

## 4. launch：配置在线模式和 Redis，不指定新的 policy

文件：`launch/load_ac_controller.launch`。确认这些 arg 存在，已有的直接复用：

```xml
<arg name="gmt_mode" default="online"/>
<arg name="gmt_redis_host" default="127.0.0.1"/>
<arg name="gmt_redis_port" default="6379"/>
<arg name="gmt_redis_db" default="0"/>
<arg name="gmt_redis_key" default="gmt_online_frame_bumi"/>
<arg name="gmt_redis_ack_key" default="gmt_online_frame_bumi_ack"/>
<arg name="gmt_redis_ack_ttl_ms" default="1000"/>
<arg name="gmt_redis_timeout" default="0.2"/>
<arg name="gmt_online_centered_window" default="false"/>

<param name="gmtMotionMode" value="$(arg gmt_mode)"/>
<param name="gmtRedisHost" value="$(arg gmt_redis_host)"/>
<param name="gmtRedisPort" value="$(arg gmt_redis_port)"/>
<param name="gmtRedisDb" value="$(arg gmt_redis_db)"/>
<param name="gmtRedisKey" value="$(arg gmt_redis_key)"/>
<param name="gmtRedisAckKey" value="$(arg gmt_redis_ack_key)"/>
<param name="gmtRedisAckTtlMs" value="$(arg gmt_redis_ack_ttl_ms)"/>
<param name="gmtRedisTimeout" value="$(arg gmt_redis_timeout)"/>
<param name="gmtOnlineCenteredWindow" value="$(arg gmt_online_centered_window)"/>
```

`ac_start.launch` 和 `ac_start_real.launch` 中同名 arg 的默认值需对应，并在 include
`load_ac_controller.launch` 的位置逐项透传，例如：

```xml
<arg name="gmt_redis_ack_key" value="$(arg gmt_redis_ack_key)"/>
<arg name="gmt_redis_ack_ttl_ms" value="$(arg gmt_redis_ack_ttl_ms)"/>
```

无需改 `gmtPolicyFile` 指向的权重；只是确保该参数沿用 GMT 自己已有的策略路径。
策略目标周期保持50 Hz。上面的 `online` 是参考来源选择，启动后还需按 GMT 原有操作
进入 GMT 控制模式，接收器才会在其策略循环中 update 并产生 ACK。

## 5. CMakeLists.txt：真正编译启用 Redis 路径

在已有 `add_library(${PROJECT_NAME} ...)` 之后合并以下片段；已有等价依赖时直接复用。
Eigen、ROS 和 ONNX Runtime 继续沿用目标 GMT 原设置。
编译标准沿用同款工作区的C++17。

```cmake
find_package(ZLIB REQUIRED)
find_library(HIREDIS_LIB hiredis HINTS /usr/local/lib /usr/lib)
find_path(HIREDIS_INCLUDE_DIR hiredis/hiredis.h HINTS /usr/local/include /usr/include)
if(NOT HIREDIS_LIB OR NOT HIREDIS_INCLUDE_DIR)
  message(FATAL_ERROR "GENMO online receiver requires hiredis headers and library")
endif()
target_compile_definitions(${PROJECT_NAME} PRIVATE RL_CONTROLLERS_HAS_HIREDIS)
target_include_directories(${PROJECT_NAME} PRIVATE ${HIREDIS_INCLUDE_DIR} ${ZLIB_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${HIREDIS_LIB} ${ZLIB_LIBRARIES})

if(CATKIN_ENABLE_TESTING)
  catkin_add_gtest(test_gmt_trajectory_protocol test/test_gmt_trajectory_protocol.cpp)
  if(TARGET test_gmt_trajectory_protocol)
    target_compile_definitions(test_gmt_trajectory_protocol PRIVATE RL_CONTROLLERS_HAS_HIREDIS)
    target_include_directories(test_gmt_trajectory_protocol PRIVATE ${HIREDIS_INCLUDE_DIR} ${ZLIB_INCLUDE_DIRS})
    target_link_libraries(test_gmt_trajectory_protocol ${catkin_LIBRARIES} cnpy ${HIREDIS_LIB} ${ZLIB_LIBRARIES})
  endif()
endif()
```

系统构建依赖为 `libhiredis-dev`、`zlib1g-dev`，已有的 Eigen/gtest/cnpy/ROS 构建环境保留。
未定义 `RL_CONTROLLERS_HAS_HIREDIS` 时头文件会进入无 Redis 的占位实现，因此仅复制头文件
并不代表接收功能已启用。上述缺依赖时直接报构建错误，避免误以为 online 已可用。

编辑和复制完成后，在目标 GMT 容器中构建：

```bash
source /opt/ros/noetic/setup.bash
cd /你的GMT工作区
catkin build rl_controllers -j4 -p2 --no-status
source devel/setup.bash
```

这些命令只用于构建，不通过终端临时修改模型或端口。

## 6. 接收的具体内容，以及 ACK 为什么必须做

```text
GENMO（不改） → ZeroMQ 7022 → Bridge（不改）
                                   ↓ Redis 6379，key=gmt_online_frame_bumi
                        GMT MotionLoaderRedis
                                   ↓ 完整轨迹 → command_window
                          GMT自己的policy与控制循环
                                   ↓ 同一个Redis端口
                        gmt_online_frame_bumi_ack → Bridge
```

实时轨迹 magic=`OMGBT001`，50 Hz，110帧×55维，加104字节头，共24,304字节。
110帧为过去10＋当前1＋未来99；策略从中使用前后各10帧构造21×52=1092维命令。
55维为根位置3、根wxyz四元数4、机体系线速度3、角速度3、关节角21、关节速度21。
52维把根姿态转换为高度1和机体系重力3，再加两种速度与q/dq。

接收器按当前 policy 的名字顺序校验 SHA256，并检查长度、FPS、CRC、有限性和四元数。
合法新包才更新缓存和到达时间，并写52字节 ACK（magic=`OMGBTA01`，TTL1000ms）。
ACK 包含 stream/sequence/revision/plan，Bridge 收到匹配回执后推进播放和音频。
**只做 Redis GET、不给 ACK，这条现有 Bridge 链路仍不能正常播放。** ACK 是接收确认，
不是动作已跟踪完成。没有有效轨迹或0.2秒断流时，GMT按第3.2节退出在线跟踪。

## 7. 11311 有实际用途，本次保留

11311 是已有 ROS master/参数服务端口。Bridge 启动时执行：

```text
读取 /gmtPolicyFile 路径
  → 读取该 ONNX 的输入形状、joint_names、default_joint_pos
  → 构造 native→GMT 关节映射、关节顺序hash、空闲和返回站姿
```

所以“只用于读取路径”描述的是网络请求，后续还要读取路径对应的模型元数据，不能理解
为只打印文件名。Bridge 不运行 GMT policy 推理、不修改这个参数，也不通过11311传动作。
这不是 GENMO 额外启动的服务，同款 ROS GMT 原本已经有 master。

本次实读当前 GMT 模型：膝关节默认角约0.322 rad、踝pitch约-0.172 rad，GENMO配套
kinematics的21个默认关节角全为0，两者不能替代。native→GMT顺序也不同：

```text
[9,15,0,10,16,1,5,11,17,2,6,12,18,3,7,13,19,4,8,14,20]
```

本次按“GENMO和Bridge不动”保留现有发现。删除它需要另外给Bridge提供关节/站姿/形状
契约，例如JSON或Redis握手；那属于修改Bridge，不是删一段无用代码。仅保留一个固定
policy路径也能避开11311，但会重新引入用户不希望维护的policy选择。

目标同款GMT只需保留原有 `/gmtPolicyFile` 参数，且Bridge可读对应文件；容器路径通过
当前 `noetic` 的实际bind mount映射。不需要新增ROS topic、ROS message或订阅节点。

## 8. 实际验证边界

复制包与当前obs来源逐项比对指纹，原接收逻辑未改；本包在临时目录独立编译协议/Redis
测试，并使用临时Redis检查包解析、21帧窗口、ACK和旧协议。结果见移植压缩包内 `validation.json`。
测试所用进程、缓存和编译目录在结束时清理。

目标新GMT工作区尚未提供，因此未替它实际合并、重编完整控制器或执行仿真/实机。
验收时应依次核对：构建启用hiredis → Bridge握手/ACK → 1092维真实窗口 → 暂停发布后
GMT退出在线状态 → 保持原有控制行为的仿真验证。不能把独立接收器通过当作跟踪验证。
