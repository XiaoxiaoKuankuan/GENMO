# GENMO 到 GMT、SONIC 和其他控制器的适配开发指南

本文以当前BUMI music-only链路为已实现基线，说明接入新控制器时保留什么、改哪里、
如何验证。文中的“建议新增”文件和接口是后续开发设计，不是已经实现的SONIC在线部署。
当前实际运行入口仍是 `demo_music_bumi_console.py` 和 `demo_bumi_gmt_bridge.py`。

环境、导出和启动命令见 [完整部署手册](BUMI_MUSIC_DEPLOYMENT.md)；
当前GMT的具体改动见 [GMT逐文件说明](BUMI_GMT_GENMO_INTERFACE.md)。

## 1. 首先确定连接边界

GENMO生成期望运动参考；控制器结合参考与机器人当前状态，产生控制动作。
GENMO的qpos不能直接当电机目标或控制器action，控制器的观测历史也不能用生成轨迹代替。

```text
生成端（基本固定）                    适配层（主要修改）                控制器端（保留自己的策略）
音乐→EDGE35→DDIM→qpos28/contact → 时间轴/关节/坐标/重采样/协议 → 参考缓存→本体观测+参考→policy→执行器
```

优先接入控制器的“参考动作/运动库/命令窗口”边界，保持它原有的policy加载、真实状态读取、
观测归一化、动作缩放、PD和执行器逻辑。只有参考API缺失时，才在控制器内增加接收缓存。

| 改动情况 | GENMO网络/ONNX/engine | Bridge/适配层 | 控制器 |
|---|---|---|---|
| 同机器人同契约，仅换GMT权重 | 保留 | 重启后读取GMT自己的配置 | 自行选权重 |
| 同机器人同参考含义，仅换传输协议 | 保留 | 新增编码器/发布器 | 使用自己的接收API |
| 同机器人，窗口或参考坐标系不同 | 保留 | 调整时间采样、坐标与字段 | 保持训练一致的观测组装 |
| 改用BUMI3 SONIC Robot Encoder | 可复用已核验同资产的BUMI qpos | 新增SONIC参考适配 | 新增在线参考缓存/接入点 |
| 改用SONIC SMPL Encoder | BUMI模型不提供SMPL；改用对应人体生成路径 | 人体关节/姿态适配 | 保持SMPL encoder契约 |
| 目标变成G1等不同机器人 | 不能按列数直接复用BUMI动作 | 需要重定向/另一生成表示 | 目标机器人自己的策略和资产 |

“同为21个关节”不是资产兼容证据。应比对名字、轴向/正负号、零位、根体、链长、
kinematics来源指纹和控制器MJCF/URDF，而不是把Bumi2、BUMI3、4340视为同一资产。

## 2. 每次接入先收集这张契约表

在编码前，从目标控制器的实际loader/配置/训练观测代码填写：

| 契约项 | 必须写清的内容 | 当前GMT基线 |
|---|---|---|
| 机器人 | 资产路径/版本、可驱动关节名字、顺序和零位 | BUMI21关节，名字从GMT自己配置的policy读取 |
| 根姿态 | 根体名字、xyz单位、世界轴、四元数排列及坐标定义 | 米、Z向上、根wxyz |
| 参考关节 | 绝对角还是相对默认角、rad还是degree | 参考q为绝对rad；控制器本体观测另处理 |
| 速度 | world/body坐标、差分方法和时间间隔 | 50Hz重算dq和根速度，根速度转机体系 |
| 参考频率 | Hz和时间戳语义，与物理/策略频率分别列出 | GENMO30Hz→GMT参考/策略50Hz |
| 时间窗 | 过去/当前/未来数量、stride、起始偏移 | 传输10/1/99；策略使用10/1/10，stride1 |
| 播放时钟 | 真实时间、仿真policy tick或消费者游标 | 当前默认wall_monotonic；buffered是另一协议 |
| 消息 | 字节序、dtype、shape、topic/key、CRC、序列、revision | trajectory_v1、Redis、小端float32 |
| 握手/反馈 | 接收确认还是已经消费，超时阈值 | ACK是接收确认，不是跟踪误差/消费位置 |
| 停止/换歌 | 谁取消旧任务、清buffer、回站姿、禁用控制 | Bridge与消费者各自状态机配合 |
| policy | 谁加载、观测字段/展平顺序/归一化、action语义 | GMT自主选择；Bridge不改policy |

ROS参数只能说明启动配置；如果控制器在运行中换权重而不更新契约，Bridge不能自行推断。
更通用的做法是由控制器发布“已加载模型对应的参考契约/版本/能力”，避免依赖读取权重文件。
此契约发布是后续适配建议，当前GMT没有新增这一发布接口。

## 3. 当前生成输出是什么，哪些代码可以复用

`gem/runtime/bumi_online_stream.py` 定义 `bumi_online_qpos_stream_v1`。
负载是小端 `float32[T,28]`，30Hz，布局如下：

```text
qpos[:, 0:3]   根xyz
qpos[:, 3:7]   根四元数wxyz
qpos[:, 7:28]  21个MuJoCo/native顺序关节角
```

JSON头包含request_id、revision、chunk_index、absolute_start_frame、total_frames、is_last、
source_fps、qpos_dim、quaternion_convention、qpos_order、CRC和完整生成身份。
当前模型身份绑定checkpoint/ONNX/实际推理文件/stats/kinematics/滑窗/足锁；这些是生成端
身份，不应该与某个控制器的policy SHA混成同一个必选模型资产清单。

| 现有代码 | 可复用内容 | 适配时注意 |
|---|---|---|
| `gem/utils/music_features.py` | 音频解码和EDGE35 | 控制器变化不影响音乐条件 |
| `gem/runtime/bumi_music_deploy.py` | ORT/TRT运行器、DDIM、滑窗和qpos输出 | 控制器变化一般不改生成模型 |
| `gem/robots/bumi/endecoder.py`、`feature_codec.py`、`kinematics.py` | stats解码和FK | 必须是生成模型配套资产 |
| `gem/robots/bumi/postprocess.py` | 因果足锁与连续后处理 | 不在新适配器重复叠加足锁 |
| `gem/runtime/bumi_online_stream.py` | 帧身份、CRC、revision、心跳和分块对象 | 其中仍有GMT窗口相关辅助逻辑，不能把整个文件称为完全通用 |
| `gem/runtime/qpos_timeline.py` | 跨块连续30→50Hz插值 | 当前实现针对30/50Hz；改频率需修改并测试，不能只改消息fps |
| `gem/runtime/bumi_robot_stream.py:BumiQposSafetyGate` | 有限性、根高、关节和速度检查 | 限值针对当前BUMI配置，按目标资产核验 |
| `scripts/demo/demo_music_bumi_console.py:BridgeClient` | begin/chunk/status/stand/heartbeat发送 | 新Bridge要兼容回复字段及状态语义 |

GENMO已经完成120/30滑窗融合，适配器收到的是连续最终qpos后缀；不要把每个chunk再当
独立120帧动作重置坐标、重置足锁或叠加30帧重叠。应跨chunk保留插值邻域、速度差分状态、
播放游标和根对齐变换，revision变化时才按协议清空。

## 4. 最小改动接入另一套控制器

若继续使用当前Console，最快的可维护做法是新增一个**按控制器类型命名**的Bridge入口，
沿用同一生产者协议，不为checkpoint/date复制脚本。以下文件是建议开发清单：

| 建议新增/修改 | 具体职责 |
|---|---|
| 新增 `gem/runtime/<controller>_reference.py` | 消费者契约、按名字重排、坐标/单位转换、重采样及参考窗口构造 |
| 新增 `gem/runtime/<controller>_transport.py` | 消费者自己的ROS/ZMQ/Redis/共享内存编码、发送和状态回读；不混入DDIM |
| 新增 `scripts/demo/demo_bumi_<controller>_bridge.py` | 接收原begin/chunk，管理revision/缓存/播放/停止，调用上面两个模块 |
| 消费端新增或修改reference loader | 把消息解码成它原本使用的运动参考对象，在线缓存与原离线motion loader共用观测构造 |
| 新增对应 `tests/bumi/test_<controller>_adapter.py` | 关节/坐标/时间窗和消息往返、断流/重复/换歌回归 |
| 原仓库新增对应独立验收入口 | 冻结一条真实qpos，校验离线参考和在线适配的逐帧一致性，再做闭环验证 |

GENMO Console当前会读取`state`、`future_buffer_seconds`、`last_error`以及请求/revision，
并按照STAND/准备/播放/返回状态协作；新Bridge不能只接收二进制不回复。可从现有
`BumiOnlineBridge.begin/accept_chunk/request_stand/status_locked`及Console的`_generate`逐一列出协议行为。
其线程结构、取消和背压经验可复用，但GMT ACK格式、110帧窗口和Redis key属于GMT适配。

更长期的重构可将原Bridge分成下面四个接口，再把GMT作为一个实现：

```python
# 设计示意，不是当前可直接import的接口。
contract = adapter.describe_reference_contract()
reference = mapper.convert(qpos_chunk, contract)  # 保留跨块状态
adapter.append(reference, request_id, revision, absolute_frame)
state = adapter.status()  # 区分 received / consumed / fault
adapter.stop(reason, revision)
```

建议新增通用`ControllerReferenceContract`和`ReferenceTimeline`，将关节、单位、根体、
FPS、窗口偏移、时钟/ACK能力显式描述；将当前`GmtPolicyContract`、`BumiIncrementalGmtPlanBuilder`
和`RedisTrajectoryPublisher`保留为GMT实现。只有完成测试后再给统一Bridge增加
`--controller gmt|sonic|...`工厂，不先把参数名字改成“通用”就宣称接口兼容。

## 5. 复用别人GMT代码时，最小移植面

先看对方是否已有完整参考轨迹输入：

1. 如果已有同样的 `OMGBT001` 接收和ACK，核对字节结构、关节hash和时间语义即可。
2. 如果只有单帧输入，在接收层补充未来轨迹缓存，并让观测构造读取真实时间位置；不能复制
   当前帧来伪造未来。按[GMT逐文件说明](BUMI_GMT_GENMO_INTERFACE.md)移植协议/loader/控制器分流。
3. 如果已有另一套完整轨迹API，在GENMO侧转换成对方格式更合适；无需让对方引入Redis。
4. 如果其policy只接受速度指令、目标位置或少量关键点，它不是任意全身qpos跟踪器；需要
   选择受支持的指令表示或训练相应tracker，不能靠消息格式转换获得缺失的控制能力。

对方可以沿用自己的policy名称、权重格式和模型管理。只要本体和参考契约发生变化，
都要以对方训练/部署代码为准调整适配，不使用当前GMT的69/690/1092作为通用标准。

当前自动发现模块 `gem/runtime/gmt_policy_source.py` 只适用于ROS1 `/gmtPolicyFile`方式。
对方没有这个参数时，在该控制器自己的contract provider实现读取配置/能力消息；不要
往其配置中硬塞本项目历史policy文件来满足路径检查。

## 6. SONIC具体怎么接

### 6.1 先分清本地两类SONIC入口

本节为2026-09-16对 `/home/weili/GR00T-WholeBodyControl` 的源码检查，未启动其控制任务。

| 路径 | 真实参考契约 | 对BUMI music-only的意义 |
|---|---|---|
| `gear_sonic/utils/mujoco_sim/bumi3_sim2sim.py`＋`gear_sonic/config/sim2sim/bumi3_sonic.yaml` | BUMI3 Robot Encoder：21关节、50Hz、未来10个参考点、stride5；480维参考＋690维本体历史=1170输入，21动作 | 若资产一致，可在其ReferenceMotion边界接qpos |
| 同一BUMI3运行器的SMPL encoder | 使用人体未来参考及自己的tokenizer | 当前BUMI qpos28没有SMPL pose/joints，不能通过改维度得到人体输入 |
| `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/zmq_endpoint_interface.hpp` | G1 C++部署的分版本ZMQ运动输入；有Robot、SMPL、联合、token路径 | 可参考其传输实现，但G1关节/模型不是BUMI21关节控制器 |

不能把以前SMPL GENMO→G1 SONIC的模式2链路视为已经支持本次BUMI qpos模型。

### 6.2 BUMI3 Robot Encoder的适配位置

已有接口：

- `Bumi3Contract`：读取关节顺序、50Hz控制时序、default/action_scale/PD等契约。
- `ReferenceMotion`：`joint_pos_policy`、`joint_vel_policy`、`root_position_world`、
  `root_quat_wxyz`、fps及可选根速度；这是适合接入GENMO参考的边界。
- `load_reference_motion()`：现有离线文件到ReferenceMotion的转换，可作为在线结果对照。
- `_build_robot_tokenizer()`、`build_observation()`：按真实状态与参考构造policy输入。
- `infer_action()`：控制器自行调用自己的策略，GENMO适配器不替换它。

建议实施：

1. GENMO侧接收30Hz qpos，按实际BUMI3资产校验后转换为50Hz，以名字映射到SONIC的
   `policy_joint_names`，填充ReferenceMotion字段。保留根z与相对轨迹；水平归零/heading对齐
   按SONIC已有约定做一次，不能在两端重复对齐。
2. SONIC侧新增 `StreamingReferenceBuffer`，提供当前帧、未来索引和可用范围查询；
   将离线`self.motion`访问抽象为同一reference provider。网络接收线程写不可变快照，
   50Hz策略线程读取，不在控制线程做网络等待或音乐特征提取。
3. 在 `run_bumi3_sim2sim.py` 增加选择离线/在线参考源的参数；保留自己的policy配置、
   机器人状态、观测历史、PD和仿真时序。
4. 新增收到/消费游标、revision和fault反馈，供Bridge控制预缓冲、音频启动、换歌及断流。
   当前GMT ACK二进制不能直接作为SONIC已消费帧的反馈。

本地Robot Encoder参考窗口是 `frame + [0,5,...,45]`，50Hz下覆盖0到0.9秒；
不是GMT的连续前后各10帧。参考480维按如下**整块展平顺序**拼接：

```text
flatten(10×21 joint_pos)
+ flatten(10×21 joint_vel)
+ flatten(10×6 relative_anchor_orientation_6d)
= 480
```

不是简单将每帧48维交错拼起来。相对朝向由**当前机器人anchor姿态**与未来参考anchor
计算，因此它应在能读取真实机器人状态的消费者侧构造，不能在GENMO端凭参考姿态独立
伪造。690维本体历史也由控制器的真实状态/历史action产生，不能由生成参考替代。

### 6.3 若接SONIC现成G1 C++ ZMQ接口

实际协议由`zmq_packed_message_subscriber.hpp`、`zmq_endpoint_interface.hpp`和
`streamed_motion_merger.hpp`定义：可选topic前缀＋固定1280字节JSON头＋按header顺序
连接的二进制字段；字段描述包含name/dtype/shape，版本决定必选字段。Robot版本要求
joint_pos/joint_vel，SMPL版本要求smpl_joints/smpl_pose，并带body_quat/frame_index。
这与GENMO Console→Bridge的REQ/REP multipart、GMT的Redis二进制都不同。

应新增对应packer及publisher，并严格匹配订阅端topic、版本、字节序、字段布局和帧合并
规则。先依据其实际机器人和encoder决定是否需要重定向或SMPL生成器；不能把BUMI的
21列关节填入G1数组后补零就称为适配完成。token-only模式还需要匹配的encoder，不是把
GENMO去噪网络的隐向量当SONIC token。

## 7. 一次适配应按什么顺序验证

| 阶段 | 必须检查 | 通过后能说明什么 |
|---|---|---|
| 契约静态检查 | 关节名字/顺序、root、单位、四元数、shape、时间窗、模型/资产身份 | 字段与代码假设相符 |
| 离线转换对照 | 使用冻结qpos；在线分块与离线整体重采样/差分一致；跨块连续；尾帧不丢 | 参考转换正确 |
| 协议测试 | 大小端、CRC/有限值、重复/乱序、revision、ACK/消费游标、断流超时 | 消息及状态机正确 |
| 假消费者联调 | 生成真实音乐但只接收记录，不驱动机器人 | 生成和通信可持续运行 |
| 控制器观测对照 | 同一机器人状态、同一参考，离线loader与在线loader构造输入一致 | 接入没有改变训练语义 |
| 仿真闭环 | 真实时间率、时延、跟踪误差、脚滑、跌倒、停止/换歌 | 特定仿真范围有效 |
| 实机流程 | 目标平台自身调试、保护和现场操作流程 | 单独的硬件验收，不能由前面阶段代替 |

推荐冻结最小样例：静止、单关节小幅正弦、纯根平移、纯yaw、跨chunk连续动作、非90倍数
尾帧、断流、换revision。对于单关节样例，应确认只有指定命名关节响应；这样比整段舞蹈
更容易定位左右交换、符号和零位错误。

调试时记录`source_frame`、重采样后的`reference_frame`、发布sequence、ACK/消费游标、
缓冲秒数及三类耗时（特征、采样、适配）。先冻结输入排除生成随机性，再定位消费者；
不同时改模型权重、坐标转换和PD参数，否则难以确认改动效果。

## 8. 本次完成范围

已实现：GMT策略从自身ROS参数发现、容器路径实际映射、GENMO v2六资产包、旧接口兼容、
环境/导出/构建/运行文档及当前GMT接收代码映射。

尚需另行实施：通用controller factory、控制器契约发布、SONIC在线参考缓存和publisher、
其他机器人重定向以及这些新路径的闭环验收。上述建议文件没有在本次伪造为空壳实现。
