# GMT 中用于接入 GENMO 的改动

本说明只针对容器 `/host/Documents/bumi_GMT_deployment_obs`，宿主机对应目录为
`/home/weili/docker_projects/bumi_GMT_deployment_listao/bumi_GMT_deployment_obs`。
本文不列出机器人描述、硬件驱动、其他动作模式等与 GENMO 在线接入无关的修改。
GMT 本次仅被读取和用于独立协议验收，不修改其工作树。

以下文件均相对于 `src/legged_rl/rl_controller/rl_controllers/`。

| 文件 | GENMO 接入所需的功能 |
|---|---|
| `include/rl_controllers/GmtTrajectoryProtocol.h` | 新增完整轨迹包解析、关节顺序/CRC/有限值/四元数检查、真实21帧命令窗口及ACK编码 |
| `include/rl_controllers/MotionLoaderRedis.h` | 在原单帧Redis读取基础上增加trajectory_v1识别、整包验证后更新、stream/sequence状态、ACK和新鲜度处理 |
| `include/rl_controllers/AcController.h`、`src/AcController.cpp` | 接入真实command window，向解析器传入policy关节契约，轨迹包绕过legacy centered-delay，无有效包/过期后回DEFAULT |
| `launch/ac_start.launch`、`ac_start_real.launch`、`load_ac_controller.launch` | 配置online模式、Redis地址/DB/key、ACK key/TTL及0.2秒数据超时，并将参数传入控制器 |
| `CMakeLists.txt` | 检查并链接hiredis，开启在线路径及相应协议测试 |
| `test/test_gmt_trajectory_protocol.cpp` | 轨迹结构、真实窗口、错误包拒绝、sequence、ACK与legacy兼容验证 |

`GmtTrajectoryProtocol.h` 的中文头注释明确将其定义为 GENMO 与 GMT 的在线协议。
原接口已能读取单帧 Redis 数据；新增的关键能力是完整未来参考、严格契约以及 ACK。

## 数据内容与顺序

Bridge 根据与 GMT 相同的 `model_135000_stage2.onnx` 元数据计算关节重排。
该文件 SHA256 为 `d2e176657d72d1b0efcb04406abd17145fc45a3a5755b62aae7b866e0a6e3d1b`。
模型输入分别为 `policy[1,69]`、`history_obs[1,690]`、`command_window[1,1092]`，
输出为 `actions[1,21]`。

Redis 的 `gmt_online_frame_bumi` 值是小端二进制 `OMGBT001`：104字节头＋110×55个
float32，总计24304字节，50 Hz。110帧包含过去10帧、当前1帧、未来99帧。
每帧55维为根位置3、根四元数wxyz4、机体系根线速度3、根角速度3、关节位置21、
关节速度21。接收端核验版本、形状、字节数、50 Hz、CRC32、joint_names SHA256、
所有数值有限性及四元数有效性；验证失败不能覆盖最后合法窗口。

GMT 取当前前后各10帧，逐帧转换为：根高度1＋机体系重力方向3＋根线速度3＋
根角速度3＋关节位置21＋关节速度21＝52维，组成1092维命令输入。每个时间槽使用
自己的真实参考，不把同一帧复制成整段未来。

## 时间、ACK 与故障处理

- `trajectory_v1` 已提供真实未来帧，所以接收器和控制器都绕过 legacy centered-delay。
  实机 launch 的 `gmt_online_centered_window=true` 不会给 GENMO 再叠加10帧延迟。
- 接收合法新包后写入 `gmt_online_frame_bumi_ack`，magic为`OMGBTA01`，52字节，包含
  stream、sequence、revision、plan和接收时间，默认TTL1000ms。
- 同流重复或倒序sequence不替换窗口、不重发ACK，也不刷新数据的新鲜度。
- 默认0.2秒没有有效新sequence，或尚无合法在线数据时，控制器切回DEFAULT并重置状态。
- Bridge通过对应ACK启动播放/音乐；这证明消费者收到参考，不能作为动力学跟踪证据。

Console到Bridge使用ZeroMQ，Bridge到GMT使用Redis SET/GET。此链路没有依赖通过
ROS topic传递生成动作。当前 `noetic` 使用host网络，因此两侧Redis地址均为
`127.0.0.1:6379`。本功能验收使用另一临时端口和测试key，不写生产动作key。

## 验收边界

原仓库验收脚本直接编译此工作区的协议单测和接收头文件，不启动ROS控制器。
接收探针检查1092维窗口、finite、centered true/false结果一致、ACK和停止发送后的
过期状态。它不加载GMT控制策略执行关节控制，因此协议通过不能表述为仿真或实机通过。
