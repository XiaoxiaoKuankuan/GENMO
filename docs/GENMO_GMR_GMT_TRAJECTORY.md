# GENMO → 完整 GMR → BUMI GMT 时序链路

## 目标与数据流

严格时序入口是 `scripts/demo/stream_smpl_params_to_gmt.py`。它与旧的
`stream_smpl_params_to_gmr.py` 并存，但两者不能同时向同一个 GMT Redis key
发布。

```text
完整 GENMO smpl_params.pt
  -> 真实 GMR-CPP 对完整序列做 BUMI3 IK
  -> 原生 MuJoCo 顺序 qpos[T,28]
  -> 30 Hz/源 FPS 四元数 SLERP + 位置线性插值到 50 Hz
  -> 对完整时间轴计算 root/joint 速度
  -> 每个 50 Hz tick 发布 trajectory_v1[110,55]
       packet = [过去10, 当前1, 未来99]
  -> GMT command_window 读取 packet[0:21]
       policy = [真实过去10, 当前1, 真实未来10]
```

55D 单帧的顺序固定为：

```text
root_xyz_world[3]
root_quaternion_wxyz[4]
root_linear_velocity_body[3]
root_angular_velocity_body[3]
joint_position_gmt_order[21]
joint_velocity_gmt_order[21]
```

平移和关节速度使用 50 Hz 完整时间轴的中心差分，首尾使用单边差分；root
角速度使用相邻单位四元数的 SO(3) 相对旋转；root 线速度从世界系旋转到当前
body 系。BUMI MuJoCo qpos 顺序通过关节名重排到 ONNX `joint_names`，不会依赖
容易写错的手工下标。

这里有两类“历史”，不能混为一谈：

- `command_window[1,1092]` 的过去/当前/未来来自上面的 qpos 参考时间轴，21 个
  slot 是不同时间点，不再复制一帧。
- `history_obs[1,690]` 仍由 GMT 控制器维护实际机器人/仿真的 10 帧观测历史；
  新 publisher 不伪造也不覆盖它。

## 协议与启动前提

publisher 只写 `OMGBT001 trajectory_v1`，默认 Redis key 是
`gmt_online_frame_bumi`，包长固定为 24304 bytes。启动时会从 GMT policy ONNX
校验：

- `policy=[1,69]`；
- `history_obs=[1,690]`；
- `command_window=[1,1092]`；
- 21 个 `joint_names`、默认关节姿态及 joint-order SHA256。

每个动作使用新的随机 `stream_id`。GMT 必须把匹配的 `OMGBTA01` ACK 写入
`gmt_online_frame_bumi_ack`，publisher 才启动动作时钟和音乐。ACK 超时或播放中
停止推进时，动作会被丢弃并切回合成站姿轨迹，避免“命令成功但机器人没有读到”
的假成功。

GMT 工程中必须已经包含 `GmtTrajectoryProtocol.h`，并重新编译一次：

```bash
cd /home/weili/docker_projects/bumi_GMT_deployment_listao/bumi_GMT_deployment_listao
catkin build rl_controllers
source devel/setup.bash
./simulation.sh
```

不要再额外运行一个把 legacy 35-float 数据写到
`gmt_online_frame_bumi` 的 GMR Redis publisher。完整 GMR 重定向由 GENMO
streamer 在隔离的 UDP 端口和临时 Redis key 上自动完成，最终 key 只接收
`trajectory_v1`。

## 常驻模式

先启动 GMT 仿真，再在另一个终端启动严格时序 streamer：

```bash
cd /home/weili/GENMO
source .venv/bin/activate
rm -f /tmp/genmo_estop

python -u scripts/demo/stream_smpl_params_to_gmt.py \
  --watch_dir outputs/music_motion_live \
  --source_filter music_only \
  --gmt_policy /home/weili/docker_projects/bumi_GMT_deployment_listao/bumi_GMT_deployment_listao/src/legged_rl/rl_controller/rl_controllers/policy/bumi/0724_lab_148500.onnx \
  --gmr_root /home/weili/GMR-CPP_e1jump_lowdpi \
  --ik_config /home/weili/GMR-CPP_e1jump_lowdpi/config/ik_configs/smplx_to_bumi3_auto.json \
  --cache_root outputs/gmr_bumi_cache \
  --capture_port 17016 \
  --redis_key gmt_online_frame_bumi \
  --publish_fps 50 \
  --audio_playback ffplay \
  --estop_file /tmp/genmo_estop
```

这个进程一直常驻：没有动作时发送由 GENMO 合成站姿经过真实 GMR 得到的静态
BUMI 轨迹；检测到新的 READY 后，先在后台完成整段 GMR 重定向和校验，期间站姿
继续以 50 Hz 发布。只有整段 qpos 可用并收到 GMT ACK 后，才平滑进入动作并播放
音乐。动作结束后平滑回到合成站姿，而不是停在舞蹈最后一帧。

首次处理一段动作时，完整 GMR 求解至少需要约等于动作时长的时间。结果按动作、
IK JSON、机器人 XML、GMR 脚本和地面参数的 SHA256 缓存在
`outputs/gmr_bumi_cache/`；相同输入再次播放不会重新重定向。GENMO 音乐扩散模型仍
由原来的 `demo_music_server.py` 常驻持有，不会因本 streamer 的请求重复加载。

watcher 启动前已经存在的 READY 默认不播放。只在明确需要重放这些旧目录时增加
`--replay_existing`。验证一个指定结果可使用：

```bash
python -u scripts/demo/stream_smpl_params_to_gmt.py \
  --motion outputs/music_motion_live/<generation>/smpl_params.pt \
  --gmt_policy /home/weili/docker_projects/bumi_GMT_deployment_listao/bumi_GMT_deployment_listao/src/legged_rl/rl_controller/rl_controllers/policy/bumi/0724_lab_148500.onnx \
  --gmr_root /home/weili/GMR-CPP_e1jump_lowdpi \
  --audio_playback ffplay \
  --once
```

## 安全与诊断

- 先在 `simulation.sh` 的 MuJoCo 中验证，不要直接把未检查动作发给实机。
- `touch /tmp/genmo_estop` 会丢弃当前动作和排队动作并持续发送站姿；删除文件不会
  恢复旧动作，必须再提交一个新 READY 才解除锁存。
- 日志出现 `WAITING_ACK` 后一直超时，优先检查 GMT 是否处于 ONLINE 模式、Redis
  key 是否一致，以及 `rl_controllers` 是否重新编译。
- Redis 中正确的 motion value 长度应为 24304 bytes，并以 `OMGBT001` 开头；
  140 bytes 表示仍有旧 GMR publisher 在覆盖 key。
- 日志中的 `publish=50.0Hz` 是 publisher 墙钟速率；`cursor` 是当前 50 Hz qpos
  下标。动作播放前必须出现 `[ACK] GMT accepted ...`。
