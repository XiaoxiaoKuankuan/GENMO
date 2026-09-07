# 恢复原显式 PD 与 2000 Hz 的记录

日期：2026-09-07。修改对象是实际部署目录：

```text
/home/weili/docker_projects/bumi_GMT_deployment_listao/bumi_GMT_deployment_obs
```

## 最新状态：已按用户要求恢复 2000 Hz

用户随后要求停止500 Hz试改、恢复2000 Hz。当前配置为物理步长0.0005秒、基础控制2000 Hz、
策略分频40；仍使用已经恢复并编译好的原显式PD，继续关闭绘图/CSV。秒制姿态过渡仍为0.5仿真秒。
实机入口保持原500/10、2秒过渡，不受本次仿真配置恢复影响。

本次修改`simulation.sh`、`ac_start.launch`、`load_ac_controller.launch`和`empty_world.world`，
不改C++或模型，因此本机无需再次编译，重启原仿真入口加载即可。

已在noetic容器解析buffered/online/offline三种启动配置，均为2000/40、implicit=false、
expected_physics_dt=.0005，且无绘图/CSV/录包节点；实机解析仍500/10、2秒、没有隐式PD参数。
Gazebo SDF解析确认.0005/2000；world忽略注释和空白后与用户14:58的2000 Hz基线结构完全一致。
本轮没有启动新的Gazebo或策略运行，不能把配置恢复当作重新完成WALK/GMT稳定性或耗时验收。

500 Hz并非天然不可用，但本次同时改变了物理积分、反馈采样和力矩更新间隔；每个20ms策略步中，
基础步数由40变为10。现有模型不变也可能得到不同的执行与反馈。尚未用完整动态对照确定具体根因，
不再把定姿实验的结论推广到所有模式，不改模型来掩盖部署异常。

配置检查产物仅使用容器临时目录`/tmp/gmt_2000_restore.8IEoNJ`；复核后清理，结果记录于根日志。

## 先前恢复显式 PD 时的结论（当时仍为500 Hz）

用户反馈 WALK 和 GMT 都不正常，明确要求撤回 PD 求解方式修改。现已从生产 Gazebo 插件中移除隐式 PD/独立 AMotor，恢复原显式力矩计算，编译并在新隔离进程中确认加载。

**仍然是物理/基础控制 500 Hz、分频 10，没有回退 2000 Hz。模型、增益、动作缩放、轨迹、机器人资产未修改。** 这次完成的是 PD 恢复与有限的公共时序核查，不是 WALK/GMT 动态跟踪问题已解决。

之前固定机身定姿的隐式/显式对照，只说明该实验中振荡指标不同，不能据此将 WALK/GMT 所有异常归为 PD 算法本身的问题。之前隐式版本的 45.54 Hz 整首结果也不能当成当前恢复版本的性能。

## 恢复范围

- `LeggedHWSim.cpp/.h`：移除独立电机创建、CFM/FMax 设置、反馈合成、额外力矩区间求解与生命周期管理。恢复 `kp*(q_des-q)+kd*(v_des-v)+ff`，随后调用原 `DefaultRobotHWSim::writeSim()` 执行原限制器和写入；原命令延迟队列完整保留。
- 恢复后的 **整个 `writeSim()` 函数去掉注释/空白后，与部署仓库原版 HEAD 完全一致**，不只是比较一个公式。
- 保留之前与隐式无关的初始化真实角度、最短角差速度、秒制姿态过渡和步长一致性检查。并非撤回全部 500 Hz 配套修改。
- `ac_start.launch` 默认 `gazebo_implicit_pd=false`。旧命令显式传 `true` 会被插件拒绝，而不是静默启用另一套 PD。`simulation.sh` 更新编译说明，继续 500/10、关闭绘图和 CSV。
- `ImplicitPd.h` 和旧数值测试保留为明确标注的历史实验，不被生产插件引用，不用它们的通过结果验收当前显式版本。CMake 注释同步更新；未将测试源文件当作临时产物删除。
- `probe_gazebo_servo.py` 拒绝已撤回的 `--implicit-pd`，新增 `--startup-only`：仅在测试内存 URDF 固定机身，保持 DEFAULT，不发送开启控制命令，检查基础控制源时间戳。

## WALK 与 GMT 公共链路检查

| 环节 | 当前核对结果 | 验证边界 |
| --- | --- | --- |
| 物理步长/基础周期 | 0.002 秒；配置、运行服务及基础控制源时间戳一致 | 本轮只在 DEFAULT 采样 |
| 两种策略的触发 | `loopCount % decimation == 0`，500/10 对应 0.02 仿真秒 | 代码检查，不是两模式运行间隔实测 |
| 观测历史 | 在各自策略观测函数内推进，不随每个基础步直接推进 | 没有更改观测布局或模型 |
| 目标保持 | 策略间隙保留上一次 actions，每基础步设置关节目标 | 同样经过公共插件的命令队列和 PD |
| 状态读取 | 关节反馈通过硬件 handle，速度差分除以实际传入 period | 不把 Redis 当作 WALK 的机器人反馈通道 |
| 命令延迟 | 保留时间戳队列及 0.009 秒参数 | 2ms 网格下原规则约为8ms，旧0.5ms网格为9ms；存在量化差异，但未证明是异常根因 |
| 力矩写入 | 已恢复原显式 PD、父类限制器与写入 | 没有加滤波、调增益或替换模型 |

本次没有找到足以证明“两种策略实际被错按500 Hz执行”或“历史更新快了10倍”的证据。也不能仅凭公共链路的配置正确，宣布所有延迟、反馈及执行响应都正确。后续需在恢复后的显式版本、相同初始化和输入条件下记录两种模式的源时间、策略更新和执行响应，再定位差异。

## 实际验证

1. `noetic` 容器中生产目标 `legged_hw_sim` 编译成功。`rl_controllers` 本轮没有修改 C++，没有重新构建或更换其库。
2. 原版 `writeSim()` 源码一致性、生产插件无 AMotor 引用、Python AST、shell 语法及限定本轮文件的 CRLF-aware 空白检查通过。最初检查脚本误用子串 `dJoint` 匹配到了 `HybridJoint`，修正为标识符边界后通过；不是生产代码错误。
3. buffered/online/offline 三种 ROS launch 均解析为 500/10、`implicit_pd=false`、预期步长 0.002，没有绘图/CSV/录包节点。
4. 隔离 ROS 会话 `6bd8bc8a-aa9a-11f1-8572-30560fa492c7`，私有端口11341/11342，运行日志确认：

   ```text
   [LeggedHWSim] pd_mode=explicit, physics_dt=0.002000
   SERVO_PHYSICS 0.002 500.0 solver_iters=50
   STARTUP_RESULT: controller_mode=6 (DEFAULT), samples=504,
                   interval_ns=2000000, simulation_control_hz=500,
                   policy_executed=false
   ```

   时间戳来自基础控制器每次更新发布的 `world→base_link` TF header，而非订阅回调到达时间。只证明该采样区间的仿真基础更新间隔；不代表真实500 Hz，也不代表 WALK/GMT 已产生真实50 Hz新结果。
5. 没有进入 LIE/STAND/WALK/GMT，没有连接生产 Redis，没有生成音乐或发送舞蹈。关机时 ROS spawner 的卸载服务报 transport error，roslaunch 随后对自己的节点升级 SIGTERM；测试主进程退出0，后续检查自己的进程及端口残留。不能称为零告警运行。

## 编译与启动

本机插件已重新编译，重新启动仿真即可加载。原启动指令不变：

```bash
cd /host/Documents/bumi_GMT_deployment_obs
bash ./simulation_buffered.sh
```

原实时入口仍是 `bash ./simulation.sh`。Bridge/Console 不改；旧进程不会自动加载新库。本次不主动停止或热改用户会话。

新 `liblegged_hw_sim.so` SHA256：`e7b19ad993f6d515f9915f87b689884ec7ac0e2ff3e0934500c25cd0257be6c2`。
未变的 `librl_controllers.so`：`d6511ef9391bc1fb68a674bd921f3b32c2f62ce8cdd39074640a7ac57617524b`。

固定模型指纹（没有修改）：

- WALK `walkrun.onnx`：`780591e53027ccf722d7cccbfb005bd4d6040cc0a739136886a12537c8afcb2e`。
- GMT `model_135000_stage2.onnx`：`d2e176657d72d1b0efcb04406abd17145fc45a3a5755b62aae7b866e0a6e3d1b`。
- BUMI3 URDF：`041d843a30b16a6b93e2cce2ba99211f339ee2be84d36bd1ec1348cdcefb2592`。

测试输出仅在容器 `/tmp/gmt_explicit_restore.BpYYPt`。摘录以上结果后，复核并清理该目录中的生成 world、ROS/Gazebo 日志和缓存；结果记入根 `记录文本.md`。用户原基线、CSV、生产库与测试源码保留。
