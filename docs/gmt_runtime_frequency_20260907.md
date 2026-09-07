# GMT 部署终端真实频率监测（2026-09-07）

## 结果与使用

已在 `bumi_GMT_deployment_obs` 增加默认开启的频率监测，并在现有 noetic 容器重新编译控制器。
仿真、整段缓存播放和实物共用控制器，下一次启动即可生效。旧进程不会自动加载新库。
本次只增加统计和日志线程，没有把推理改成异步，没有启动仿真或实物、发送控制命令或操作 Redis。

在容器 `/host/Documents/bumi_GMT_deployment_obs` 中，原启动方式无需增加统计参数。
以下实物命令仅说明接口，未在本任务执行；机器人型号及原硬件配置仍需已校验匹配：

```bash
./real.sh robot_model:=bumi_4340 gmt_onnx_provider:=cuda gmt_cuda_device_id:=0
```

可选参数（仿真、实物、共享控制器 launch 均贯通）：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `runtime_frequency_log` | `true` | 是否创建统计线程、输出频率；`false` 完全关闭本次新增统计 |
| `runtime_frequency_log_interval` | `2.0` | 每次报告的真实时间窗口，允许 1～60 秒，不按仿真时间等待 |

例如在原命令后添加 `runtime_frequency_log_interval:=5`，改为每5秒输出；
添加 `runtime_frequency_log:=false` 关闭。参数在控制器初始化读取，不支持运行中热切换。

## 终端内容与正确解释

下面仅为输出格式示例，不是本次实物实测数据：

```text
[运行频率] mode=GMT provider=cuda wall=2.00s control=500.00/500.00Hz GMT_infer=50.00Hz GMT_command=50.00/50.00Hz 状态=平均接近目标(±1%，不保证无超时) gap_max=20.80ms gap>20.00ms=12 infer_failed=0 ros/wall=1.000
```

- `wall`：由 `std::chrono::steady_clock` 测量的真实经过时间；不受 `/clock` 暂停或系统日期调整影响。
- `control`：控制器 `update()` 完成次数/真实秒，以及配置的基础目标频率。不是物理引擎内部迭代次数，
  也不是电机内部电流环频率；计数位置在控制器内部，不是整个硬件 `read/update/write` 的分段耗时测量。
- `GMT_infer`：`computeActionsGmt()` 成功返回的次数/真实秒，不含失败帧。
- `GMT_command`：本帧全部 GMT 关节目标写入控制器命令缓存的次数/真实秒。基础循环重复保持旧目标不重复计数，
  不等于 EtherCAT 已送达、更不等于电机物理执行确认；Redis 接收 ACK 也不用于此计数。
- `gap_max`：当前窗口内观测到的新动作提交之间最长真实间隔；包含两次提交之间全部等待，不是 GPU 推理耗时。
- `gap>20.00ms`：超过目标动作周期的间隔数量，严格大于20ms即计入，微小调度偏差也可能计数。
- `infer_failed`：本窗口捕获到的 GMT 推理失败次数；保持原有失败退出处理，不继续计提交成功。
- `sim/wall`：仿真控制时钟推进量与真实时间的比值；实物显示 `ros/wall`，它是辅助时钟信息，不是频率计算分母。

目标来自当前配置的 `control_frequency / decimation`，现有仿真2000/40、实物500/10均为50Hz。
频率低于目标99%显示“**不足目标**”；99%～101%显示“**平均接近目标(±1%，不保证无超时)**”；
高于101%显示“高于目标”。这是窗口平均值口径，不将容差冒充严格达到50Hz或硬实时保证。
进入GMT/重置/模式切换的混合窗口显示等待完整窗口；非GMT不判定GMT频率。
控制计数不再增加时，独立日志线程仍会提示“无控制更新”，不能把暂停时的仿真50Hz误报为真实50Hz。
日志线程自身仍受普通Linux调度、CPU负载及终端速度影响；统计字段之间允许一个控制周期左右的采样边界偏差。

## 修改内容和理由

部署仓库路径：`/home/weili/docker_projects/bumi_GMT_deployment_listao/bumi_GMT_deployment_obs`。
下述代码位于 `src/legged_rl/rl_controller/rl_controllers`：

1. 新增 `include/rl_controllers/RuntimeFrequencyMonitor.h`、`src/RuntimeFrequencyMonitor.cpp`：
   固定大小无锁原子计数、分段标识、间隔统计、独立低频日志线程；控制路径不加日志锁、不打印、不分配统计容器。
   只有日志线程等待自己的条件变量并格式化消息，析构通知后回收线程，不引入无限排队。
2. `RLControllerBase.h/.cpp`：初始化统计开关，在控制器update末尾计数，使用原ROS日志入口输出。
3. `AcController.cpp`：仅增加首帧分段、推理成功/失败、完整新动作写入命令缓存的计数；
   不修改原观测、动作值、历史、模式切换、轨迹时钟或控制执行顺序。
4. `ac_start.launch`、`ac_start_real.launch`、`load_ac_controller.launch`：两个可选参数贯通，默认每2秒一行。
5. `CMakeLists.txt`：增加统计实现与测试目标；在最终动态库检查时发现原库已有两个TF未解析符号，
   对比修改前备份确认不是本次计数引入。补上已有package.xml声明的tf catkin组件及显式链接，
   不再依赖宿主进程预先加载libtf。该修改仅补齐编译依赖，不改变TF计算逻辑。
6. 新增 `test/test_runtime_frequency.cpp`：原生独立统计测试，不构造机器人控制器、不启动ROS节点。

保持不变：2000Hz仿真物理步进、分频40、实物500/10、显式PD、模型/精度、关节和电机映射、
Redis协议、轨迹、CPU默认/CUDA显式选择、WALK继续CPU。此前停止的仿真/实物未被本次启动。

## 验证结果与边界

- noetic `catkin build rl_controllers --no-deps -j4 -p1 --no-status --summarize` 成功；
  保留现有CMake/Python/gtest兼容性警告，TF未解析符号与对应依赖警告已消除。
- 新增10项统计测试通过：50Hz、42.8Hz且0.856倍仿真、暂停、非GMT、模式/分段切换、重复保持旧动作、
  推理/提交分离及失败计数、间隔阈值、配置拒绝、日志线程独立运行与快速回收、并发读写不丢计数。
- 原有ControlTiming 3项和GMT输出失败检查1项通过，共14项。修改TF链接后再次全部通过。
- 6种launch解析通过：仿真/实物 × 默认、关闭监测、5秒间隔并使用CUDA/4340。
  第一次实物解析测试未设置脚本本来会导出的ROBOT_TYPE而失败；按真实脚本补齐测试环境后通过，未修改硬件入口。
- 静态核对失败分支在提交前返回，计数落点在全部setCommand之后，重复保持目标不增加GMT_command。
- CPU和CUDA库路径下，`ldd -r`均无not found/undefined symbol，原生`dlopen(RTLD_NOW)`均成功；
  此项只加载共享库，不实例化控制器、不创建推理会话、不连接机器人。
- 最终控制器SHA256：`30ac8307f443cee9111969ee570e685a9fc3fb50435cc2b1c7881b6622d5d8ce`。
- 物理插件SHA256不变：`e7b19ad993f6d515f9915f87b689884ec7ac0e2ff3e0934500c25cd0257be6c2`。
- GMT模型SHA256不变：`d2e176657d72d1b0efcb04406abd17145fc45a3a5755b62aae7b866e0a6e3d1b`。

没有进行Gazebo闭环或实物控制测试，没有以此声明实际部署已达到50Hz；下一次用户启动后可直接观察新日志。
旧库及修改前主要源码保留容器 `/opt/bumi-gmt-ort/backups/pre-frequency.SLlXkb`，为可恢复的正式备份。
临时测试目录和本次构建日志在摘录结果后清理；测试源代码、正式编译产物及旧用户日志保留。
