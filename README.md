# BUMI Stage1 s595000 部署

本分支只支持当前Stage1十一输入网络，使用ONNX Runtime或TensorRT FP32。
部署运行时不加载训练checkpoint；音乐编码、历史编码、动作前缀和CFG已包含在ONNX图中。
图外保留当前Stage1的确定性DDIM、因果proprio48、120/12/108帧续接和qpos28解码。

迁移到`/home/user/liwei`、复用已有虚拟环境及CUDA12构建见
[liwei部署说明](docs/LIWEI_DEPLOYMENT_MIGRATION.md)。

2026-10-10已在本机RTX 4090实际编译CUDA插件、构建FP32 engine，并通过正式包加载、
原始ONNX单步对齐、完整20步DDIM及三首音乐片段的两窗口生成对齐。
验收覆盖模型生成，未连接GMT仿真或实机；完整证据与阈值见
[Stage1引擎验收记录](docs/BUMI_STAGE1_TRT_VALIDATION_20261010.md)。

2026-10-10已在目标机`/home/user/liwei`完成CUDA12迁移、共享环境保护检查、GMT原生
CUDA探针、Release编译、离线buffered及在线仿真，并确认Gazebo/MuJoCo窗口可用。
目标机结果、初始化条件、备份位置和验收范围见
[liwei迁移验收记录](docs/LIWEI_DEPLOYMENT_VALIDATION_20261010.md)。

以下运行步骤面向目标机，所有主机命令均在目标机执行。日常运行从
[图形显示准备](#图形显示准备)开始，然后选择[离线buffered模式](#离线buffered模式三个终端)
或[在线模式](#在线模式三个终端)。首次安装与构建仅在环境或模型未准备好时执行。
本次README更新只补充操作说明，未重新运行GMT、仿真或模型验收；上述历史结果保持原记录。

## 目标机目录与配置

```text
/home/user/liwei/
├── GENMO-deploy-bumi/
│   ├── .venv -> /home/user/robot_genmo_webcam/GENMO/.venv
│   ├── deployment.ini
│   ├── models/bumi_stage1_s595000/deployment.json
│   ├── sdk/tensorrt-10.13.3.9/include/
│   └── inputs/demo/
│       ├── 新春锣鼓.mp3
│       ├── Bouncy sleigh bells.mp3
│       └── Robot Choreography.mp3
├── bumi_GMT_deployment_obs/
├── runtime/bumi-gmt-ort/
└── migration/
```

新部署容器名为`bumi_genmo_gmt_deploy`。主机GMT目录在容器内对应
`/host/Documents/bumi_GMT_deployment_obs`；隔离运行库在容器内对应`/opt/bumi-gmt-ort`。
旧容器`bumi_sonic_deploy`已停止并保留，日常运行使用新容器。

目标机`deployment.ini`保留以下本地设置，其余模型、端口和Redis设置沿用部署配置：

```ini
[runtime]
mode = gmt

[model]
manifest = models/bumi_stage1_s595000/deployment.json
backend = tensorrt
device = cuda:0
ddim_steps = 20
guidance_scale = 2.5

[gmt]
ros_master_uri = http://127.0.0.1:11311
container = bumi_genmo_gmt_deploy
```

这只是关键字段摘录，不能用它覆盖完整`deployment.ini`。
`run_bumi_online_*.sh`及`run_bumi_buffered_*.sh`会显式选择GMT连接及对应播放模式，
不受仓库默认`runtime.mode=preview`影响；目标机的容器名仍须正确。
脚本直接使用部署目录`.venv/bin/python`，不需要先激活Conda或运行`source .venv/bin/activate`。

### GENMO模型与GMT策略的加载方式

GENMO从`models/bumi_stage1_s595000/deployment.json`加载当前Stage1模型、插件和配套资源。
GMT策略由GMT原有`load_ac_controller.launch`中的`gmtPolicyFile`决定，当前BUMI默认文件为：

```text
/host/Documents/bumi_GMT_deployment_obs/src/legged_rl/rl_controller/rl_controllers/policy/bumi/model_135000_stage2.onnx
```

下面的启动命令无需额外指定策略文件；`gmt_onnx_provider:=cuda`只选择GMT推理后端，
`gmt_cuda_device_id:=0`选择GPU。桥通过ROS参数读取实际GMT策略并解析容器挂载路径。
需要查看当前加载路径时，在仿真启动后的另一个容器终端执行：

```bash
source /opt/ros/noetic/setup.bash
rosparam get /gmtPolicyFile
```

## 首次安装与模型构建

目标机已经完成迁移、环境安装和CUDA12 engine构建；日常启动跳过本节。
如果需要重新准备部署环境，使用指定共享虚拟环境及CUDA12依赖：

```bash
cd /home/user/liwei/GENMO-deploy-bumi
bash install.sh \
  --venv /home/user/robot_genmo_webcam/GENMO/.venv \
  --tensorrt-cuda-major 12
```

目标机570驱动使用TensorRT 10.13.3.9 CUDA12包。安装器复用匹配依赖并补缺项，
不自动执行模型推理检查；无参数的`bash install.sh`默认选择CUDA13，因此本目标机使用上面的命令。
共享环境完整备份及迁移记录位于`/home/user/liwei/migration/`，
具体备份路径见[liwei迁移验收记录](docs/LIWEI_DEPLOYMENT_VALIDATION_20261010.md)。

需要重新构建时，明确使用CUDA12.8编译器和随部署包携带的TensorRT SDK：

```bash
cd /home/user/liwei/GENMO-deploy-bumi
BUMI_STAGE1_NVCC=/usr/local/cuda-12.8/bin/nvcc \
BUMI_STAGE1_TRT_INCLUDE="$PWD/sdk/tensorrt-10.13.3.9/include" \
bash scripts/export/build_bumi_stage1_engine.sh
```

构建默认读取部署包内四项源资产，不依赖原主机`/home/weili`路径：

```text
models/bumi_stage1_s595000/bumi_stage1_denoiser.onnx
models/bumi_stage1_s595000/bumi_stage1_denoiser.onnx.json
models/bumi_stage1_s595000/assets/qpos30_train_stats.json
models/bumi_stage1_s595000/assets/bumi_kinematics.json
```

源文件只读，不加载`s595000.pt`。构建成功后发布
`models/bumi_stage1_s595000/deployment.json`，在线与buffered控制台共用此清单。
更换GPU或CUDA/TensorRT环境时在目标环境重新构建；构建入口本身不执行推理或GMT验收。

## 图形显示准备

在目标机桌面会话打开终端，确认当前显示环境，并给root容器授权：

```bash
echo "$DISPLAY"
echo "$XAUTHORITY"
xhost +si:localuser:root
```

迁移时目标机桌面为`DISPLAY=:1`，Xauthority为`/run/user/1000/gdm/Xauthority`。
实际值以当前桌面终端输出为准，退出桌面或重新登录后可能变化；X11授权也属于当前桌面会话。
在SSH终端运行时，先从目标机桌面取得上述值，再设置到运行命令的主机终端。
仅当当前会话仍使用迁移时的值时，可执行：

```bash
export DISPLAY=:1
export XAUTHORITY=/run/user/1000/gdm/Xauthority
xhost +si:localuser:root
```

从本机连接目标机的命令为：

```bash
ssh -p 2222 -o ProxyCommand=none user@127.0.0.1
```

普通SSH登录不会自动提供目标机桌面的显示环境。终端1的容器Gazebo与终端3的主机MuJoCo
均使用目标机桌面显示；终端3也要继承或设置正确的`DISPLAY`和`XAUTHORITY`。
新容器已经具备X11挂载和图形驱动能力，无需每次重新创建容器。

日常启动前，在目标机主机查看新容器状态：

```bash
docker ps -a --filter name=bumi_genmo_gmt_deploy
```

若新容器显示`Exited`，先启动它；已显示`Up`时跳过此命令：

```bash
docker start bumi_genmo_gmt_deploy
```

主机Redis服务需要已运行于`127.0.0.1:6379`、DB 0；桥及控制台脚本不会启动Redis或GMT。
使用安装好的`redis-cli`可查看连接是否正常：

```bash
redis-cli -h 127.0.0.1 -p 6379 -n 0 ping
```

返回`PONG`后再继续。一次运行选择一种模式，启动顺序为终端1仿真、终端2桥、终端3控制台。

## 离线buffered模式：三个终端

离线buffered流程是：输入音乐 → 生成完整舞蹈 → 上传完整缓存 →
GMT每个仿真策略步前进一帧。自动音乐播放关闭，避免慢仿真下音画错位。
这里保证的是仿真时间下50Hz播放，实际每真实秒执行的策略步数取决于仿真速度。

### 终端1：目标机主机进入新容器，启动Gazebo与GMT

```bash
docker exec -it \
  -e DISPLAY="$DISPLAY" \
  -e QT_X11_NO_MITSHM=1 \
  bumi_genmo_gmt_deploy bash

# 以下命令在容器内执行。
source /opt/ros/noetic/setup.bash
cd /host/Documents/bumi_GMT_deployment_obs
bash ./simulation_buffered.sh gmt_onnx_provider:=cuda gmt_cuda_device_id:=0
```

等待Gazebo载入机器人、控制器及策略加载完成后，再启动桥。保持此终端运行。
`simulation_buffered.sh`已选择`gmt_mode:=buffered`及独立缓存键，其他控制和物理配置沿用GMT原脚本。

### 终端2：目标机主机缓存桥

```bash
cd /home/user/liwei/GENMO-deploy-bumi
bash scripts/demo/run_bumi_buffered_bridge.sh
```

桥监听`127.0.0.1:7023`，使用Redis键`gmt_buffered_frame_bumi`，
ACK键为`gmt_buffered_frame_bumi_ack`。等待桥完成GMT策略身份检查并开始监听，再启动控制台。

### 终端3：目标机主机生成控制台，带MuJoCo参考动作窗口

```bash
cd /home/user/liwei/GENMO-deploy-bumi
bash scripts/demo/run_bumi_buffered_console.sh --preview
```

`--preview`打开主机MuJoCo参考动作窗口，GMT物理跟踪效果在容器Gazebo窗口查看。
不需要MuJoCo窗口时，去掉`--preview`即可，buffered链路不变。
出现`bumi>`提示符后，按下节步骤进入GMT控制模式并输入音乐。

## 在线模式：三个终端

先退出上一套仿真、桥和控制台，再启动在线模式；不要同时运行两套ROS/GMT仿真。
在线流程增量生成并提交轨迹，沿用实时播放、高低水位、心跳与ACK逻辑。
Stage1历史来自自身生成轨迹，本轮部署没有接入实际机器人反馈。

### 终端1：目标机主机进入新容器，启动在线GMT

```bash
docker exec -it \
  -e DISPLAY="$DISPLAY" \
  -e QT_X11_NO_MITSHM=1 \
  bumi_genmo_gmt_deploy bash

# 以下命令在容器内执行。
source /opt/ros/noetic/setup.bash
cd /host/Documents/bumi_GMT_deployment_obs
bash ./simulation.sh gmt_onnx_provider:=cuda gmt_cuda_device_id:=0
```

### 终端2：目标机主机在线桥

```bash
cd /home/user/liwei/GENMO-deploy-bumi
bash scripts/demo/run_bumi_online_bridge.sh
```

桥监听`127.0.0.1:7022`，使用Redis键`gmt_online_frame_bumi`，
ACK键为`gmt_online_frame_bumi_ack`。音频播放沿用`deployment.ini`中的`bridge.audio_playback`，
当前为`ffplay`；在线播放以真实时间时钟推进。

### 终端3：目标机主机在线控制台，带MuJoCo参考动作窗口

```bash
cd /home/user/liwei/GENMO-deploy-bumi
bash scripts/demo/run_bumi_online_console.sh --preview
```

同样等Gazebo及GMT先启动、桥完成初始化，再启动控制台。
需要MuJoCo界面使用`--preview`，不需要时去掉该参数。

## 进入GMT模式与输入音乐

沿用原来的手柄或控制操作，顺序为启动控制 → LIE到STAND → WALK → GMT。
先确认机器人站姿正常，再进入WALK和GMT。若机器人已经倒地或触发原摔倒保护，
先按现有GMT流程恢复初始站姿再切换模式；具体初始化条件见
[迁移验收记录](docs/LIWEI_DEPLOYMENT_VALIDATION_20261010.md#初始化条件与失败记录)。
控制台`stand`用于桥返回站姿参考，与GMT控制器模式切换是两个操作。

如果使用ROS命令切换，可另开一个目标机终端进入容器；以下为已启用仿真的初始控制流程。
每条命令执行后确认日志/姿态再继续，尤其LIE到STAND需等待站姿过渡完成：

```bash
docker exec -it bumi_genmo_gmt_deploy bash
source /opt/ros/noetic/setup.bash
cd /host/Documents/bumi_GMT_deployment_obs
source ./devel/setup.bash

# 仅在尚未启动控制时执行：启动后进入LIE。
rostopic pub -1 /start_control std_msgs/Float32 "data: 2.0"

# 确认LIE后执行，切到STAND；与前次切换至少间隔0.8秒仿真时间。
rostopic pub -1 /switch_mode std_msgs/Float32 "data: 2.0"

# 确认站稳后执行，STAND到WALK。
rostopic pub -1 /walk_mode std_msgs/Float32 "data: 2.0"

# 确认WALK后执行，WALK到GMT；与前次切换至少间隔0.2秒仿真时间。
rostopic pub -1 /gmt_mode std_msgs/Float32 "data: 2.0"
```

`/start_control`及`/gmt_mode`具有切换行为，不能在已到达目标模式时反复发送；
暂停仿真时仿真时间不前进，先恢复Gazebo播放再操作。查看当前控制器模式可用：

```bash
rostopic echo -n 1 /data_analysis/controller_mode
```

模式编号为`0=LIE`、`1=STAND`、`2=WALK`、`3=DANCE`、`4=GMT`、`5=GMT_FUTURE`、`6=DEFAULT`。
准备播放时确认GMT模式为`4`，并在主机终端3的`bumi>`中输入音乐命令。
这里的路径属于目标机主机，不是容器内路径；含空格的音乐路径必须加引号。

播放三首完整音乐，逐首输入，等当前播放完成或先执行`stand`再切换：

```text
"/home/user/liwei/GENMO-deploy-bumi/inputs/demo/新春锣鼓.mp3"
play "/home/user/liwei/GENMO-deploy-bumi/inputs/demo/Bouncy sleigh bells.mp3" full --start 0 --seed 42
play "/home/user/liwei/GENMO-deploy-bumi/inputs/demo/Robot Choreography.mp3" full --start 0 --seed 42
```

仅输入带引号的路径默认播放完整音乐。若希望先运行前10秒，在同一控制台输入：

```text
play "/home/user/liwei/GENMO-deploy-bumi/inputs/demo/新春锣鼓.mp3" 10 --start 0 --seed 42
```

`10`为选取时长（秒），`full`为从`--start`位置到音乐结束，`--start`单位为秒，
`--seed`固定生成随机种子。这些语法同时适用于在线与buffered控制台。
buffered只有收齐整段后才提交动作，控制台将打印“整段…帧已生成，开始上传GMT缓存”；
生成期间尚未开始舞蹈。在线则按生成窗口增量提交，不等待整首生成结束。

## 查看状态、停止与退出

在`bumi>`中输入：

```text
status
stand
quit
shutdown
```

这些命令分别执行，不需要一次全输入：

| 命令 | 行为 |
|---|---|
| `status` | 查看生成状态、错误、桥状态及MuJoCo窗口状态 |
| `stand` | 取消当前生成/播放请求，让桥平滑返回站姿参考 |
| `quit` | 返回站姿后退出控制台与MuJoCo窗口，桥保留STAND并继续运行 |
| `shutdown` | 返回站姿后退出控制台、MuJoCo窗口和桥；Gazebo/ROS及Docker容器仍需独立退出 |

播放模式与时钟在`status`输出的`bridge`字段中查看：

| 项目 | 离线buffered | 在线 |
|---|---|---|
| `playback_mode` | `buffered` | `realtime` |
| `playback_clock` | `gmt_policy_step` | `wall_monotonic` |
| 桥端口 | `7023` | `7022` |
| 自动音乐 | 强制关闭 | 使用`bridge.audio_playback`配置 |

使用`--preview`且查看器正常时，`status.preview.alive`应为`true`，无查看器错误。
桥的`state: STAND`表示桥提供站姿参考，不等同于ROS控制器的STAND模式。
qpos28、30Hz源数据、50Hz重采样、55维缓存、CRC和ACK沿用原协议。

完整退出或切换在线/buffered模式时：

1. 在终端3输入`stand`，用`status`确认桥已回到`STAND`。
2. 输入`shutdown`，等待控制台和终端2的桥退出。
3. 在终端1按`Ctrl+C`退出ROS/Gazebo，等待launch结束；需要离开容器shell再输入`exit`。
4. 切换模式时重新按对应的三个终端步骤启动，重新进入GMT模式后输入音乐。

日常退出可以保留新容器运行；`shutdown`不执行`docker stop`。
只想关掉控制台、继续保留桥时使用`quit`，不要把它当作整套部署退出。

## 常见启动问题

| 现象 | 处理步骤 |
|---|---|
| Qt提示`could not connect to display`、X11授权失败 | 回到目标机桌面确认`DISPLAY`，执行`xhost +si:localuser:root`，按上面的`docker exec -e DISPLAY=...`重新进入；主机MuJoCo使用同一桌面环境 |
| 只有Gazebo，没有MuJoCo窗口 | 在主机控制台启动命令末尾加`--preview`，并检查`status.preview.error`；MuJoCo展示参考动作 |
| 桥无法读取GMT策略或ROS参数 | 先启动终端1，确认ROS/GMT加载完成；检查`deployment.ini`中的容器名与ROS地址，容器内查看`rosparam get /gmtPolicyFile` |
| Redis连接被拒绝 | 在主机确认Redis服务及`127.0.0.1:6379`、DB 0，使用`redis-cli ping`查看连接；脚本不会启动Redis |
| 控制台连接桥失败或播放模式不匹配 | 检查终端2与终端3使用同一套online或buffered脚本，GMT也使用对应仿真脚本 |
| buffered生成中Gazebo尚未跳舞 | 等待整段生成、缓存上传及ACK；确认ROS控制器已经进入GMT模式 |
| 机器人已倒地，切换模式后触发保护 | 按原GMT流程先恢复初始站姿，再进入WALK/GMT；参考迁移验收记录中的初始化条件 |
| 缺少清单、插件或engine身份不符 | 确认模型包资源齐全，按本机CUDA12.8/SDK构建命令重新生成清单，成功后再启动控制台 |

## 可选：独立引擎数值验收

本节供需要重新核验模型时手动执行，不属于日常启动步骤，本次文档更新未执行。
该命令不连接GMT或发送动作；输出写入系统临时目录。默认20步DDIM、seed=42，
三首音乐各读取开头167帧（约5.57秒），覆盖120帧首窗、12帧续接前缀和59帧短尾窗，
实际输出块为120+47帧。

```bash
cd /home/user/liwei/GENMO-deploy-bumi
.venv/bin/python -B scripts/demo/validate_bumi_stage1_engine.py \
  --audio "/home/user/liwei/GENMO-deploy-bumi/inputs/demo/新春锣鼓.mp3" \
  --audio "/home/user/liwei/GENMO-deploy-bumi/inputs/demo/Bouncy sleigh bells.mp3" \
  --audio "/home/user/liwei/GENMO-deploy-bumi/inputs/demo/Robot Choreography.mp3" \
  --output /tmp/bumi-stage1-validation.json
```

结果中的`pass: true`表示所列数值检查通过。接触head同时记录logits与sigmoid概率误差，
不会把中间logits阈值调整隐藏为模型误差消失。资产身份由SHA256绑定，旧模型不进入验收。
历史迁移报告、环境备份、容器配置及验收日志保存在目标机`/home/user/liwei/migration/`，
具体证据见[liwei迁移验收记录](docs/LIWEI_DEPLOYMENT_VALIDATION_20261010.md)。

## TensorRT构建策略与自定义参数

用户三次手工构建均触发同一Myelin内部异常；优化等级0、debug标记和显式Select广播
未解决该问题。当前策略为`cuda_mask_plugins_int32_v3`：用独立IPluginV3 CUDA内核
执行动态Where/Cast/布尔比较及逻辑，GRU、Transformer、归一化、前缀和CFG仍来自原图。
Where按广播坐标逐位复制选中的分支，不用浮点掩码乘加，保留NaN/Inf隔离、注意力
负无穷及最终有效帧补零语义。编译前只折叠小型静态形状/逻辑常量，清除尺寸路径Where。

原始ONNX和采样器仍使用四个bool掩码；由于插件API不支持BOOL，派生TensorRT图内部
以0/1 INT32表示，运行器在拷贝持久缓冲时转换。十一输入的名称/尺寸、INT64时间步及
两项FP32输出保持，原始ONNX Runtime后端仍读取未改写ONNX。源模型及其元数据只读。
构建关闭TF32/FP16，默认优化等级0；不宣称禁用整个Myelin后端。
该策略已在TensorRT 10.13.3.9、RTX 4090及原主机CUDA13、目标机CUDA12.8环境构建成功。
插件编译用-Xlinker传入实际版本化libnvinfer库，并用具名命名空间避免CUDA13启动桩冲突。
已测片段的数值和短程生成耗时详见验收记录，不将其外推为整首播放或实机验收。

构建入口发现ONNX缺少或版本不符时用已有uv/pip安装锁定的`onnx==1.18.0`，不执行推理验收。
插件编译需要已有CUDA Toolkit `nvcc`和匹配的TensorRT SDK，不自动安装CUDA或修改驱动。
先编译插件，再生成独立`bumi_stage1_denoiser.trt.onnx`并解析构建，保存同目录
`network_lowering.json`，其中包括原图/派生图SHA、常量折叠、插件节点和构建身份。
缓存绑定插件源码、转换源码、共享库、模型及环境指纹，新策略不复用旧失败缓存；
重跑同一命令即可，不需要删除旧目录或传`--overwrite`。
成功后才原子发布v4清单，包含原ONNX、ONNX元数据、engine、engine元数据、统计、
运动学及`libbumi_stage1_mask.so`七项资产。运行器先校验并加载同包插件再反序列化。

可通过`BUMI_STAGE1_SOURCE_ROOT`显式读取完整源仓库的导出资源，或者用
`BUMI_STAGE1_ONNX`、`BUMI_STAGE1_ONNX_METADATA`、`BUMI_STAGE1_STATS`、
`BUMI_STAGE1_KINEMATICS`、`BUMI_STAGE1_OUTPUT_DIR`和`BUMI_STAGE1_DEVICE`
覆盖源路径、输出路径和GPU。便携模型包是未设置这些覆盖项时的默认来源。
额外参数透传Python构建器，例如`--workspace-gib 8`或明确的`--overwrite`。
`--optimization-level 0`可显式固定默认构建等级；其他等级只在用户明确传入时使用，
仍使用同一独立CUDA插件图，不视为已通过构建、数值或性能验收。
可用`--nvcc /path/to/nvcc`和`--trt-include-dir /path/to/include`指定编译器与匹配SDK。
使用自定义输出目录时，同时修改`deployment.ini`中的`model.manifest`。

`deployment.ini`默认使用新Stage1模型和TensorRT；`model.backend`可设为`onnx`。
原`bash run.sh genmo`按`runtime.mode`配置选择运行方式；仅本地预览需相应设置为`preview`。
GMT代码、policy、PD和容器脚本不修改；旧模型资产不会自动删除。
