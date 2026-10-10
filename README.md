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

## 准备模型

在本部署目录操作，已有.venv可直接使用；缺少环境时先运行：

```bash
bash install.sh
```

安装器只准备依赖，不自动执行模型推理检查。随后手工构建当前模型的FP32 engine和部署清单：

```bash
bash scripts/export/build_bumi_stage1_engine.sh
```

默认读取本机
`/home/weili/bumi-closedloop-worktrees/GENMO/outputs/onnx/stage1_server2_s595000_20261008/bumi_stage1_denoiser_s595000.onnx`
及同目录ONNX元数据，并读取
`inputs/checkpoints/stage1_server2_s595000_20261008/`下的统计量与运动学资源。
源文件只读，不加载s595000.pt。

用户三次手工构建均触发同一Myelin内部异常；优化等级0、debug标记和显式Select广播
未解决该问题。当前策略改为`cuda_mask_plugins_int32_v3`：用独立IPluginV3 CUDA内核
执行动态Where/Cast/布尔比较及逻辑，GRU、Transformer、归一化、前缀和CFG仍来自原图。
Where按广播坐标逐位复制选中的分支，不用浮点掩码乘加，保留NaN/Inf隔离、注意力
负无穷及最终有效帧补零语义。编译前只折叠小型静态形状/逻辑常量，清除尺寸路径Where。

原始ONNX和采样器仍使用四个bool掩码；由于插件API不支持BOOL，派生TensorRT图内部
以0/1 INT32表示，运行器在拷贝持久缓冲时转换。十一输入的名称/尺寸、INT64时间步及
两项FP32输出保持，原始ONNX Runtime后端仍读取未改写ONNX。源模型及其元数据只读。
构建关闭TF32/FP16，默认优化等级0；不宣称禁用整个Myelin后端。
该策略已在TensorRT 10.13.3.9、CUDA 13 nvcc及RTX 4090上实际构建成功。
插件编译用-Xlinker传入实际版本化libnvinfer库，并用具名命名空间避免CUDA 13启动桩冲突。
已测片段的数值和短程生成耗时详见验收记录，不将其外推为整首播放或GMT动态验收。

构建入口发现ONNX缺少或版本不符时用已有uv/pip安装锁定的`onnx==1.18.0`，不执行推理验收。
插件编译需要已有CUDA Toolkit `nvcc`和匹配的TensorRT SDK；本机默认
`/usr/local/cuda/bin/nvcc`与`/usr/include/x86_64-linux-gnu`，不自动安装CUDA或修改驱动。
先编译插件，再生成独立`bumi_stage1_denoiser.trt.onnx`并解析构建，保存同目录
`network_lowering.json`，其中包括原图/派生图SHA、常量折叠、插件节点和构建身份。
缓存绑定插件源码、转换源码、共享库、模型及环境指纹，新策略不复用旧失败缓存；
重跑同一命令即可，不需要删除旧目录或传`--overwrite`。
成功后才原子发布v4清单，包含原ONNX、ONNX元数据、engine、engine元数据、统计、
运动学及`libbumi_stage1_mask.so`七项资产。运行器先校验并加载同包插件再反序列化。
输出：`models/bumi_stage1_s595000/deployment.json`。构建不执行推理预热或数值验收。
不同GPU或CUDA/TensorRT环境应在目标环境构建，旧engine不能复用为Stage1 engine。

可通过`BUMI_STAGE1_SOURCE_ROOT`、`BUMI_STAGE1_ONNX`、`BUMI_STAGE1_ONNX_METADATA`、
`BUMI_STAGE1_STATS`、`BUMI_STAGE1_KINEMATICS`、`BUMI_STAGE1_OUTPUT_DIR`和
`BUMI_STAGE1_DEVICE`覆盖源路径、输出路径和GPU。
额外参数透传Python构建器，例如`--workspace-gib 8`或明确的`--overwrite`。
`--optimization-level 0`可显式固定默认构建等级；其他等级只在用户明确传入时使用，
仍使用同一独立CUDA插件图，不视为已通过构建、数值或性能验收。
可用`--nvcc /path/to/nvcc`和`--trt-include-dir /path/to/include`指定编译器与匹配SDK。
使用自定义输出目录时，同时修改deployment.ini中的model.manifest。

## 引擎数值验收

构建入口不自动推理；需要复核时独立运行以下命令，输出写入系统临时目录。
默认20步DDIM、seed=42，下面三首音乐各读取开头167帧（约5.57秒），覆盖120帧首窗、
12帧续接前缀和59帧短尾窗，实际输出块为120+47帧。验收不连接GMT或发送动作。

```bash
cd /home/weili/GENMO-deploy-bumi
.venv/bin/python -B scripts/demo/validate_bumi_stage1_engine.py \
  --audio "/home/weili/下载/新春锣鼓.mp3" \
  --audio "/home/weili/下载/Bouncy sleigh bells.mp3" \
  --audio "/home/weili/下载/Robot Choreography.mp3" \
  --output /tmp/bumi-stage1-validation.json
```

结果中的`pass: true`表示所列数值检查通过。接触head同时记录logits与sigmoid概率误差，
不会把中间logits阈值调整隐藏为模型误差消失。资产身份由SHA256绑定，旧模型不进入验收。

## 在线三终端

终端1，容器内沿用GMT入口：

```bash
cd /host/Documents/bumi_GMT_deployment_obs
bash ./simulation.sh gmt_onnx_provider:=cuda gmt_cuda_device_id:=0
```

终端2，主机桥：

```bash
cd /home/weili/GENMO-deploy-bumi
bash scripts/demo/run_bumi_online_bridge.sh
```

终端3，主机控制台：

```bash
cd /home/weili/GENMO-deploy-bumi
bash scripts/demo/run_bumi_online_console.sh
```

在线增量生成沿用实时播放、高低水位、心跳和ACK逻辑。Stage1历史来自自身轨迹，
没有接入实际机器人反馈。默认桥为7022，Redis键为gmt_online_frame_bumi。

## 离线buffered三终端

终端1，容器内沿用现有完整缓存入口：

```bash
cd /host/Documents/bumi_GMT_deployment_obs
bash ./simulation_buffered.sh gmt_onnx_provider:=cuda gmt_cuda_device_id:=0
```

终端2，主机缓存桥：

```bash
cd /home/weili/GENMO-deploy-bumi
bash scripts/demo/run_bumi_buffered_bridge.sh
```

终端3，主机控制台：

```bash
cd /home/weili/GENMO-deploy-bumi
bash scripts/demo/run_bumi_buffered_console.sh
```

buffered默认桥为7023，Redis键为gmt_buffered_frame_bumi，强制关闭自动音乐播放。
输入音乐后完整生成，只有收齐整段才发送一个最终块并上传缓存。
GMT仍在每个仿真策略步前进一帧，保证的是仿真时间50Hz，不提高真实控制频率。

## 控制台操作与配置

仍按原操作进入GMT控制模式，在bumi>中输入：

```text
"/home/weili/下载/新春锣鼓.mp3"
play "/home/weili/下载/Bouncy sleigh bells.mp3" full --start 0 --seed 42
play "/home/weili/下载/Robot Choreography.mp3" 30 --start 0 --seed 42
status
stand
quit
shutdown
```

buffered生成结束后打印“整段…帧已生成，开始上传GMT缓存”。status中的bridge状态保持
`playback_mode: buffered`和`playback_clock: gmt_policy_step`。
qpos28、30Hz源数据、50Hz重采样、55维缓存、CRC、ACK和播放逻辑沿用原协议。

deployment.ini默认使用新Stage1模型和TensorRT；model.backend可设为onnx。
在线/buffered启动入口显式选择GMT连接和播放模式，不受保留的runtime.mode=preview影响。
原`bash run.sh genmo`仍按该配置运行本地预览。
GMT代码、policy、PD和容器脚本不修改；旧模型资产不会自动删除。
