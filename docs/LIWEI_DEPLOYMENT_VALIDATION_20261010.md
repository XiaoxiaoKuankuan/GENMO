# 2026-10-10 liwei部署迁移与仿真验收

本记录对应用户批准的独立部署迁移：GENMO与GMT统一放到目标机`/home/user/liwei`，
GENMO复用既有虚拟环境，GMT复制原工作树和容器环境。记录区分模型数值、轨迹协议、
图形和短片段仿真结果；不将短验收当作三首整曲动态评估或实机验收。

## 实际目录与环境

- GENMO：`/home/user/liwei/GENMO-deploy-bumi`，分支`deploy/bumi-music-only-gmt`。
- GMT：`/home/user/liwei/bumi_GMT_deployment_obs`。
- 隔离运行库：`/home/user/liwei/runtime/bumi-gmt-ort/1.19.0-cu124-cudnn9`。
- 备份、正式报告与日志：`/home/user/liwei/migration`。
- 部署`.venv`链接`/home/user/robot_genmo_webcam/GENMO/.venv`。
- GPU为RTX 4090；驱动仍为570.211.01；插件编译使用CUDA 12.8 nvcc、TensorRT 10.13.3.9 SDK。
- TensorRT frontend、bindings、libs均为CUDA12的10.13.3.9包。实际`libnvinfer.so.10`
  来自共享环境的`tensorrt_libs`，`libcudart.so.12`来自`nvidia/cuda_runtime/lib`。

原共享环境完整备份位于：

```text
/home/user/liwei/migration/genmo-venv-20261010T103500Z-1772114/.venv
```

最终比对确认所有原有包版本、editable安装的direct_url及原有`.pth`均未改变；仅增加
七项缺失运行依赖和三项TensorRT包。`python -m pip check`输出`No broken requirements found.`。
原GENMO三个修改文件`demo_webcam.py`、`test_demo_webcam_sonic_cli.py`、`记录文本.md`
的指纹与迁移前一致，未提交或覆盖这些修改。

## 源码与资产身份

GENMO迁移起点为`23c857272dd6ebb4b9c3b7cbf68cea3695a0cf54`。安装与便携构建入口调整
提交为`6a1fe7c8232d36ff8b97542de912cbf37bd8b8a9`，后续记录一并通过单分支bundle同步；
最终源/目标HEAD、分支、目标配置差异保存在`migration/genmo-final-snapshot.json`。

GMT复制935项实际源文件，共981788784字节；复制前后及编译、仿真后均比对内容哈希和
文件权限。源码、控制器、策略、Redis协议和仿真脚本均没有改动。原`.git`、旧编译目录、
缓存和运行日志排除；新容器生成自己的Catkin产物。目标独立配置只调整
`runtime.mode=gmt`和`gmt.container=bumi_genmo_gmt_deploy`。

只携带Stage1 s595000的原始ONNX、元数据、统计与运动学源资产，以及查看器和三首MP3。
九项匹配SDK头文件包含实际C++依赖闭包，来源及逐文件SHA保存在`sdk-source-manifest.json`。
原主机CUDA13 engine、旧s350000和训练checkpoint没有传输。

| 项目 | SHA256或身份 |
|---|---|
| Stage1 checkpoint来源 | `51b522fbdeb0ec289e5ffc24e2fccca9888df90495f57f96714c1bb43c05f8e2` |
| 原始ONNX | `fa197f412f4c649327f264efb9003fe80bff76d0b482a0900c8647963231420e` |
| 目标机FP32 engine | `677f98ff97dfddeea747b2871e853b5004b200054e865004a0e9638e3ec75341` |
| 目标机CUDA12插件 | `9d33a58a0c93c79798e2b9cc370366fae31bb62802dc6f7a02f1abd029f0ec4e` |
| GMT既有策略 | `model_135000_stage2.onnx`，`d2e176657d72d1b0efcb04406abd17145fc45a3a5755b62aae7b866e0a6e3d1b` |

目标engine缓存键为`09c93a2dd67c42eaa3c451f7e725fcfe2bb665065ef20efaec334c2206475d3b`；
保持FP32、关闭TF32/FP16、十一输入/两输出，构建约8.35秒。独立数值报告单独保留；
构建清单中的`inference_validation=not_run`仅表示构建器自身不执行推理，不覆盖独立报告。

## 已执行验收

| 验收 | 结果与范围 | 目标机证据 |
|---|---|---|
| 文件与共享环境 | GMT935项一致；原GENMO修改、包版本与editable保持；依赖检查通过 | `environment-files-verification-final.json`、`pip-check-final.log` |
| Stage1数值 | 历史/前缀掩码、无效NaN/Inf隔离、CUDA Graph、20步DDIM及三首音乐双窗口通过 | `stage1-validation.json`、`stage1-validation.log` |
| GMT原生 | 实际加载隔离ORT1.19.0 CUDA库；100组相同合成输入CPU/CUDA动作最大差4.29153e-6 | `gmt-native-verification.json`、CPU/CUDA探针及profile |
| GMT编译 | Release Catkin九个包通过，包含`RL_CONTROLLERS_HAS_HIREDIS` | `gmt-catkin-build.log` |
| buffered | 新春锣鼓前10秒，300源帧整段一次提交、收到ACK、结束返回STAND | `buffered-runtime-verification.json` |
| 仿真策略步时钟 | 连续600策略步缓存帧逐步+1；暂停Gazebo2秒，游标95保持不动 | `buffered-policy-step-verification.json`、`buffered-diagnostics.json` |
| 在线 | Bouncy前10秒，120/108/72帧分三块提交、ACK和自然返回STAND通过 | `online-runtime-verification.json` |
| 停止与心跳 | Robot Choreography片段的`stand`通过；仅暂停本次Console中断心跳，桥按原阈值自动返回STAND，再恢复Console | `online-stop-verification.json`、`online-heartbeat-verification.json` |
| 图形 | 目标桌面`:1`上的Gazebo和MuJoCo窗口已映射且实际渲染；预览进程alive、无错误 | `gui-windows.txt`、`gui-online-windows.txt`、Console日志和窗口截图 |

20步DDIM qpos30最大误差为1.71065e-5，接触概率最大误差5.69224e-5，接触标签无差异。
三首音乐数值对齐各取开头167帧，生成120+47帧，两窗口qpos最大误差依次为
3.39746e-6、5.84126e-6、2.86102e-6；FK位置最大误差为7.00355e-7、1.89245e-6、
6.03497e-7米，沿用现有验收阈值。

原生CUDA探针100次计时平均约0.529毫秒、P95约0.713毫秒；这是独立推理开销。
Gazebo实际运行约42至43个策略步/真实秒，buffered仍按每个仿真策略步消费一帧，
不能把缓存播放理解为保证真实控制50Hz。buffered自动音乐关闭；在线音频行为保持原设置。

## 初始化条件与失败记录

首次启动后等待、再进入WALK时，机器人姿态触发原控制器的摔倒保护，尚未提交音乐。
验收在STAND阶段暂停物理，按桥的固定站姿重置关节、根姿态及零速度，随后
恢复物理并沿用原WALK到GMT切换。原控制器、摔倒保护阈值、重力、步长和策略未修改。
该初始化过程及之后的模式分别记录在`buffered-sim-reset.json`、`buffered-gmt-mode.json`
和`online-gmt-mode.json`。最终buffered根高度约0.473米，原重力为[0,0,-9.8]，步长0.0005秒。

首次状态采样启动晚于片段完成，因此未捕获ACK和暂停过程；正式复验在提交前启动监测，
覆盖完整过程并通过。首次诊断把固定idle的网络包序号混入缓存帧；正式统计按装载/退出
缓存的序号重置边界提取601条缓存样本，600次相邻策略步全部逐帧+1。早期观察与原始
诊断另存，未把采样或统计错误报告为控制器故障，也未改变阈值来通过验收。

## 容器与最终状态

原`bumi_sonic_deploy`停止并保留，可写层保存为`bumi-genmo-gmt:base-20261010`，
据此创建`bumi_genmo_gmt_deploy`。新容器保留GPU、host网络、设备权限与X11访问，增加
新GMT与隔离运行库的绑定；启动参数保存在`migration/container-create.json`。
原GMT目录和原GENMO仓库保留。

验收结束后Console、Bridge与预览正常退出，新容器重启清理本次ROS/Gazebo进程；
新容器保持运行等待启动，旧容器保持停止。安装包、正式报告和备份保留在migration；
本次临时下载中转、Numba缓存和截图中间格式清理，正式PNG和日志保留。

## 重建与日常启动

环境、模型已经安装构建完成，日常使用直接按[迁移说明](LIWEI_DEPLOYMENT_MIGRATION.md)
中的三终端命令启动。需要重建时执行：

```bash
cd /home/user/liwei/GENMO-deploy-bumi
BUMI_STAGE1_NVCC=/usr/local/cuda-12.8/bin/nvcc \
BUMI_STAGE1_TRT_INCLUDE="$PWD/sdk/tensorrt-10.13.3.9/include" \
bash scripts/export/build_bumi_stage1_engine.sh
```

图形授权应在目标机桌面终端运行`xhost +si:localuser:root`，使用该会话实际`DISPLAY`。
进入GMT的原操作和音乐输入语法不变，`--preview`打开主机MuJoCo参考窗口。
仿真验收使用了明确的站姿初始化；启动后若已倒下，应先恢复站姿再切到GMT。

## 恢复边界

只有需要回退时，先退出新Console/Bridge并停止新容器；保留当前共享环境到一个新的
migration备份目录，再把上述完整旧`.venv`复制回原路径。部署目录的`.venv`仍指向原路径。
随后启动旧`bumi_sonic_deploy`，按原流程恢复旧任务。旧容器与旧仓库都没有删除。
本轮验收通过，未执行环境回退，也未启动旧任务。
