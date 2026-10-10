# liwei目录的Stage1 GENMO与GMT部署

本说明记录将当前Stage1 s595000部署迁移到另一台RTX 4090电脑的方法。
GENMO部署仓库与GMT当前工作树分别位于`/home/user/liwei/GENMO-deploy-bumi`和
`/home/user/liwei/bumi_GMT_deployment_obs`。主机复用原GENMO Python环境；GMT运行于
独立复制的ROS Noetic容器。GMT源码、策略与轨迹协议保持原样。

## 安装与构建

目标机570驱动使用CUDA12 TensorRT 10.13.3.9，不安装CUDA13或更换驱动。
原环境通过全部已安装锁定依赖检查后先完整备份；部署`.venv`链接原环境，保留原仓库
editable安装。缺失依赖补充后核验实际运行库、代码导入来源及依赖完整性。

```bash
cd /home/user/liwei/GENMO-deploy-bumi
bash install.sh --venv /home/user/robot_genmo_webcam/GENMO/.venv --tensorrt-cuda-major 12
BUMI_STAGE1_NVCC=/usr/local/cuda-12.8/bin/nvcc \
BUMI_STAGE1_TRT_INCLUDE="$PWD/sdk/tensorrt-10.13.3.9/include" \
bash scripts/export/build_bumi_stage1_engine.sh
```

仅携带原始Stage1 ONNX、元数据、统计与运动学四项源资产，目标机产生新的FP32
engine、元数据、CUDA12插件和完整部署清单。原CUDA13 engine不能用于此次部署。
TensorRT C++头文件版本固定10.13.3.9。目标机默认nvcc为11.5，必须明确使用12.8。
独立数值验收使用`scripts/demo/validate_bumi_stage1_engine.py`，保留原有误差阈值。

## 容器与启动

旧`bumi_sonic_deploy`停止后保留；当前环境快照用于创建`bumi_genmo_gmt_deploy`。
新容器将主机GMT目录挂载到`/host/Documents/bumi_GMT_deployment_obs`，将主机
`/home/user/liwei/runtime/bumi-gmt-ort`挂载到`/opt/bumi-gmt-ort`，使隔离ORT/CUDA库
不随容器重建丢失。运行原安装脚本并重新Catkin编译，不携带旧build/devel产物。
部署配置使用`runtime.mode=gmt`及`gmt.container=bumi_genmo_gmt_deploy`。

图形终端先执行`xhost +si:localuser:root`，使用目标机当前DISPLAY进入容器：

```bash
docker exec -it -e DISPLAY="$DISPLAY" -e QT_X11_NO_MITSHM=1 bumi_genmo_gmt_deploy bash
cd /host/Documents/bumi_GMT_deployment_obs
bash ./simulation_buffered.sh gmt_onnx_provider:=cuda gmt_cuda_device_id:=0
```

另两个主机终端均在`/home/user/liwei/GENMO-deploy-bumi`：

```bash
bash scripts/demo/run_bumi_buffered_bridge.sh
```

```bash
bash scripts/demo/run_bumi_buffered_console.sh --preview
```

在线模式分别使用`simulation.sh`、`run_bumi_online_bridge.sh`和
`run_bumi_online_console.sh --preview`。进入GMT模式与音乐输入语法不变。
buffered整段上传、每仿真策略步一帧、自动音乐关闭；在线仍使用实时提交与心跳。
MuJoCo窗口只展示参考姿态，物理跟踪效果由Gazebo查看。

## 记录与恢复

文件指纹、旧容器配置、共享环境备份及实际验收记录放在`/home/user/liwei/migration`。
GMT按实际工作树复制，保留未提交修改，排除Git历史、旧编译产物与运行日志。
恢复时先停止新容器，再在原路径恢复共享环境备份，启动旧容器并按原流程恢复旧任务。
原GENMO仓库、旧GMT目录和旧容器均保留。具体已完成结果以migration中的验收记录为准。
