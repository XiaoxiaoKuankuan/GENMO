# BUMI music-only GENMO＋GMT 部署

本分支只运行原 GENMO 仓库导出的 BUMI 模型。模型资产位于 `models/bumi_v5_s350000/`，不包含训练 checkpoint、模型导出或训练代码。

首次运行 `bash install.sh`。先按控制器自己的方式启动 GMT，再分别执行
`bash run.sh bridge` 和 `bash run.sh genmo`。模型、GPU、端口与容器名用编辑器修改
根目录 [deployment.ini](deployment.ini)，无需在终端临时配置。

- [环境安装、模型检查和三个终端启动](docs/BUMI_MUSIC_DEPLOYMENT.md)
- [GMT 中直接用于 GENMO 接入的改动](docs/BUMI_GMT_GENMO_INTERFACE.md)
- [适配其他GMT、SONIC和通用控制器](docs/GENMO_CONTROLLER_ADAPTATION.md)
- [依赖闭包清单](DEPLOYMENT_FILES.json)
- [实现与实际验收记录](记录文本.md)

默认 DDIM20、CFG2.5、seed42，生成30 Hz、GMT参考50 Hz。迁移时复制代码和完整模型目录，重新创建环境并运行模型检查；目标GPU或TensorRT环境不兼容时在原仓库重新构建engine。

GMT由自身配置选择policy；Bridge默认只读其ROS参数，新版模型包不携带GMT权重。
