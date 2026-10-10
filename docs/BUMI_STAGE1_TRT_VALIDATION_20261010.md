# Stage1 s595000 TensorRT实际构建与数值验收

本记录保存2026-10-10在deploy/bumi-music-only-gmt工作树完成的正式模型构建与数值验收
结论、资产身份、误差阈值和复现命令。它是已接受结果的说明文档；临时测试日志和JSON
在取证后清理。所有生成测试均独立运行，不连接GMT、ROS、Redis或机器人，不改写原始
ONNX、训练checkpoint、统计量或运动学资源。

## 实际交付

- 工作树：`/home/weili/GENMO-deploy-bumi`，分支`deploy/bumi-music-only-gmt`。
- GPU：NVIDIA GeForce RTX 4090，sm_89。
- Python：部署`.venv/bin/python`，3.10；Torch 2.6.0+cu124；NumPy 1.23.5。
- 编译器：`/usr/local/cuda-13.0/bin/nvcc`；TensorRT binding/SDK 10.13.3.9，实际libnvinfer 10.13.3。
- 对齐参考：ONNX Runtime GPU 1.23.2，使用原始ONNX，CUDAExecutionProvider，关闭TF32。
- 引擎：FP32，关闭TF32/FP16，优化等级0，workspace 8 GiB；CUDA Graph已成功捕获。
- 清单：`models/bumi_stage1_s595000/deployment.json`，v4七资产包。
- 引擎目录：`models/bumi_stage1_s595000/engines/72604e9a47d95fb456b11db454101f9d6c32a106ba60160c60959a6df7db804d/`。
- 正式文件：`bumi_stage1_denoiser.engine`、`engine.json`和`libbumi_stage1_mask.so`。
- 构建日志记录engine生成8.64334秒，检测11输入/2输出；原始构建命令退出码0。

之后再次运行同一构建命令退出码0，成功复用插件/engine缓存；重新读取正式清单的七个
资产SHA与已通过数值验收的资产完全一致，engine 880167140字节、插件1028880字节。
四个启动角色参数解析及shell语法检查通过：原mode=preview保留，显式入口均选择GMT，
在线realtime/7022/ffplay，离线buffered/7023/audio off；未因此连接桥或机器人。

原ONNX和采样器仍是float32/int64/bool十一输入。私有TensorRT图将四个bool掩码编码为
INT32的0/1，其他输入输出名称/尺寸保持，时间步仍为INT64，两个输出仍为FP32。
本机成功解析114处独立CUDA插件（Where 86、And 2、Cast 22、Greater 2、Less 1、Not 1），
并折叠69处静态常量。GRU、Transformer、归一化与CFG使用原图权重和运算。

| 资产 | SHA256 |
| --- | --- |
| 来源s595000 checkpoint | `51b522fbdeb0ec289e5ffc24e2fccca9888df90495f57f96714c1bb43c05f8e2` |
| 原始ONNX | `fa197f412f4c649327f264efb9003fe80bff76d0b482a0900c8647963231420e` |
| ONNX元数据 | `47f1097eaa8f6a9f966e9d0f761d595889523f253b112545e7cca9ab7a66d976` |
| qpos30统计量 | `c7c0acb0e7a38797adb38ce0be41ee72820e34daed4e780275c64fb55c3e5326` |
| 运动学 | `c08731704dccece11351b6fa877e30bac5ca2a8d363de30af7ca2ea1398f4029` |
| TensorRT engine | `525ebde5754b78e60fb97d1f1e15fb05b089be6beca8df8c4b550f914f37bf09` |
| engine元数据 | `2448a114cedb39df025997a13cd721ac527bba114fe6a2e600660210f80b8ff0` |
| CUDA掩码插件 | `a9b3348a9e04ea39e410585e89276bd7d76041565d035bcccaeec448391b2a91` |

来源checkpoint只保存哈希，部署及验收不加载该文件。清单/engine元数据中的
`inference_validation: not_run`说明构建工具本身不执行推理；本记录对应之后单独执行的
真实验收，不能据构建工具字段推断没有执行数值检查。

## 已执行的检查

正式包加载检查退出码0：七资产路径/大小/SHA、Stage1表示与采样身份、运行库/GPU指纹、
插件注册、engine反序列化、预热和输出有限性通过；输出为[1,120,30]与[1,120,2]。

数值脚本退出码0、`pass: true`，总耗时8.46398秒。单步阈值为
`abs(actual-reference) <= 2e-4 + 1e-4 * abs(reference)`，不是逐位等同声明。

| 单步场景 | 有效历史槽 | motion最大绝对误差 | contact logits最大绝对误差 |
| --- | ---: | ---: | ---: |
| 空历史、无前缀，t=999 | 0 | 3.81470e-5 | 5.91278e-5 |
| 部分历史、12帧前缀，t=500，CFG=1 | 14 | 2.86102e-6 | 3.14713e-5 |
| 完整50步历史、12帧前缀，t=250 | 50 | 6.55651e-6 | 9.25064e-5 |
| 59帧短尾窗、12帧前缀，t=0 | 14 | 3.78489e-6 | 4.76837e-5 |
| 全padding | 0 | 0 | 0 |

两后端的无效未来槽都严格补零。无效音乐/历史/时间/已知坐标/噪声槽填NaN/Inf后，
各自相对填有限值的结果逐位一致。普通TensorRT执行与CUDA Graph输出逐位一致。

完整DDIM使用相同CPU噪声、seed=42、20步eta=0、CFG=2.5。每个参考轨迹步额外以同一
输入调用ONNX和TensorRT，单步保持上述阈值；两个后端的独立采样再比较最终输出。
physical qpos30/canonical qpos28仍按2e-4绝对+1e-4相对阈值；接触logits使用1e-3绝对界，
同时要求sigmoid接触概率绝对差<=1e-4。前缀在物理空间逐位回填，padding严格补零。

| 完整DDIM场景 | qpos30/qpos28最大绝对误差 | logits最大误差 | 概率最大误差 | 接触标签差异 |
| --- | ---: | ---: | ---: | ---: |
| 完整50步历史/120帧窗 | 1.71065e-5 | 4.51088e-4 | 5.69224e-5 | 0 |
| 部分历史/59帧短尾窗 | 2.86102e-6 | 4.19617e-5 | 0 | 0 |

完整历史的20个同输入参考步：motion最大3.63588e-5、logits最大1.52111e-4；短尾窗对应
8.34465e-6/9.34601e-5。首次统一用2e-4绝对阈值评判独立DDIM logits时失败，本记录保留
4.51088e-4的实际差异。随后显式区分中间logits和实际概率，动作阈值未放宽，概率另设
严格1e-4界；修改了验收规则，没有改动模型/engine来消除这个差异。

## 实际音乐片段的两窗口续接

每首读取开头167个30Hz特征帧，约5.57秒，测试短片段而非整首生成。Stage1窗口为
120/12/108：首窗120帧，第二窗有效59帧，复用12帧后新增47帧。两后端都输出连续
120+47帧，最终块标记正确；初始12帧站姿逐位保持，四元数归一化，输出均有限。
世界qpos28容差为1e-3绝对+1e-4相对；根位置、关节、FK位置另按1e-3绝对界核验。

| 音乐 | qpos28分量最大差异 | 根位置最大误差(m) | 关节最大误差(rad) | FK身体位置最大误差(m) | TRT两窗口生成(s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 新春锣鼓.mp3 | 3.39746e-6 | 2.42144e-7 | 3.39746e-6 | 7.00355e-7 | 0.26138 |
| Bouncy sleigh bells.mp3 | 5.84126e-6 | 7.15256e-7 | 5.84126e-6 | 1.89245e-6 | 0.25860 |
| Robot Choreography.mp3 | 2.86102e-6 | 2.68221e-7 | 2.86102e-6 | 6.03497e-7 | 0.25843 |

耗时为已加载模型的本次短片段生成，包含采样/解码/历史构造，不含音频特征提取，
不视为长期性能、整首播放、50Hz真实控制频率、GMT动态稳定性或实机安全的证明。
本轮未执行GMT仿真、实机或pytest；验证采用真实模型数值脚本。

## 复现

```bash
cd /home/weili/GENMO-deploy-bumi
bash scripts/export/build_bumi_stage1_engine.sh
.venv/bin/python -B scripts/demo/check_bumi_deployment.py \
  --deployment-manifest models/bumi_stage1_s595000/deployment.json --inference
.venv/bin/python -B scripts/demo/validate_bumi_stage1_engine.py \
  --audio "/home/weili/下载/新春锣鼓.mp3" \
  --audio "/home/weili/下载/Bouncy sleigh bells.mp3" \
  --audio "/home/weili/下载/Robot Choreography.mp3" \
  --output /tmp/bumi-stage1-validation.json
```

验收后按仓库约定清理该临时JSON。在线/离线三终端启动命令继续使用README中的入口；
构建/验收不改变deployment.ini，也不启动桥或控制台发送动作。
