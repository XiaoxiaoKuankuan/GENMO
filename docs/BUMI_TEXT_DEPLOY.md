# BUMI 文本模型独立运行包

这个包只运行已经导出的BUMI文本模型，不携带训练checkpoint，不连接GMT/ROS/Redis。
按模型自身契约生成30FPS的机器人qpos28（当前crop120为4–120帧，历史full300为60–300帧），可在MuJoCo窗口查看纯运动学动画。

1. 目标环境为Ubuntu22.04、x86_64、Python3.10及可用的NVIDIA驱动。engine要求匹配
   构建GPU与TensorRT10.13.3.9；同为4090也必须通过启动检查。不兼容时在训练仓库对应
   环境重新构建，不能靠重命名文件解决。
2. 将完整T5-3B本地目录放在`models/t5-3b`，或编辑`deployment.ini`中的`t5_model`。
   若源目录是HF快照，复制时使用`cp -aL`解引用软链接。包不会联网下载T5。
3. 在包根目录运行`bash install.sh`。脚本自动准备uv、Python和包内.venv；缺少
   FFmpeg/libGL时安装系统依赖，可能要求sudo。不会安装训练框架或自动修改驱动。
4. 编辑`deployment.ini`选择`backend=tensorrt`或`onnx`、`device=cuda:0`或`cpu`
   （TensorRT必须cuda）。无engine的开发包默认ONNX，仍需检查数值验证状态。
5. 运行`bash run.sh`。无需先activate环境。默认打开MuJoCo窗口；不需要窗口时将
   `[preview] enabled=false`。

控制台输入文本，或`play Walk forward.`。`frames 120`和`steps 50`修改后续生成参数；
`pause/resume/stand`控制本地动作播放，`status`查看状态，`quit`退出。默认CFG2.5、
seed42、DDIM50、120帧，T5最大150token。关闭窗口不关闭控制台，窗口不会执行动力学。

每次结果保存在`outputs/bumi_text/独立ID`，包括motion.npz与metadata.json。NPZ同时保留
qpos_raw、足锁后的qpos、左右脚接触logits、关节顺序与FPS；足锁只修正根XY，不固定
根高度。CLI附加`--render`时另起MuJoCo渲染子进程，视频经完整解码检查后保存。

模型文件、ONNX外部权重、stats、kinematics和XML/mesh都由deployment.json及资源清单
绑定SHA，必须成套替换。只换一个ONNX或engine会被拒绝。更新模型时重新生成整个包，
保留旧包，验收成功后再切换。`software_inventory`是打包源码记录，deployment.ini可编辑。

`validation_status=not_validated`表示尚未附带匹配的数值对照报告。即便passed，也只代表
相应数值测试通过，不代表动作语义、平衡、动力学或实物跟踪通过。真实新电脑安装和
窗口/GPU体验需在该机器验证。

训练、数据、导出与数值对照工具均在`feature/bumi-text-only`训练仓库，详见其中
`docs/BUMI_TEXT_CROP120.md`。压缩/搬运本包时排除`.venv`和`outputs`，到新电脑重新
执行安装器；不要复制依赖原绝对路径的虚拟环境。
