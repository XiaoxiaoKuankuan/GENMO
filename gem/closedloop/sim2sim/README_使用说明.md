# GENMO＋GMT：MuJoCo实时闭环

这里实际执行 **GENMO在线生成 → 冻结GMT 140500控制 → MuJoCo动力学 → 实际状态回到GENMO**。
机器人除episode重置外由动力学推进。它与“把Isaac已执行轨迹放进MuJoCo画视频”的旧回放工具不同。
本入口只推理，不加载Critic/优化器，不启动Isaac、不修改服务器训练。

## 启动

本机环境为 `/home/weili/GENMO/.venv/bin/python`，配置固定在
`configs/closedloop/sim2sim_mujoco.yaml`。已有模型和素材不进入Git，移动机器后需要配置对应资产的绝对路径。

```bash
cd /home/weili/bumi-closedloop-worktrees/GENMO

# 现有四来源清单，打开窗口并播放声音
PYTHONDONTWRITEBYTECODE=1 /home/weili/GENMO/.venv/bin/python -B \
  scripts/demo/run_genmo_gmt_sim2sim.py

# 任意整首WAV或MP3；特征在开始物理执行前提取并缓存
PYTHONDONTWRITEBYTECODE=1 /home/weili/GENMO/.venv/bin/python -B \
  scripts/demo/run_genmo_gmt_sim2sim.py --audio /绝对路径/歌曲.mp3 --record

# 替换成一个明确的Stage2 checkpoint，不能传latest指针或目录
PYTHONDONTWRITEBYTECODE=1 /home/weili/GENMO/.venv/bin/python -B \
  scripts/demo/run_genmo_gmt_sim2sim.py \
  --actor-checkpoint outputs/sim2sim/models/actor_iteration147.pt \
  --audio /绝对路径/歌曲.wav --seed 42 --record

# 无窗口自动验证；--mute明确关闭声音设备，音乐条件与音乐时钟继续工作
PYTHONDONTWRITEBYTECODE=1 /home/weili/GENMO/.venv/bin/python -B \
  scripts/demo/run_genmo_gmt_sim2sim.py --headless --mute \
  --audio /绝对路径/歌曲.wav --max-seconds 10
```

`--playlist`接受现有`selection.json`、逐行manifest、JSON音频条目列表或每行一个音频路径的文本清单。
selection的`source/row`从配置`dataset_root`解析；manifest从所属数据集目录解析；普通音频清单相对清单位置解析。
同曲动作片段的音乐分组身份保留，不称作独立音乐。原特征若只覆盖短片段，终端明确打印提示，整首重新提取EDGE35，
报告同时保留旧特征身份和新缓存身份；配对活动度只在有真实配对覆盖的区间有效。

## 操作

| 按键 | 行为 |
| --- | --- |
| 空格 | 控制边界暂停/恢复物理、音乐和逻辑截止时钟 |
| R | 同一曲、同一随机种子，从站立初态重播 |
| N / P | 下一首/上一首，新episode |
| 1 / 2 / 3 | Stage1 s595000 / 固定旧基线147轮 / 固定旧基线246轮，重新开始当前歌曲 |
| G | 显示/隐藏已提交参考的半透明机器人 |
| 鼠标拖动 / 滚轮 | 转动视角 / 缩放 |
| Esc或关闭窗口 | 安全结束，保存报告，随后编码已请求的视频 |

物理失败后停在结果界面，不自动循环重试；R/N/P/模型键仍可使用。自然音乐结束按配置切到下一首。
无窗口模式在失败时结束，避免无人值守挂起。`--max-seconds`是**当前sim2sim验收的实际仿真总秒数**，含站立预热，
达到后记行政结束；它与训练的停止策略无关。SIGINT/SIGTERM与Esc都走安全收尾。

## 控制和数值合同

- GENMO实际输入60帧（2秒，30Hz），每0.5秒重规划，完整20步随机扩散。120帧需显式配置并匹配checkpoint身份。
- GMT CPU ONNX，50Hz；MuJoCo200Hz，每控制步真实执行四个子步。模型来源必须为140500，PD/默认角/动作缩放/力矩上限均取自ONNX元数据。
- GMT历史为69×10，按观测项展开；GENMO历史为50×48，未知历史是无效mask。48维关节顺序与Isaac训练后端一致。
- 保留参考reserve/prepare/commit、未来10个控制步、插值和差分右点保护。缺参考直接结束，不重复末帧。
- 一张本机4090、一个GENMO推理线程；物理在推理期间继续。窗口独立MjData，25fps并关闭实时阴影，物理不降频。
- 每个模型首次使用时在本机预热，并实测GENMO＋参考准备与GMT/物理/窗口并发负载。预算为实测最大值×1.2＋60ms，向20ms网格上取整；
  不能超过0.5秒减一个控制周期。这个部署合同不复用服务器的0.30秒。旧/迟到结果不能提交到新episode。
- Stage2只导入Actor权重和绑定的噪声、eta、CFG、Horizon、数值算子。跨机器Torch版本写入新的推理身份，并重新标定；
  不声称与原训练机器逐位等价，不复用旧rollout，也不接受无身份权重作为训练恢复。
- 窗口/控制落后超过100ms或音频游标偏差超过100ms时结束当前episode，不跳过物理步追赶。

本机已补齐用户级PortAudio动态库，位置为既有虚拟环境`lib/`；入口需要时只为本进程设置动态库搜索路径。
其他机器需要可用的PortAudio/声音设备、MuJoCo、GLFW、PyTorch、ONNX Runtime、librosa、soundfile、sounddevice和ffmpeg。
`--mute`不会移除音乐条件；缺少声音依赖时不会静默伪装为有声运行。

## 输出和结果边界

全部集中在 `outputs/sim2sim/`：

```text
models/                         固定147/246的Actor权重和导出来源、SHA
music_cache/                    按音频内容＋提取配置命名的EDGE35缓存
<独立运行目录>/
  launch.json                  CLI实际配置
  model_*_identity.json         实际权重、随机核和训练来源
  gmt_native_onnx_parity.json    启动时原checkpoint与CPU ONNX数值对照
  calibration_model*.json       本机并发部署延迟测量和预算
  run_report.json               所有episode、冻结检查和源码SHA
  episode*/
    report.json                完成/失败原因、活动度、跟踪、边界、时间和请求记录
    control_trace.npz           每50Hz真实状态、GMT输入/动作、四子步PD力矩和mask
    physics_with_audio.mp4      --record时生成，实际MuJoCo状态＋原音乐
    video.json                 动力学来源、视频时钟和音频延迟说明
```

视频结束后编码，不占控制线程。视频按实际模拟时间播放，开头有1秒站立预热静音；用户主动暂停的墙钟等待不写进视频。
25fps编码可能与控制时长差不到一帧。视频及报告明确标记`physics_backend=mujoco`。

“活动度”是最近0.5秒实际关节速度RMS；门控为配对动作的诊断，**没有配对时是null/无效mask**。
音乐同步性只是动作速度低点与音乐拍点的非奖励诊断。训练奖励不在这里伪造：MuJoCo显式PD/接触与Isaac传感器
存在差别，当前报告`training_reward=null`，分别报告真实活动度、跟踪误差和关节边界。
负的`joint_limit_margin_rad`表示实际关节越过模型声明边界，不能因MuJoCo软约束允许少量越界就当作正常。

`real_time_pass`只说明当前执行区间的计时通过，物理失败仍是失败；短段、四来源输入通过也不等于所有完整歌曲通过。
三模型同条件大规模评估属于后续工作。本首版验证固定模型加载/切换与真实闭环，不据此宣布Stage2优于Stage1。

## 本机验收结果（2026-10-14）

36项自动回归通过；原播放器提取前后100控制步/400物理步的观测、动作、力矩及实际状态逐位一致。
真实外力会改变实际历史和后续GENMO输出；三模型加载、暂停、同种子重播与切歌在途请求丢弃均已验证。

带窗口和无窗口各有一次63.46秒实时通过记录，均因朝向误差物理终止，不能当作长歌完成。
另一轮窗口复核在55.94秒发生生成截止超期，正确停止；长期实时稳定性仍未证明。
7.941秒WAV已完整自然结束。长段还有28个控制步发生关节边界越界，报告保留这些实际质量问题。

详细证据：[验收报告](../../../outputs/sim2sim/acceptance_20261014/验收报告.md)；
[主视频](../../../outputs/sim2sim/acceptance_20261014/window_65s/episode0001-d86c0480/physics_with_audio.mp4)。
这些是本机交付产物，不随Git上传。重复的早期测试视频和临时目录已清理，失败证据集中保留。

## 回归测试

```bash
PYTHONDONTWRITEBYTECODE=1 /home/weili/GENMO/.venv/bin/python -B -m pytest \
  -p no:cacheprovider -q tests/closedloop/sim2sim

# 显式占用本机GPU：真实外力是否通过实际历史改变下一段GENMO输出
SIM2SIM_REAL_TESTS=1 PYTHONDONTWRITEBYTECODE=1 /home/weili/GENMO/.venv/bin/python -B \
  -m pytest -p no:cacheprovider -q tests/closedloop/sim2sim/test_real_feedback.py

cd /home/weili/bumi-closedloop-worktrees/legged_lab_gmt
PYTHONDONTWRITEBYTECODE=1 /home/weili/GENMO/.venv/bin/python -B -m pytest \
  -p no:cacheprovider -q tests/test_mujoco_closedloop_backend.py tests/test_sim2sim_sonic_playlist.py
```
