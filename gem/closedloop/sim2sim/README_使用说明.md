# GENMO＋GMT：MuJoCo实时闭环

这里实际执行 **GENMO在线生成 → 冻结GMT 140500控制 → MuJoCo动力学 → 实际状态回到GENMO**。
机器人除episode重置外由动力学推进。它与“把Isaac已执行轨迹放进MuJoCo画视频”的旧回放工具不同。
本入口只推理，不加载Critic/优化器，不启动Isaac、不修改服务器训练。

## 启动

本机环境为 `/home/weili/GENMO/.venv/bin/python`，配置固定在
`configs/closedloop/sim2sim_mujoco.yaml`。该文件只保留需要替换的模型、资产路径、音乐入口、输出位置和GPU，
每组旁边都注明更换方法。固定采样、精度、显示、标定和告警默认值在`gem/closedloop/sim2sim/config.py`，
启动时自动补齐；运行报告仍保存完整解析配置。已有模型和素材不进入Git。

GENMO相关相对路径以本仓库为基准，GMT资产相对`paths.gmt_repo`，与启动目录无关；绝对路径也可使用。
GMT仓库搬家只改`gmt_repo`，其余GMT相对路径自动跟随。更换Stage1改`stage1_checkpoint`并检查配套
`stats/kinematics`，模式1通过YAML锚点自动复用该权重；更换Stage2只改对应`models`条目的`checkpoint`路径，
`name`仅用于显示。默认只启用Stage1 s595000和服务器1 P1第500轮；旧147/246路径
仅在YAML中注释保留，需要时替换第二项。无需手动计算或填写文件指纹，程序自动记录实际文件身份；
Stage2绑定的基础资产仍须匹配，当前GMT后端仍限定140500。
更换默认音乐可直接改`playlist`为音频目录或清单，命令行音乐参数优先。专项验收仍可在自定义YAML中
覆盖运行项，例如`runtime.audio: false`；普通启动无需填写整套固定参数。

```bash
cd /home/weili/bumi-closedloop-worktrees/GENMO

# 现有四来源清单，打开窗口并播放声音
PYTHONDONTWRITEBYTECODE=1 /home/weili/GENMO/.venv/bin/python -B \
  scripts/demo/run_genmo_gmt_sim2sim.py

# 任意整首WAV或MP3；特征在开始物理执行前提取并缓存
PYTHONDONTWRITEBYTECODE=1 /home/weili/GENMO/.venv/bin/python -B \
  scripts/demo/run_genmo_gmt_sim2sim.py --audio /绝对路径/歌曲.mp3 --record

# 整个音频文件夹；递归读取WAV/MP3，按相对路径排序
PYTHONDONTWRITEBYTECODE=1 /home/weili/GENMO/.venv/bin/python -B \
  scripts/demo/run_genmo_gmt_sim2sim.py --audio-dir "/home/weili/下载/music/music" --record

# 替换成一个明确的Stage2 checkpoint，不能传latest指针或目录
PYTHONDONTWRITEBYTECODE=1 /home/weili/GENMO/.venv/bin/python -B \
  scripts/demo/run_genmo_gmt_sim2sim.py \
  --actor-checkpoint outputs/sim2sim/models/actor_iteration500.pt \
  --audio /绝对路径/歌曲.wav --seed 42 --record

# 无窗口自动验证；--mute明确关闭声音设备，音乐条件与音乐时钟继续工作
PYTHONDONTWRITEBYTECODE=1 /home/weili/GENMO/.venv/bin/python -B \
  scripts/demo/run_genmo_gmt_sim2sim.py --headless --mute \
  --audio /绝对路径/歌曲.wav --max-seconds 10
```

`--playlist`接受现有`selection.json`、逐行manifest、JSON音频条目列表或每行一个音频路径的文本清单。
selection的`source/row`从配置`dataset_root`解析；manifest从所属数据集目录解析；普通音频清单相对清单位置解析。
`--audio-dir`接受音频目录，`--audio`也兼容目录；二者不能与`--playlist`同时使用。目录递归扫描、不跟随子目录
符号链接，排除隐藏文件/目录和`__MACOSX`等附属文件。同一个真实文件的符号链接别名只读取一次；
目录输入标为外部音频，不根据文件名猜测训练/测试划分或配对动作，也不把内容重复的文件自动认定为独立音乐。

“清单特征仅覆盖9.60s，整曲48.00s”表示旧清单只存了9.6秒配对片段的音乐特征，而音频有48秒。
整曲执行需要覆盖48秒的EDGE35（30Hz、每帧35维）；首次在episode开始前提取，之后按音频内容和提取配置复用缓存。
终端分别显示“首次提取”或“复用缓存”，不会因这条提示每次重复提取，也不是重新训练GENMO。
报告保留旧特征和新缓存的身份；同曲动作片段仍属于原音乐组，配对活动度只在有真实配对覆盖的区间有效。

音乐流程是**每首歌先准备整曲特征，再执行该曲GENMO＋GMT闭环**。目录输入只先扫描播放列表，
进入每首歌时才读取已有完整特征、复用缓存或首次提取该曲；不会先把整个目录全部提取完才开始第一首。
同曲再次播放会复用缓存。音乐特征为30Hz、35维，闭环每0.5秒重规划时按当前音乐时间切出默认60帧
（2秒）的条件窗口，包含对应未来音乐；GENMO动作在线生成，GMT以50Hz控制MuJoCo以200Hz执行。
当前没有麦克风/直播音频的流式特征提取，特征提取耗时也不计入正式闭环的实时延迟。

## 操作

| 按键 | 行为 |
| --- | --- |
| 空格 | 控制边界暂停/恢复物理、音乐和逻辑截止时钟 |
| R | 同一曲、同一随机种子，从站立初态重播 |
| N / P | 下一首/上一首，新episode |
| 1 / 2 | Stage1 s595000 / 当前P1第500轮，重新开始当前歌曲 |
| G | 显示/隐藏半透明蓝色GENMO运动学参考 |
| 鼠标拖动 / 滚轮 | 转动视角 / 缩放 |
| Esc或关闭窗口 | 安全结束，保存报告，随后编码已请求的视频 |

窗口固定为单画面、同一时刻和同一相机的叠加对照，右上角标出两类机器人的含义：

- **半透明蓝色机器人**：GENMO生成并已提交的当前参考姿态。仅在独立MjData中设置参考qpos、做运动学
  正向计算和渲染，不执行GMT控制、PD、接触动力学或mj_step；30Hz源轨迹转换为控制参考时保留既有插值。
- **实体机器人**：GMT根据实际状态和GENMO参考输出动作，由PD和MuJoCo接触、重力等完整动力学推进。
  渲染读取真实qpos；这是仿真中的动力学执行结果。

参考原始姿态不加显示平滑、不平移、不偏移时间。蓝色层只绘制可见机器人几何体，不把辅助site或隐藏形状
染成蓝色。按用户偏好恢复原三维半透明网格：蓝色alpha为22%，保留网格光照和真实深度遮挡，
不再使用平面轮廓合成。G可隐藏参考；这个外观恢复不等于GENMO原始动作抖动已经解决。

默认`runtime.continue_on_failure: true`用于用户要求的连续仿真：姿态/高度/末端误差仅告警，摔倒后仍由真实
动力学继续推进，不自动扶正、不写真实qpos；生成超期/重叠或参考拒绝会记录并丢弃，继续用已有有效参考。
参考不足时显式在源尾部追加静止保持，独立重算并核验保护帧，标记`fallback_hold`；窗口图例同步写明保持参考，
不会冒充新的GENMO输出。后续新计划仍须通过原reserve/prepare/commit保护。控制迟到时重新对齐节拍但保留全部
真实物理步和延迟统计，音频偏差大时显式对齐并计数。非有限数值/持续执行异常记录后重新开始episode，避免停在
失效数据或同tick死循环。用户暂停/退出与验收时限仍生效。严格验收可设为`false`，沿用异常即结束的策略。

完整播放`completed`只表示走完音乐，`physical_success`、`real_time_pass`、`warnings`、`reference_fallback_steps`
分别反映物理表现、计时及降级；持续告警后走完整曲不能写成稳定跳舞。默认`runtime.auto_next: true`：
自然音乐结束后播放下一首，最后一首结束后退出，不自动循环列表；单首输入正常完成后退出。
自动切歌时先从新episode准备下一首特征和站立参考，不承诺无缝连续音频。
无窗口模式在失败时结束，避免无人值守挂起。`--max-seconds`是**当前sim2sim验收的实际仿真总秒数**，含站立预热，
达到后记行政结束；它与训练的停止策略无关。SIGINT/SIGTERM与Esc都走安全收尾。

## 控制和数值合同

- GENMO实际输入60帧（2秒，30Hz），每0.5秒重规划，完整20步扩散采样。120帧需显式配置并匹配checkpoint身份。
- Stage1和Stage2（默认按键1/2及`--actor-checkpoint`）统一使用20步eta=0 DDIM，仅初始噪声随机；
  每步只沿DDIM均值更新，实际附加噪声标准差全0，末步同样不添加探索噪声。保留音乐条件、真实状态反馈、
  已知前缀及checkpoint绑定的冻结前段。条件缓存和合批CFG复用既有执行算子，不做动作平滑或限速。
  同一初始噪声和条件下去噪过程确定；不同初始噪声仍允许动作多样性，`--seed`控制请求噪声序列。
  概率参数构造器内部的正floor不被采样，不是部署实际噪声。当前推理身份明确记为
  `ddim20_eta0_initial_noise_only`、`effective_std_floors=[0,...,0]`，所有模型重新本机标定。
- GMT CPU ONNX，50Hz；MuJoCo200Hz，每控制步真实执行四个子步。模型来源必须为140500，PD/默认角/动作缩放/力矩上限均取自ONNX元数据。
- GMT历史为69×10，按观测项展开；GENMO历史为50×48，未知历史是无效mask。48维关节顺序与Isaac训练后端一致。
- 保留参考reserve/prepare/commit、未来10个控制步、插值和差分右点保护。训练/严格模式缺参考直接结束；
  连续仿真仅通过显式`extend_hold`补足，来源标记与保护校验始终保留。
- 一张本机4090、一个GENMO推理线程；物理在推理期间继续。窗口独立MjData，25fps并关闭实时阴影，物理不降频。
- 每个模型首次使用时在本机预热，并实测GENMO＋参考准备与GMT/物理/窗口并发负载。预算为实测最大值×1.2＋60ms，向20ms网格上取整；
  严格模式不能超过0.5秒减一个控制周期；连续模式完整保存实测预算，超限时采用0.48秒截止并明确标为
  best_effort，迟到仍丢弃。标定prepare使用请求时冻结的参考副本计时，不向已推进的物理时间轴提交过时样本，
  避免标定先被protection_advanced打断。这个部署合同不复用服务器预算。
- Stage2只导入Actor权重，先严格重建并校验checkpoint的原训练核，再把推理eta和实际附加噪声设为0；
  CFG、Horizon、数值算子及冻结前段保留。原身份保存于`source_training_identity/source_training_kernel`，
  实际确定性行为保存于`kernel/sampler`。这是独立部署行为，不冒充原随机策略的训练评估或训练恢复；
  不改训练配置、原checkpoint或服务器运行。跨机器Torch版本亦写入新身份。
- 窗口/控制落后超过100ms或音频游标偏差超过100ms时记录告警；严格模式结束，连续模式继续且不跳过物理步。

本机已补齐用户级PortAudio动态库，位置为既有虚拟环境`lib/`；入口需要时只为本进程设置动态库搜索路径。
其他机器需要可用的PortAudio/声音设备、MuJoCo、GLFW、PyTorch、ONNX Runtime、librosa、soundfile、sounddevice和ffmpeg。
`--mute`不会移除音乐条件；缺少声音依赖时不会静默伪装为有声运行。

`ALSA ... underrun occurred`是声音设备的播放缓冲暂时缺数据，可能造成短暂断音；不是音频文件损坏或机器人失败的
判定。当前音频回调采用低延迟，推理/窗口/物理并发时需要关注调度。报告记录`underflow_callbacks`与游标差，
超过现有100ms音频偏差门槛仍会结束episode。短测无欠载不代表长时间播放已彻底解决，`--mute`仅用于明确关闭声音。

## 输出和结果边界

全部集中在 `outputs/sim2sim/`：

```text
models/                         当前第500轮Actor权重；旧模型文件保留但默认不启用
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

同日显示/目录改进：10项sim2sim回归通过，另1项真实GPU反馈测试默认跳过（首版已单独通过，本次未改反馈链）。
本机真实窗口＋声音＋分屏运行12秒：448＋152控制步、1792＋608物理子步；第一首自然结束并切到第二首，
第二首因显式12秒验收上限结束。两段实时速度比均约1.000，无音频欠载，模型/源码冻结检查通过。
这次只验证目录、显示与自然切歌，不能替代长期实时验收。临时轨迹和视频已清理，诊断摘要与截图保留用于闪烁问题复核。

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
