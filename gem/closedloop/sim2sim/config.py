"""加载便于替换本机资产的GENMO＋GMT sim2sim配置，并补齐固定运行参数。

用户YAML只需维护模型、配套资产、音乐入口、输出目录和GPU选择；20步确定性
DDIM、数值精度、物理告警阈值、标定预算及显示参数统一在本模块保存默认值。
加载时先复制默认配置，再合并YAML各节，因此调用方调整音频或测试参数不会污染
下一次加载；确需专项验收时仍可在自定义YAML覆盖相应运行项。

GENMO相关相对路径以本仓库为基准，GMT模型/机器人/参考合同以gmt_repo为基准，
均不依赖启动时的工作目录或配置文件存放位置；绝对路径及用户主目录路径照常
支持。Stage1的YAML锚点复用后也分别解析为绝对路径，避免重复填写基础权重。
返回的完整配置交给既有运行时，并原样写入run_report，保持轨迹重放可追溯。
本模块只解析配置，不加载权重、启动仿真或创建输出目录。用户只填写文件路径，
模型内置的配套资产校验及程序自动生成的运行指纹由既有运行时负责。
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml


GENMO_ROOT=Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG=dict(
    schema='genmo.gmt.mujoco_sim2sim.v1',
    paths=dict(genmo_repo=str(GENMO_ROOT)),
    generation_frames=60,
    sampling=dict(generation_frames=60,denoising_steps=20,eta=0.,guidance_scale=2.5,
        # 只复用概率构造器的均值；这些正floor在确定性推理中绝不被采样。
        std_schedule=[.001]*20),
    precision=dict(mode='native_fp32',linear_execution='stable_tf32x3',attention='sdpa_math',
        compile=False,precise_output=True,weight_gradient='direct_joint_gemm'),
    runtime=dict(continue_on_failure=True,device='cuda:0',torch_threads=1,warmup_control_steps=50,
        calibration_warmup=3,calibration_samples=12,latency_guard_seconds=.06,lateness_limit_seconds=.1,
        auto_next=True,audio=True,viewer_fps=25,video_fps=25,video_width=1280,video_height=720),
    termination=dict(root_height_error_m=.2,non_yaw_orientation_error_rad=.6,yaw_error_rad=None,
        end_effector_relative_height_error_m=.15))
_MERGED_SECTIONS=('paths','sampling','precision','runtime','termination')
_GMT_PATHS=('gmt_policy','gmt_checkpoint','gmt_agent','mjcf','urdf','isaac_contract')
_GENMO_PATHS=('stage1_checkpoint','stats','kinematics','dataset_root','output_root','gmt_validation_python')


def _absolute(value,base):
    path=Path(value).expanduser()
    # 保留符号链接入口，尤其不能把venv/bin/python解析成系统Python而丢失虚拟环境。
    return os.path.abspath(path if path.is_absolute() else base/path)


def load_config(path):
    """合并本机资产配置与运行默认值，返回包含绝对路径的独立完整字典。"""
    supplied=yaml.safe_load(Path(path).expanduser().read_text(encoding='utf-8'))
    if not isinstance(supplied,dict):raise ValueError('sim2sim配置必须是YAML映射')
    config=copy.deepcopy(_DEFAULT_CONFIG)
    for key,value in supplied.items():
        if key in _MERGED_SECTIONS:
            if not isinstance(value,dict):raise ValueError(f'sim2sim配置{key}必须是映射')
            config[key].update(copy.deepcopy(value))
        else:config[key]=copy.deepcopy(value)
    if config['schema']!='genmo.gmt.mujoco_sim2sim.v1':raise ValueError('不支持的sim2sim配置')
    paths=config['paths']
    paths['genmo_repo']=_absolute(paths['genmo_repo'],GENMO_ROOT)
    genmo=Path(paths['genmo_repo'])
    paths['gmt_repo']=_absolute(paths['gmt_repo'],genmo)
    gmt=Path(paths['gmt_repo'])
    for key in _GENMO_PATHS:paths[key]=_absolute(paths[key],genmo)
    for key in _GMT_PATHS:paths[key]=_absolute(paths[key],gmt)
    for model in config['models']:model['checkpoint']=_absolute(model['checkpoint'],genmo)
    config['playlist']=_absolute(config['playlist'],genmo)
    return config
