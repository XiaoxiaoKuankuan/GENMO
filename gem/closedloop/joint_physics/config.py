"""联合训练的静态配置和资产装配，不读取旧GENMO checkpoint。

配置继承仅用于本方法的A/B对照；Stage1构造器只消费架构、原stats及FK配置。数据加载
显式覆盖motion_frames=60，禁止依赖旧YAML的120帧默认值。SM/WM按需从同一固定文件
只读加载，且不注册为GENMO子模块，不创建其优化器。源码版本用于追溯，不作为门禁。
"""

from pathlib import Path
import random
import sys

import numpy as np
from omegaconf import OmegaConf
import torch

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "genmo.joint_diffusion_physics.v1"


def read_config(path, overrides=()):
    path = Path(path).expanduser().resolve()
    value = OmegaConf.load(path)
    if value.get("base_config"):
        parent = read_config(path.parent / value.pop("base_config"))
        value = OmegaConf.merge(parent, value)
    value = OmegaConf.merge(value, OmegaConf.from_dotlist(list(overrides)))
    cfg = OmegaConf.to_container(value, resolve=True)
    if cfg["schema"] != SCHEMA or cfg["initialization"]["mode"] != "random":
        raise ValueError("仅支持本方法随机初始化，恢复走独立完整checkpoint")
    if cfg["initialization"].get("pretrained_genmo") is not None:
        raise ValueError("从零训练不允许提供GENMO初始化权重")
    expected = {"motion_frames": 60, "motion_fps": 30, "ddim_steps": 20, "eta": 0., "cfg_scale": 2.5}
    if any(cfg["sampling"].get(k) != v for k, v in expected.items()):
        raise ValueError("固定60帧30Hz、20步DDIM、eta0、CFG2.5")
    if cfg["physics"]["prediction_mode"] != "denoised_x0" or cfg["physics"]["horizon"] not in (8, 16, 32, 64):
        raise ValueError("仅同一样本denoised_x0，H8/H16/H32/H64需分别真实验收")
    if cfg["execution_loss"].get("soft_joint_limit_enabled", False) or cfg["physics"]["lambda_safety"] != 0:
        raise ValueError("新执行目标软关节限位已关闭，lambda_safety必须为0；原监督项保留")
    if cfg["physics"]["backend"] != "isaaclab3_newton_mujoco_warp" or cfg["physics"]["determinism"] not in ("run_to_run", "gpu_to_gpu"):
        raise ValueError("必须匹配Newton/MuJoCo-Warp及其实际确定性枚举")
    if cfg["stage1"]["model"]["diffusion_steps"] != 1000:
        raise ValueError("完整监督必须保持1000步日程")
    if cfg["execution_loss"]["action_rate_weight"] != 0:
        raise ValueError("首版Action Rate权重必须为0")
    if cfg["training"]["precision"] != "fp32":
        raise ValueError("本版先完成FP32导数验收，尚未开放混合精度")
    return cfg


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def data_config(cfg):
    value = OmegaConf.load(ROOT / cfg["data"]["config"])
    if cfg["data"].get("root"):
        value.server1_example.root = cfg["data"]["root"]
    value.qpos30_stats.path = cfg["stage1"]["endecoder"]["stats_path"]
    value.sample_contract = OmegaConf.merge(value.sample_contract, cfg["data"]["sample_contract"])
    value.sample_contract.future_frames = 60
    for entries in value.datasets.values():
        for item in entries.values():
            item.motion_frames = 60
            item.kinematics_path = str(ROOT / cfg["stage1"]["endecoder"]["kinematics_path"])
    return value


def build_actor(cfg, device):
    from gem.closedloop.training import build_stage1_actor, build_stage1_losses
    from .actor import JointPhysicsActor
    stage = OmegaConf.create(cfg["stage1"])
    stage.endecoder.kinematics_path = str(ROOT / stage.endecoder.kinematics_path)
    # 这里故意不调用initialize_stage1_weights/warm_start/load_stage1_checkpoint。
    base = build_stage1_actor(stage, data_config(cfg))
    actor = JointPhysicsActor(base).to(device)
    losses = build_stage1_losses(base, stage).to(device)
    return actor, losses


def build_adapter(cfg, device):
    sys.path.insert(0, str(Path(cfg["supermimic"]["repo"]).resolve()))
    from supermimic.bridge.bundle import load_bundle
    from gem.closedloop.mb_control.sm_adapter import SMAdapter
    from gem.robots.bumi.kinematics import BumiKinematics
    bundle = load_bundle(cfg["supermimic"]["checkpoint"], device,
                         robot_asset_dir=cfg["supermimic"]["robot_asset_dir"])
    kin = BumiKinematics(ROOT / cfg["stage1"]["endecoder"]["kinematics_path"])
    return SMAdapter(bundle, kin.joint_order, device)
