"""独立入口的配置、真实资产和组件装配。

只按显式路径读取固定SM、音乐、初态、冻结GENMO名义缓存和训练基线s0；不下载、
重命名last模型或猜测生产路径。歌曲与音频内容共同阻止训练/验证泄漏。读取配置与
构建CPU网络不启动Isaac；真实物理由entry中的显式上下文管理。
"""

import json
from pathlib import Path
import random

import numpy as np
import torch
from omegaconf import OmegaConf

from supermimic.bridge.bundle import load_bundle, file_digest
from supermimic.bridge.runtime import SuperMimicRuntime
from .policy import ReferenceAdapter
from .music_reward import AudioTrack, MusicReward
from .rewards import Rewards
from .collector import ControlStep
from .dynamics import DynamicsModel
from .critic import AdapterCritic
from .features import critic_dimension
from .imagination import Imagination
from .nominal import load_nominal


def read_config(path):
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    exact = dict(algorithm="genmo_supermimic_reference_adapter10", physics_backend="isaaclab3_newton_mujoco_warp",
                 reference_steps=10, hidden_dims=[512, 512, 256], output_dim=270, control_hz=50)
    if any(cfg.get(k) != v for k, v in exact.items()) or cfg["music"]["lag_steps"] != 2:
        raise ValueError("Adapter算法/三隐藏层/10帧/50Hz/音乐延迟合同不匹配")
    if cfg["bptt_steps"] < 1 or cfg["training"]["batch_size"] < 1:
        raise ValueError("展开时域及更新批量必须为正")
    return cfg


def require_file(value, name):
    if value is None or not Path(value).expanduser().is_file():
        raise FileNotFoundError(f"缺少显式资产{name}: {value}")
    return Path(value).expanduser().resolve()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_tasks(path, *, preparing=False):
    tasks = json.loads(require_file(path, "tasks").read_text())
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("tasks必须为非空JSON数组")
    songs, hashes, ids = {}, {}, set()
    for task in tasks:
        for key in ("id", "song_id", "split", "seed", "initial_state", "music_features"):
            if key not in task:
                raise ValueError(f"任务缺少{key}")
        if task["id"] in ids or task["split"] not in ("train", "validation"):
            raise ValueError("任务id重复或缺少显式数据划分")
        ids.add(task["id"])
        require_file(task["initial_state"], "initial_state.npy")
        music = require_file(task["music_features"], "music_features")
        for mapping, key in ((songs, task["song_id"]), (hashes, file_digest(music))):
            if mapping.setdefault(key, task["split"]) != task["split"]:
                raise ValueError("同歌或相同音频内容泄漏到训练/验证两侧")
        if not preparing:
            require_file(task.get("nominal_reference"), "nominal_reference")
    return tasks


def raw_task(task, device):
    path = require_file(task["music_features"], "music_features")
    music = torch.from_numpy(np.load(path, allow_pickle=False)) if path.suffix == ".npy" else torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(music, torch.Tensor) or music.ndim != 2 or music.shape[1] != 35:
        raise ValueError("音乐必须是原库EDGE35 Tensor或npy")
    initial = torch.from_numpy(np.load(task["initial_state"], allow_pickle=False)).float().reshape(1, 55).to(device)
    if not torch.isfinite(initial).all() or initial[:, 3:7].norm() < 1e-6:
        raise ValueError("初态state55非有限或四元数无效")
    return music.float().to(device)[None], initial


def task_assets(task, engine, device):
    music, initial = raw_task(task, device)
    audio = AudioTrack.from_features(music, start_tick=task.get("music_start_tick", 0),
                                    sigma_s=engine.step.rewards.music_sigma_s)
    nominal, contract = load_nominal(task["nominal_reference"], engine.step.bundle, device=device,
                                     music_path=task["music_features"], initial_path=task["initial_state"], seed=task["seed"],
                                     music_start_tick=task.get("music_start_tick", 0))
    return initial, nominal, audio, contract


def build_engine(cfg, device="cpu", *, calibrating=False):
    bundle = load_bundle(require_file(cfg["supermimic"]["checkpoint"], "supermimic.checkpoint"), device,
                         robot_asset_dir=cfg["supermimic"]["robot_asset_dir"])
    runtime = SuperMimicRuntime(bundle, device)
    if calibrating:
        # 仅采集零Adapter速度统计；此临时标尺不参与任何参数更新，标定后必须重新加载。
        s0 = torch.ones(21)
    else:
        baseline = torch.load(require_file(cfg["music_baseline"], "music_baseline"), map_location="cpu", weights_only=False)
        if baseline.get("source") != "zero_adapter_real_training_episodes" or tuple(baseline["joint_names"]) != tuple(bundle.spec.joint_names):
            raise ValueError("s0必须由零Adapter真实训练集速度统计得到，不能在线自归一化")
        s0 = baseline["s0"]
    adapter = ReferenceAdapter(bundle.spec).to(device)
    music_cfg = {k: v for k, v in cfg["music"].items() if k not in ("sigma_s", "lag_steps")}
    music = MusicReward(s0, **music_cfg)
    rewards = Rewards(runtime, music, cfg["rewards"] | ({"music": 0.} if calibrating else {}))
    rewards.music_sigma_s = cfg["music"]["sigma_s"]
    step = ControlStep(adapter, bundle, runtime, rewards, observation_noise=cfg["observation_noise"])
    dynamics = DynamicsModel(bundle).to(device)
    critic = AdapterCritic(critic_dimension(adapter)).to(device)
    return Imagination(step, dynamics, critic, bptt_steps=cfg["bptt_steps"], gamma=cfg["training"]["gamma_per_control"])


def behavior_config(cfg):
    # 只绑定实际采样/奖励/优化合同，不比较源码SHA或可重定位的目录。
    return {k: cfg[k] for k in ("algorithm", "physics_backend", "reference_steps", "hidden_dims", "output_dim", "control_hz",
                               "bptt_steps", "observation_noise", "music", "rewards")} | {
        "optimization": {k: cfg["training"][k] for k in ("adapter_lr", "critic_lr", "wm_lr", "wm_fit_window", "gamma_per_control", "lambda_return_per_control", "grad_clip_norm")}}
