"""新Stage2入口的资产装配、原配对数据游标与任务读取。

Stage1从原checkpoint配置构造并严格加载，只在内存重定位stats/FK路径；SM从同一固定
文件加载pi/wm。运行前显式提供音乐任务、延迟标定及原配对库，不生成替代数据或假权重。
模块导入不初始化Isaac，真实环境由CLI在launch_simulation中构造；源码版本只记录。
"""

from dataclasses import asdict
from pathlib import Path
import json
import random
import subprocess

import numpy as np
import torch
from omegaconf import OmegaConf

from supermimic.bridge.bundle import load_bundle, file_digest
from gem.closedloop.checkpoint import load_stage1_checkpoint, actor_asset_identity
from gem.closedloop.training import build_stage1_actor, build_stage1_losses, build_stage1_loader, batch_to_device
from .actor import ForecastActor
from .sm_adapter import SMAdapter
from .scheduler import Scheduler
from .critic import PlanAwareCritic
from .imagination import Imagination, PhysicalAnchorImagination
from .rewards import TrackingRewards


def read_config(path):
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if cfg["algorithm"] != "genmo_supermimic_model_based":
        raise ValueError("算法身份不匹配")
    if cfg["physics_backend"] != "isaaclab3_newton_mujoco_warp":
        raise ValueError("必须使用SM匹配的物理后端")
    expected = {"motion_frames": 60, "motion_fps": 30, "ddim_steps": 20, "eta": 0.0}
    if any(cfg["model"].get(k) != v for k, v in expected.items()):
        raise ValueError("首版固定60@30、全20步DDIM eta0")
    if cfg["model"]["cfg_scale"] != 2.5:
        raise ValueError("本次正式合同固定CFG=2.5")
    return cfg


def require_file(value, name):
    if value is None or not Path(value).expanduser().is_file():
        raise FileNotFoundError(f"缺少实际资产 {name}: {value}；不创建假权重或重建数据")
    return Path(value).expanduser().resolve()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_components(cfg, device):
    path = require_file(cfg["stage1"]["checkpoint"], "stage1.checkpoint")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    original = OmegaConf.create(checkpoint["config"])
    original.endecoder.stats_path = str(require_file(cfg["stage1"]["stats"], "stage1.stats"))
    original.endecoder.kinematics_path = str(require_file(cfg["stage1"]["kinematics"], "stage1.kinematics"))
    # 评估/桥接不需要加载完整训练库；网络形状与原条件历史直接来自checkpoint。
    data_contract = OmegaConf.create({"datasets": {}, "sample_contract": {"history_steps": 50},
                                     "qpos30_stats": {"path": original.endecoder.stats_path},
                                     "dataset_defaults": {"kinematics_path": original.endecoder.kinematics_path}})
    base = build_stage1_actor(original, data_contract)
    load_stage1_checkpoint(base, checkpoint)
    base.eval()
    actor = ForecastActor(base, activation_checkpointing=cfg["model"]["activation_checkpointing"],
                          guidance_scale=cfg["model"]["cfg_scale"], batch_cfg=cfg["model"].get("batch_cfg", True)).to(device)
    bundle = load_bundle(require_file(cfg["supermimic"]["checkpoint"], "supermimic.checkpoint"), device,
                         robot_asset_dir=cfg["supermimic"]["robot_asset_dir"])
    adapter = SMAdapter(bundle, base.endecoder.kinematics.joint_order, device)
    if cfg["model"].get("compile_networks", False):
        from .compilation import compile_networks
        compile_networks(actor, adapter)
    scheduler = Scheduler(**cfg["clock"])
    rewards = TrackingRewards(adapter.runtime, cfg["rewards"])
    critic = PlanAwareCritic(scheduler, **cfg["critic"]["network"]).to(device)
    imagination = Imagination(actor, adapter, scheduler, rewards, decisions=cfg["imagination"]["decision_intervals"],
                              gamma=cfg["critic"]["gamma_per_control"], forecast_enabled=cfg["model"]["forecast_enabled"],
                              max_model_steps=cfg["imagination"]["validated_model_steps"])
    if cfg["imagination"].get("mode") == "physical_anchor_short_bptt":
        imagination = PhysicalAnchorImagination(actor, adapter, scheduler, rewards,
            horizon_steps=cfg["imagination"]["horizon_steps"], gamma=cfg["critic"]["gamma_per_control"],
            forecast_enabled=cfg["model"]["forecast_enabled"])
    return actor, adapter, scheduler, rewards, critic, imagination, original


def behavior_config(cfg, actor, adapter):
    return {k: cfg[k] for k in ("algorithm", "physics_backend", "model", "clock", "imagination", "critic", "rewards", "training")} | {
        "formal": cfg.get("formal"),
        "physics_determinism": cfg.get("execution", {}).get("physics_determinism", "not_guaranteed"),
        "stage1_asset": {k: v for k, v in actor_asset_identity(actor.stage1).items() if not k.endswith("_path")},
        "robot_contract": adapter.bundle.metadata["robot"],
        "task_sequence": [task_identity(t) for t in read_tasks(cfg["tasks"])],
        "deployment_latency_samples": json.loads(Path(cfg["latency_profile"]).read_text())["latency_ticks"],
        "supervision_sampling": OmegaConf.to_container(OmegaConf.load(cfg["stage1"]["data_config"]), resolve=True)}


def task_identity(task):
    """绑定实际音乐/初态与种子数据；这是输入数据合同，不是源码hash门禁。"""
    return {"id": task["id"], "music_sha256": file_digest(task["music_features"]),
            "initial_state_sha256": file_digest(task["initial_state"]) if task.get("initial_state") else "nominal_robot_spec",
            "seeds": task.get("seeds", [42])}


def provenance(cfg):
    result = {"config": cfg, "validation": "execution_report_only_after_explicit_run"}
    for name, path in (("genmo", Path(__file__).resolve().parents[3]), ("supermimic", Path(cfg["supermimic"]["repo"]))):
        result[name] = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return result


def read_tasks(path):
    path = require_file(path, "tasks")
    tasks = json.loads(path.read_text())
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("任务清单必须为非空JSON数组")
    if len({task["id"] for task in tasks}) != len(tasks):
        raise ValueError("任务id必须唯一")
    for task in tasks:
        require_file(task["music_features"], f"task {task['id']} 既有EDGE35特征")
    return tasks


def read_music_features(path):
    path = require_file(path, "music_features")
    if path.suffix in (".pt", ".pth"):
        music = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(music, torch.Tensor):
            raise ValueError("原配对库音乐必须是EDGE35 Tensor，不能猜测字典字段")
        music = music.detach().float()
    elif path.suffix == ".npy":
        music = torch.from_numpy(np.load(path, allow_pickle=False)).float()
    else:
        raise ValueError("音乐只接受原库.pt Tensor或明确的.npy EDGE35，不猜测压缩容器格式")
    if music.ndim != 2 or music.shape[1] != 35 or len(music) < 1 or not torch.isfinite(music).all():
        raise ValueError("音乐资产必须是原30Hz EDGE35有限[N,35]")
    return music.contiguous()


def task_inputs(task, adapter, device):
    music = read_music_features(task["music_features"]).to(device)[None]
    if task.get("initial_state"):
        initial = torch.as_tensor(np.load(require_file(task["initial_state"], "initial_state"), allow_pickle=False),
                                  dtype=torch.float32, device=device).reshape(1, 55)
    else:
        initial = torch.zeros(1, 55, device=device)
        initial[:, 2], initial[:, 3] = adapter.bundle.spec.spawn_height, 1.
        initial[:, 13:34] = initial.new_tensor(adapter.bundle.spec.default_joint_pos)
    return music, initial


class LatencyProfile:
    def __init__(self, path, seed):
        report = json.loads(require_file(path, "新部署延迟profile").read_text())
        self.metadata = report
        if report.get("scope") != "single_robot_forecast_genmo_reference_full_path":
            raise ValueError("延迟必须是新框架完整部署路径，不能使用训练batch墙钟")
        self.samples = report["latency_ticks"]
        if not self.samples or any(type(v) is not int or v < 0 for v in self.samples):
            raise ValueError("延迟profile需要非负整数tick")
        self.rng = random.Random(seed)

    def sample(self, count):
        return [self.rng.choice(self.samples) for _ in range(count)]


class SupervisionStream:
    def __init__(self, data_config, batch_size, seed, device):
        self.device, self.epoch, self.consumed = device, 0, 0
        data = OmegaConf.load(require_file(data_config, "原Stage1配对data_config"))
        self.loader = build_stage1_loader(data, split="train", batch_size=batch_size, seed=seed,
                                          num_workers=0, resumable=True)
        self.iterator = None

    def next(self):
        if self.iterator is None:
            self.loader.sampler.set_epoch(self.epoch, self.consumed)
            self.iterator = iter(self.loader)
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.epoch, self.consumed, self.iterator = self.epoch + 1, 0, None
            return self.next()
        self.consumed += len(batch["target_qpos30"])
        return batch_to_device(batch, self.device)

    def state_dict(self):
        return {"epoch": self.epoch, "consumed": self.consumed,
                "loader_rng": self.loader.generator.get_state()}

    def load_state_dict(self, value):
        self.epoch, self.consumed = value["epoch"], value["consumed"]
        self.loader.generator.set_state(value["loader_rng"].cpu())
        self.iterator = None
