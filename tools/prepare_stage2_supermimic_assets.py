"""在服务器上装配已有Stage1、冻结SM及原配对数据的独立验收配置。

输入为真实编号权重、已有四库manifest和派生机器人目录；只读取原库，保存任务索引、
少量已知参考的数值副本、原监督配置与预先声明的物理验收标准。不会训练模型或重建
音乐动作配对库，不把动作标签用于在线GENMO条件。输出目录独占创建以保留证据。
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-dir", required=True)
    parser.add_argument("--sm-checkpoint", required=True)
    parser.add_argument("--sm-repo", required=True)
    parser.add_argument("--robot-asset-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    stage = Path(args.stage1_dir).resolve()
    checkpoint = stage / "s595000.pt"
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    original = OmegaConf.create(ck["config"])
    data = OmegaConf.load(REPO / original.dataset_config)
    data.server1_example.root = str(Path(args.dataset_root).resolve())
    data.qpos30_stats.path = str(stage / "qpos30_train_stats.json")
    data.qpos30_stats.documented_sha256 = digest(data.qpos30_stats.path)
    data.dataset_defaults.kinematics_path = str(stage / "bumi_kinematics_robot_retargeter_fe934_v1.json")
    data.sample_contract = OmegaConf.merge(data.sample_contract, original.sample_contract)
    # YAML merge复制的子项必须逐项恢复原Stage1采样参数，不能继续使用模板P=0。
    for split in data.datasets.values():
        for dataset in split.values():
            dataset.kinematics_path = data.dataset_defaults.kinematics_path
            for key in ("history_steps", "prefix_min_frames", "prefix_max_frames", "prefix_zero_probability", "prefix_random_seed"):
                dataset[key] = data.sample_contract[key]
    supervision = output / "supervision_original.yaml"
    OmegaConf.save(OmegaConf.create(OmegaConf.to_container(data, resolve=True)), supervision)
    cfg = OmegaConf.load(REPO / "configs/closedloop/stage2_supermimic_worldmodel.yaml")
    cfg.stage1 = {"checkpoint": str(checkpoint), "stats": str(stage / "qpos30_train_stats.json"),
                  "kinematics": data.dataset_defaults.kinematics_path, "data_config": str(supervision)}
    cfg.supermimic = {"repo": str(Path(args.sm_repo).resolve()), "checkpoint": str(Path(args.sm_checkpoint).resolve()),
                       "robot_asset_dir": str(Path(args.robot_asset_dir).resolve())}
    cfg.clock.bootstrap_latency_ticks = 180  # 首次测量的显式预算，尚不是标定结论。
    train, validation, evaluation = [], [], []
    root = Path(args.dataset_root).resolve()
    rng = random.Random(20261024)
    for source in ("AIST++", "AIOZ-GDANCE", "FineDance", "Mine"):
        for split in ("train", "val", "test"):
            manifest = root / source / "manifests" / f"{split}.jsonl"
            rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
            rng.shuffle(rows)
            tasks = []
            for row in rows:
                music = (root / source / row["music_feature_path"]).resolve()
                tasks.append({"id": row["sample_id"], "music_features": str(music), "seeds": [42],
                              "source": source, "split": split, "generator_unseen_music": split != "train",
                              "generator_unseen_motion": split != "train", "source_manifest": str(manifest),
                              "motion_path": str((root / source / row["motion_path"]).resolve()),
                              "distribution_id": "bumi_90505_30hz_edge35_nominal_flat"})
            if split == "train":
                train.extend(tasks)
            # 预先固定每来源两个任务；P3使用val，正式评估另用test，不按表现挑选。
            selected = tasks[:2]
            if split == "val":
                validation.extend(selected)
            evaluation.extend(selected)
    # 已知参考仅供WM预测审计，在线条件与监督数据完全独立。
    for index, task in enumerate(validation):
        payload = torch.load(task["motion_path"], map_location="cpu", weights_only=False)
        qpos = payload["qpos"]
        qpos = qpos.detach().cpu().numpy() if isinstance(qpos, torch.Tensor) else np.asarray(qpos)
        if qpos.ndim != 2 or qpos.shape[1] != 28:
            raise ValueError(f"真实动作不是qpos28: {task['id']}")
        path = output / f"known_reference_{index:02d}.npy"
        np.save(path, qpos)
        task["known_reference_world_qpos28"] = str(path)
    for name, tasks in (("train", train), ("validation", validation), ("evaluation", evaluation)):
        (output / f"tasks_{name}.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=2))
    cfg.tasks = str(output / "tasks_validation.json")
    OmegaConf.save(cfg, output / "acceptance.yaml")
    criteria = {"declared_before_measurement": True,
                "basis": "WM误差预算低于既有BUMI关节0.15rad/身体5cm跟踪精度判据；阈值不随结果放宽",
                "horizons": [5, 10, 15, 20, 32], "minimum_directions": 3,
                "minimum_direction_fraction": 2/3, "minimum_optimized_delta": 0., "reset_replay_max_abs": 1e-4,
                "state_rmse_limits": {"root_xy_m": .05, "root_height_m": .025, "root_orientation_rad": .10,
                                      "root_linear_velocity_m_s": .4, "root_angular_velocity_rad_s": .8,
                                      "joint_position_rad": .075, "joint_velocity_rad_s": 1.5,
                                      "foot_position_m": .03, "foot_height_m": .02}}
    (output / "wm_criteria.json").write_text(json.dumps(criteria, ensure_ascii=False, indent=2))
    (output / "asset_manifest.json").write_text(json.dumps({"stage1": str(checkpoint), "stage1_sha256": digest(checkpoint),
        "sm": cfg.supermimic.checkpoint, "sm_sha256": digest(cfg.supermimic.checkpoint),
        "stage1_global_step": ck["global_step"], "original_loss": OmegaConf.to_container(original.loss, resolve=True),
        "sample_contract": OmegaConf.to_container(data.sample_contract, resolve=True),
        "training_tasks": len(train), "validation_tasks": len(validation), "evaluation_tasks": len(evaluation),
        "rebuilt_paired_library": False}, ensure_ascii=False, indent=2))
    print(str(output / "acceptance.yaml"))


if __name__ == "__main__":
    main()
