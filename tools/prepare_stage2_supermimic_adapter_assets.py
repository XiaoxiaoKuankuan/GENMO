"""从服务器真实配对库装配Adapter的四组任务和独立运行配置。

只读取既有Stage1/SM/RobotSpec/四库manifest，按稳定排序选择预先声明数量的歌曲，
以音乐组和实际音频SHA同时隔离train/validation/audit/evaluation。state55用原动作
第一姿态按名称重排、零速度显式物理reset，不把动作标签当GENMO名义缓存或未来状态。
输出独占目录、资产SHA、初态、任务和runtime YAML；真实名义缓存/s0需后续物理准备，
此入口不宣称仿真已通过，不改原数据/权重/示例配置，不初始化CUDA或启动训练。
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage1-dir", "sm-repo", "sm-checkpoint", "robot-asset-dir", "dataset-root", "output"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--train-per-source", type=int, default=2)
    parser.add_argument("--heldout-per-source", type=int, default=1)
    args = parser.parse_args()
    sys.path.insert(0, args.sm_repo)
    from supermimic.bridge.bundle import load_bundle
    from gem.closedloop.reference_adapter.application import read_tasks
    bundle = load_bundle(args.sm_checkpoint, "cpu", robot_asset_dir=args.robot_asset_dir)
    stage, data, output = Path(args.stage1_dir).resolve(), Path(args.dataset_root).resolve(), Path(args.output).resolve()
    cfg = OmegaConf.load(ROOT/"configs/closedloop/stage2_alternating/gpu0.yaml")
    cfg.genmo.checkpoint = str(stage/"s595000.pt")
    cfg.genmo.stats = str(stage/"qpos30_train_stats.json")
    cfg.genmo.kinematics = str(stage/"bumi_kinematics_robot_retargeter_fe934_v1.json")
    cfg.genmo.bootstrap_latency_ticks = 180  # 显式保守候选，不冒充实测GENMO延迟。
    cfg.supermimic = dict(repo=str(Path(args.sm_repo).resolve()), checkpoint=str(Path(args.sm_checkpoint).resolve()),
                         robot_asset_dir=str(Path(args.robot_asset_dir).resolve()))
    checkpoint = torch.load(cfg.genmo.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("global_step") != 595000:
        raise ValueError("必须使用声明的真实Stage1 s595000")
    output.mkdir(parents=True, exist_ok=False)
    tasks, groups, audio_hashes, feature_hashes = [], set(), set(), set()
    for source in ("AIST++", "AIOZ-GDANCE", "FineDance", "Mine"):
        for split, original_split, count in (("train", "train", args.train_per_source),
                ("validation", "val", args.heldout_per_source), ("audit", "test", args.heldout_per_source),
                ("evaluation", "test", args.heldout_per_source)):
            manifest = data/source/"manifests"/f"{original_split}.jsonl"
            rows = sorted((json.loads(line) for line in manifest.read_text().splitlines() if line.strip()),
                          key=lambda r: hashlib.sha256(("adapter_20261026"+r["sample_id"]).encode()).hexdigest())
            selected = 0
            for row in rows:
                group = source+":"+str(row.get("music_group_id", row.get("audio_key", row["sample_id"])))
                if group in groups or not row.get("quality_accepted"):
                    continue
                audio_path = data/source/row["audio_path"]
                music_path, motion_path = data/source/row["music_feature_path"], data/source/row["motion_path"]
                ah, fh = digest(audio_path), digest(music_path)
                if ah in audio_hashes or fh in feature_hashes:
                    continue
                motion = torch.load(motion_path, map_location="cpu", weights_only=False)
                music = torch.load(music_path, map_location="cpu", weights_only=True)
                qpos = torch.as_tensor(motion["qpos"]).float()
                if (qpos.ndim != 2 or qpos.shape[1] != 28 or motion.get("quaternion_convention") != "wxyz"
                        or music.ndim != 2 or music.shape[-1] != 35 or len(music) != len(qpos)):
                    raise ValueError(f"原配对数据形状/时钟/四元数合同不符: {row['sample_id']}")
                names = tuple(motion["joint_names"])
                if len(set(names)) != 21 or set(names) != set(bundle.spec.joint_names):
                    raise ValueError("初态关节名称与RobotSpec不符")
                initial = torch.zeros(55)
                initial[:7] = qpos[0, :7]
                initial[13:34] = qpos[0, 7:][[names.index(n) for n in bundle.spec.joint_names]]
                initial_path = output/f"initial_{len(tasks):04d}.npy"
                np.save(initial_path, initial.numpy())
                tasks.append(dict(id=source+"_"+row["sample_id"], song_id=group, split=split, seed=42,
                    initial_state=str(initial_path), music_features=str(music_path), music_start_tick=0,
                    audio_path=str(audio_path), audio_sha256=ah, music_sha256=fh, source_manifest=str(manifest),
                    source_motion=str(motion_path), initial_semantics="first_source_pose_zero_velocity_explicit_reset",
                    state_order="root_p3_quat_wxyz4_root_v3_root_w3_RobotSpec_q21_qd21",
                    original_frames=len(music), source=source))
                groups.add(group); audio_hashes.add(ah); feature_hashes.add(fh)
                selected += 1
                if selected >= count:
                    break
            if selected != count:
                raise ValueError(f"{source}/{split}无足够互斥合法歌曲，不重复歌曲补数量")
    cfg.tasks = str(output/"tasks_prepare.json")
    cfg.music_baseline = None
    cfg.physical_criteria = str(output/"criteria.json")
    (output/"tasks_prepare.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=2))
    (output/"criteria.json").write_text((ROOT/"configs/closedloop/stage2_supermimic_adapter_criteria.example.json").read_text())
    OmegaConf.save(cfg, output/"runtime.yaml")
    read_tasks(cfg.tasks, preparing=True)
    assets = {k: dict(path=str(path), sha256=digest(path)) for k, path in dict(
        stage1=cfg.genmo.checkpoint, stats=cfg.genmo.stats, kinematics=cfg.genmo.kinematics,
        supermimic=cfg.supermimic.checkpoint, mjcf=Path(args.robot_asset_dir)/"bumi3_4340.xml").items()}
    (output/"assets.json").write_text(json.dumps(dict(assets=assets, joint_names=bundle.spec.joint_names,
        tasks=len(tasks), splits={s: sum(t["split"] == s for t in tasks) for s in ("train", "validation", "audit", "evaluation")},
        stage1_global_step=checkpoint["global_step"], physics_run=False, nominal_cache_prepared=False,
        music_s0_calibrated=False), ensure_ascii=False, indent=2))
    print(json.dumps(dict(config=str(output/"runtime.yaml"), tasks=tasks, assets=assets), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
