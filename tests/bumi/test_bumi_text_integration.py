"""BUMI文本数据拒绝路径、联合采样、网页路由和运动学预览集成测试。

真实MJCF仅做CPU FK；渲染器明确用纯色CPU替身，生成真实MP4以检查媒体协议。
窗口也使用替身，不打开图形显示、不创建网络连接；网页使用既有测试工作进程。
这些测试不代表真实GPU渲染或机器人动作质量通过。
"""

import copy
import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from gem.datasets.pure_motion.bumi_text import (
    AssetCache,
    BumiTextDataset,
    read_embedding,
    validate_record,
)
from gem.runtime.bumi_text_contract import sha256_file
from tests.bumi.test_bumi_text_fullseq import KIN, ROOT
from tests.bumi.test_bumi_text_fullseq import text_release as text_release
from tests.bumi.test_bumi_text_runtime import small_checkpoint as small_checkpoint
from tests.bumi.text_source_fixtures import release as release
from tests.test_text_motion_web import fake_worker, payload, wait_job


@pytest.mark.parametrize(
    "mutation", ["length", "shape", "finite", "quat", "interval", "ground", "caption"]
)
def test_reject_bad_records(text_release, mutation):
    record = copy.deepcopy(BumiTextDataset(text_release[0], "train").read_record(0))
    if mutation == "length":
        record["frames"] += 1
    elif mutation == "shape":
        record["qpos"] = record["qpos"][:, :-1]
    elif mutation == "finite":
        record["qpos"][0, 0] = float("nan")
    elif mutation == "quat":
        record["qpos"][0, 3:7] = 0
    elif mutation == "interval":
        record["provenance"]["interval_seconds"] = [0, 20]
    elif mutation == "ground":
        record["ground_alignment"].pop("reference")
    elif mutation == "caption":
        record["caption_ids"] = ["same", "same"]
    with pytest.raises(ValueError):
        validate_record(record)


def test_embeddings_index_joint_contract_and_source_unchanged(text_release):
    root, _, _ = text_release
    ds = BumiTextDataset(root, "train")
    record = ds.read_record(0)
    original = copy.deepcopy(record)
    before = {p: sha256_file(p) for p in (root / "shards").glob("*.pt")}
    ref = copy.deepcopy(record["embeddings"][0])
    ref["text_index"] = 1
    with pytest.raises(ValueError):
        read_embedding(ref, record["captions"][0], ds.cache, root)
    ref["text_index"] = 0
    ref["caption_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        read_embedding(ref, record["captions"][0], ds.cache, root)
    path = Path(record["embeddings"][0]["path"])
    value = torch.load(path, weights_only=False)
    value["max_text_len"] = 50
    torch.save(value, path)
    ref = copy.deepcopy(record["embeddings"][0])
    ref["sha256"] = sha256_file(path)
    with pytest.raises(ValueError, match="150"):
        read_embedding(ref, record["captions"][0], AssetCache(), root)
    assert {p: sha256_file(p) for p in before} == before
    manifest_path = root / "manifests/train.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["joint_names"][0], manifest["joint_names"][1] = (
        manifest["joint_names"][1],
        manifest["joint_names"][0],
    )
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="绑定"):
        BumiTextDataset(root, "train")
    torch.testing.assert_close(original["qpos"], record["qpos"])


def test_multi_dataset_sampler_and_config(text_release):
    from hydra import compose, initialize_config_dir

    from gem.datamodule.sequence_sampler import (
        ShardAwareDistributedSampler,
        ShardConcatDataset,
    )
    from gem.utils.sequence_contract import validate_sequence_experiment

    root = text_release[0]
    datasets = [
        BumiTextDataset(root, "train", dataset=name) for name in ("motionmillion", "humanml3d")
    ]
    joined = ShardConcatDataset(datasets)
    ranks = [
        ShardAwareDistributedSampler(joined, rank=r, num_replicas=2, drop_last=True)
        for r in range(2)
    ]
    samples = [list(s) for s in ranks]
    assert len(samples[0]) == len(samples[1]) == 3
    assert not set(samples[0]) & set(samples[1]) and set(sum(samples, [])) == set(range(6))
    for sampler in ranks:
        sampler.set_epoch(3)
    assert [list(s) for s in ranks] != samples
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=["exp=gem_bumi_text_fullseq"])
    validate_sequence_experiment(cfg)
    assert not cfg.pl_trainer.use_distributed_sampler and cfg.pl_trainer.max_epochs == -1
    assert cfg.pipeline.args.weights.root_tilt == 1.0
    cfg.training_budget.max_steps = 8
    with pytest.raises(ValueError, match="预算"):
        validate_sequence_experiment(cfg)
    cfg.training_budget.warmup_steps = cfg.training_budget.auxiliary_warmup_steps = 2
    validate_sequence_experiment(cfg)


@pytest.mark.parametrize("pose", ["stand", "walk", "jump", "crouch", "lie"])
def test_pose_domain_and_fk(pose):
    from gem.robots.bumi.feature_codec import BumiMotionFeatureCodec
    from gem.robots.bumi.kinematics import BumiKinematics

    kin = BumiKinematics(KIN)
    codec = BumiMotionFeatureCodec(kin)
    qpos = kin.default_qpos[None].repeat(97, 1)
    if pose == "walk":
        qpos[:, 0] = torch.linspace(0, 1, 97)
    elif pose == "jump":
        qpos[:, 2] += torch.sin(torch.linspace(0, torch.pi, 97)) * 0.5
    elif pose == "crouch":
        qpos[:, 2] -= 0.25
        qpos[:, 7] += 0.1
    elif pose == "lie":
        qpos[:, 3:7] = torch.tensor([0.70710678, 0, 0.70710678, 0])
        qpos[:, 2] = 0.15
    encoded = codec.encode(qpos)
    decoded = codec.decode_to_canonical_qpos(encoded.physical_features)
    assert torch.isfinite(decoded).all()
    fk = kin.forward_kinematics(qpos)
    assert torch.isfinite(fk["body_pos_w"]).all()
    # 用真实MJCF的CPU前向运动学核对训练FK，渲染不依赖这里的替身。
    import mujoco

    from gem.runtime.bumi_preview import validate_robot_assets

    xml, _ = validate_robot_assets(ROOT / "assets/bumi_viewer/manifest.json", KIN)
    mj_model = mujoco.MjModel.from_xml_path(str(xml))
    mj_data = mujoco.MjData(mj_model)
    bodies = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name) for name in kin.body_order
    ]
    assert min(bodies) >= 0
    for frame in (0, 48, 96):
        mj_data.qpos[:] = qpos[frame].numpy()
        mujoco.mj_forward(mj_model, mj_data)
        np.testing.assert_allclose(mj_data.xpos[bodies], fk["body_pos_w"][frame].numpy(), atol=2e-6)
        dot = (mj_data.xquat[bodies] * fk["body_quat_w"][frame].numpy()).sum(-1)
        np.testing.assert_allclose(np.abs(dot), 1, atol=2e-6)
    # 只检查可表示性和竖直轨迹保留，不宣称姿态可跟踪。
    torch.testing.assert_close(encoded.physical_features[:, 2], qpos[:, 2] - kin.default_qpos[2])


def test_bumi_web_contract_range_and_recovery(small_checkpoint, tmp_path):
    from gem.runtime.text_motion_web.app import create_app
    from gem.runtime.text_motion_web.models import ModelRegistry
    from gem.runtime.text_motion_web.service import JobService

    registry = ModelRegistry(tmp_path / "web", roots=[])
    model = registry.add(str(small_checkpoint))
    assert model["motion_backend"] == "bumi" and (model["min_frames"], model["max_frames"]) == (
        60,
        300,
    )
    service = JobService(
        tmp_path / "web", registry=registry, worker_target=fake_worker, discover=False
    )
    try:
        client = create_app(service).test_client()
        for frames in (59, 301):
            assert (
                client.post("/api/jobs", json=payload(model, num_frames=frames)).status_code == 400
            )
        good = wait_job(service, service.submit(payload(model, num_frames=240))["id"])
        failed = wait_job(service, service.submit(payload(model, prompt="render_fail"))["id"])
        assert good["status"] == "done" and failed["status"] == "failed"
        result = client.get(good["video_url"], headers={"Range": "bytes=0-4"})
        assert result.status_code == 206
        result.close()
        assert service.history()["history_limit"] == 60
    finally:
        service.close()


def test_cpu_video_artifact_and_no_network_preview(tmp_path, monkeypatch):
    import mujoco

    from gem.runtime.bumi_text_viewer import TextPreviewPlayer
    from gem.runtime.text_motion_web.worker import check_video, render_job

    spec = json.loads(KIN.read_text())
    qpos = np.tile(spec["default_qpos"], (60, 1))
    np.savez(
        tmp_path / "motion.npz",
        qpos=qpos,
        qpos_raw=qpos,
        foot_contact_logits=np.zeros((60, 2)),
        fps=30,
        quaternion_convention="wxyz",
        joint_names=spec["joint_order"],
    )
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            dict(
                motion_backend="bumi",
                robot_manifest=str(ROOT / "assets/bumi_viewer/manifest.json"),
                kinematics_path=str(KIN),
            )
        )
    )

    class CpuRenderer:
        def __init__(self, *args, **kwargs):
            pass

        def update_scene(self, data, camera):
            pass

        def render(self):
            return np.zeros((720, 1280, 3), dtype=np.uint8)

        def close(self):
            pass

    monkeypatch.setattr(mujoco, "Renderer", CpuRenderer)
    render_job(tmp_path, 60)
    assert check_video(tmp_path / "video.mp4", 60)["fully_decoded"]
    assert (tmp_path / "thumbnail.jpg").is_file()

    class WindowStub:
        def __init__(self, manifest, kin, factory):
            self.reader = factory()

        def close(self):
            self.reader.close()

    monkeypatch.setattr("gem.runtime.bumi_text_viewer.MujocoPreview", WindowStub)
    import socket

    monkeypatch.setattr(
        socket, "socket", lambda *a, **k: (_ for _ in ()).throw(AssertionError("不应创建网络连接"))
    )
    player = TextPreviewPlayer(ROOT / "assets/bumi_viewer/manifest.json", KIN)
    try:
        player.play(qpos)
        time.sleep(0.06)
        assert player.request({"command": "preview_frame"})["state"] == "PLAYING"
        player.pause()
        assert player.state == "PAUSED"
        player.resume()
        player.stand()
        assert player.frames is None
    finally:
        player.close()


def test_fixed_cohort_identity(text_release, tmp_path):
    from tools.eval.evaluate_bumi_text import cohort

    a = cohort(text_release[0], tmp_path / "cohort1.json", per_dataset=1)
    b = cohort(text_release[0], tmp_path / "cohort2.json", per_dataset=1)
    assert a == b and sorted(row["frames"] for row in a["records"]) == [60, 97]
    assert a["protocol"] == "full_sequence_matched_length" and not a["official_human_evaluator"]
    with pytest.raises(ValueError, match="不足"):
        cohort(text_release[0], tmp_path / "cohort128.json")


def test_reuse_original_motionmillion_embeddings(release):
    from gem.datasets.pure_motion.bumi_text import caption_hash

    motion, text, *_ = release
    mp, ep = motion / "manifests/train.json", text / "manifests/train.json"
    ref = dict(
        format="motionmillion_t5_v1",
        source_split="train",
        motion_manifest=str(mp),
        embedding_manifest=str(ep),
        motion_manifest_sha256=sha256_file(mp),
        embedding_manifest_sha256=sha256_file(ep),
        shard_id=0,
        record_index=1,
        text_index=1,
        motion_id="MotionGV/1",
        caption_sha256=caption_hash("a person walks"),
    )
    before = [sha256_file(p) for p in (mp, ep, motion / "shard.pth", text / "shard.pth")]
    embedding, mask = read_embedding(
        ref, "a person walks", AssetCache(), motion, expected_frames=97
    )
    assert mask.sum() == 5 and (embedding[:5] == 2).all()
    with pytest.raises(ValueError, match="帧数"):
        read_embedding(ref, "a person walks", AssetCache(), motion, expected_frames=120)
    wrong = copy.deepcopy(ref)
    wrong["motion_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256"):
        read_embedding(wrong, "a person walks", AssetCache(), motion)
    assert before == [sha256_file(p) for p in (mp, ep, motion / "shard.pth", text / "shard.pth")]


def test_new_feature_writer_only_encodes_missing(text_release, tmp_path, monkeypatch):
    from gem.runtime.bumi_text_runtime import ResidentBumiTextEngine
    from tools.data.bumi.encode_text_features import encode

    root, _, records = text_release
    rows = copy.deepcopy(records)
    rows[0]["embeddings"] = []
    source = tmp_path / "missing_features.json"
    source.write_text(
        json.dumps(
            dict(
                schema="genmo.bumi_text_conversion.v1",
                kinematics=dict(path=str(KIN), sha256=sha256_file(KIN)),
                records=rows,
            )
        )
    )
    calls = []

    def fake_encode(self, caption):
        calls.append(caption)
        return torch.ones(1, 150, 1024), torch.arange(150)[None] < 5

    monkeypatch.setattr(ResidentBumiTextEngine, "encode_prompt", fake_encode)
    converted = encode(source, tmp_path / "new_features", "unused-t5-test-stub", device="cpu")
    result = json.loads(converted.read_text())
    assert calls == rows[0]["captions"]
    assert result["records"][1]["embeddings"] == rows[1]["embeddings"]
    ref = result["records"][0]["embeddings"][0]
    read_embedding(ref, rows[0]["captions"][0], AssetCache(), converted.parent)


def test_cpu_small_model_diagnostic_report(text_release, small_checkpoint, tmp_path):
    from tools.eval.evaluate_bumi_text import cohort, run

    path = tmp_path / "cohort.json"
    cohort(text_release[0], path, per_dataset=1)
    report = run(text_release[0], path, small_checkpoint, tmp_path / "evaluation", device="cpu")
    assert len(report["records"]) == 2 and report["semantic_metrics"] == "manual_only"
    for row in report["records"]:
        with np.load(tmp_path / "evaluation" / row["output_dir"] / "motion.npz") as data:
            assert len(data["qpos"]) == row["frames"]
        assert set(row["metrics"]) == {"gt", "raw", "postprocessed"}


def test_web_reuses_bumi_engine_and_rejects_smpl(tmp_path, monkeypatch):
    import queue

    import gem.runtime.bumi_text_runtime as bumi
    from gem.runtime.text_motion_web import worker

    calls = []

    class EngineStub:
        max_text_len = 150
        backend = "smpl"

        def __init__(self, **kwargs):
            calls.append(("create", self.backend))

        def initialize(self):
            pass

        def set_ddim_steps(self, steps):
            calls.append(("ddim", steps))

        def generate(self, request):
            return dict(ok=True, output_dir=request["output_root"], timing={})

        def close(self):
            calls.append(("close", self.backend))

    class RobotStub(EngineStub):
        backend = "bumi"

    monkeypatch.setattr(bumi, "ResidentBumiTextEngine", RobotStub)
    monkeypatch.setattr(worker, "follow_parent", lambda: None)
    monkeypatch.setattr(worker, "run_renderer", lambda *args: {"fully_decoded": True})
    commands, events = queue.Queue(), queue.Queue()
    for index, backend in enumerate(["bumi", "bumi", "smpl", "bumi"]):
        path = tmp_path / backend
        if not path.exists():
            path.write_text("stub")
        commands.put(
            dict(
                id=str(index),
                model=dict(
                    path=str(path), fingerprint=worker.fingerprint(path), motion_backend=backend
                ),
                prompt="test-only",
                num_frames=120,
                ddim_steps=20 + index,
                task_dir=str(tmp_path / str(index)),
            )
        )
    commands.put(None)
    worker.worker_main(commands, events)
    assert calls == [
        ("create", "bumi"),
        ("ddim", 21),
        ("close", "bumi"),
        ("create", "bumi"),
        ("close", "bumi"),
    ]
    assert sum(events.get()["status"] == "done" for _ in range(events.qsize())) == 3


def test_checkpoint_assets_can_move_without_changing_identity(small_checkpoint, tmp_path):
    import shutil

    from gem.runtime.text_motion_web.models import inspect_checkpoint

    payload = torch.load(small_checkpoint, weights_only=False)
    directory = tmp_path / "moved_checkpoint"
    (directory / "assets").mkdir(parents=True)
    for name, asset in payload["bumi_text_contract"]["assets"].items():
        shutil.copyfile(asset["path"], directory / "assets" / f"{name}.json")
        asset["path"] = f"/nonexistent-training-server/{name}.json"
    path = directory / "model.ckpt"
    torch.save(payload, path)
    assert inspect_checkpoint(path)["motion_backend"] == "bumi"


def test_public_model_keeps_backend_length_without_paths():
    from gem.runtime.text_motion_web.share import public_model

    public = public_model(
        dict(
            id="test",
            motion_backend="bumi",
            min_frames=60,
            max_frames=300,
            path="/private/model.ckpt",
            contract={"max_text_len": 150, "assets": {"path": "private"}},
        )
    )
    assert (public["motion_backend"], public["min_frames"], public["max_frames"]) == (
        "bumi",
        60,
        300,
    )
    assert "path" not in public and "assets" not in public["contract"]
