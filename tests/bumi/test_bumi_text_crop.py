"""BUMI四库120帧训练的行为回归测试。

使用真实BUMI运动学和合成完整qpos，沿用旧测试的明确标记替身T5特征；验证长源动作
不被截存、事件文本与窗口对应、短动作mask、确定性验证和分布式分层概率。
微型CPU训练与ONNX测试用于检查实现及恢复契约，不代表真实四库训练或生成质量。
全部临时数据、checkpoint和导出文件只写入pytest管理的临时目录。
"""

import copy
import json
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from gem.datamodule.bumi_text_sampler import BumiTextDistributedSampler
from gem.datamodule.mocap_trainX_testY import collate_fn
from gem.datasets.pure_motion.bumi_text import BumiTextDataset
from gem.runtime.bumi_text_contract import inspect_payload
from gem.utils.sequence_contract import validate_sequence_experiment
from tests.bumi.test_bumi_text_fullseq import KIN, ROOT
from tests.bumi.test_bumi_text_fullseq import text_release as text_release
from tools.data.bumi.prepare_bumi_text import build, preflight, statistics
from tools.data.bumi.text_windows import internal_split, temporal_captions


@pytest.fixture
def crop_release(text_release, tmp_path):
    source = json.loads((tmp_path / "conversion.json").read_text())
    source["source_storage"] = "full"
    spec = json.loads(KIN.read_text())
    for i, record in enumerate(source["records"]):
        dataset = ["motionmillion", "humanml3d", "kitml", "bones_seed"][i % 4]
        frames = [600, 45, 180, 600][i % 4]
        qpos = np.tile(spec["default_qpos"], (frames, 1)).astype(np.float32)
        qpos[:, 0] = np.arange(frames) / 1000
        np.savez(record["qpos_path"], qpos=qpos, fps=30, joint_names=np.array(spec["joint_order"]))
        record.update(dataset=dataset, split="train" if i < 4 else "val")
        record["provenance"]["interval_seconds"] = [0, frames / 30]
        if dataset == "bones_seed":
            record.update(
                text_annotation_scope="temporal", caption_intervals=[[30, 240], [400, 460]]
            )
    path = tmp_path / "crop_conversion.json"
    path.write_text(json.dumps(source))
    root = tmp_path / "crop_release"
    build(path, root, records_per_shard=2)
    statistics(root, root / "stats.json", sequence_mode="crop")
    return root


def crop_config(root):
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(
            config_name="train",
            overrides=[
                "exp=gem_bumi_text_fullseq",
                "network.model_cfg.denoiser.latent_dim=32",
                "network.model_cfg.denoiser.num_layers=2",
                "network.model_cfg.denoiser.num_heads=4",
            ],
        )
    cfg.endecoder.kinematics_path, cfg.endecoder.stats_path = str(KIN), str(root / "stats.json")
    for values in (cfg.train_datasets, cfg.test_datasets):
        for ds in values.values():
            ds.root = str(root)
    validate_sequence_experiment(cfg)
    return cfg


def dataset(root, name, split="train"):
    return BumiTextDataset(
        root,
        split,
        dataset=name,
        sequence_mode="crop",
        pad_to_frames=120,
        caption_sampling="random",
        require_temporal_annotations=name == "bones_seed",
    )


def test_complete_source_random_crop_short_mask_and_validation(crop_release):
    mm = dataset(crop_release, "motionmillion")
    assert mm.read_record(0)["qpos"].shape == (600, 28)
    starts = {mm.get_window(0, random_seed=i)["meta"]["crop_start"] for i in range(20)}
    assert len(starts) > 10 and max(starts) > 120
    a, b = [mm.get_window(0, random_seed=42) for _ in range(2)]
    torch.testing.assert_close(a["qpos"], b["qpos"])
    short = dataset(crop_release, "humanml3d")[0]
    assert short["length"] == 45 and short["mask"]["valid"].sum() == 45
    assert short["qpos"].shape == (120, 28)
    assert not short["foot_contact_available"][45:].any()
    torch.testing.assert_close(short["qpos"][45:], short["qpos"][44:45].expand(75, -1))
    val = dataset(crop_release, "motionmillion", "val")
    samples = [val.get_window(0, random_seed=i) for i in range(5)]
    assert all(s["meta"]["crop_start"] == 240 and s["meta"]["text_index"] == 0 for s in samples)
    for s in samples:
        torch.testing.assert_close(s["qpos"], samples[0]["qpos"])
    report = preflight(crop_release, limit=0, sequence_mode="crop")
    assert report["crop_count"] == 0 and report["records_checked"] == 4


def test_bones_event_pairing_and_short_event(crop_release):
    ds = dataset(crop_release, "bones_seed")
    seen = set()
    for i in range(20):
        sample = ds.get_window(0, random_seed=i)
        meta = sample["meta"]
        a, b = meta["annotation_interval_frames"]
        assert a <= meta["crop_start"] < meta["crop_end"] <= b
        assert sample["text_embed"][0, 0] == meta["text_index"] + 1
        assert sample["length"] == (120 if meta["text_index"] == 0 else 60)
        seen.add(meta["text_index"])
    assert seen == {0, 1}
    ds.read_record(0).pop("caption_intervals")
    with pytest.raises(ValueError, match="BONES"):
        ds[0]


def test_online_release_parallel_build_stats_and_no_zero_text(crop_release, tmp_path):
    source = json.loads((tmp_path / "crop_conversion.json").read_text())
    for record in source["records"]:
        record["embeddings"] = []
    path = tmp_path / "online_conversion.json"
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="数量"):
        build(path, tmp_path / "missing_embeddings")
    root = tmp_path / "online_release"
    build(path, root, records_per_shard=2, workers=2, text_feature_mode="online_t5")
    serial = statistics(root, root / "stats.json", sequence_mode="crop")
    parallel = statistics(root, root / "stats_parallel.json", sequence_mode="crop", workers=2)
    for key in ("mean", "std", "valid_element_counts"):
        np.testing.assert_allclose(serial[key], parallel[key], atol=1e-10)
    sample = dataset(root, "bones_seed").get_window(0, random_seed=42)
    assert "text_embed" not in sample and sample["meta"]["text_feature_mode"] == "online_t5"
    cfg = crop_config(root)
    batch = collate_fn([sample], "train", cfg.data.collate_cfg)
    assert "text_embed" not in batch
    model = instantiate(cfg.model, _recursive_=False).cpu()
    with pytest.raises(ValueError, match="冻结 T5"):
        model.prepare_batch(batch, "diffusion")
    offline = dataset(crop_release, "bones_seed")[0]
    with pytest.raises(ValueError, match="混用"):
        collate_fn([sample, offline], "train", cfg.data.collate_cfg)
    assert not (tmp_path / "missing_embeddings").exists()


def test_temporal_seconds_resampling_and_invalid_ranges():
    events = [
        dict(start_time=1.88, end_time=3.53, description="原始事件描述"),
        dict(start_time=3.53, end_time=4.83, description="最后事件"),
    ]
    texts, intervals, origin = temporal_captions(events, 143)
    assert texts == ["原始事件描述", "最后事件"]
    assert intervals == [[57, 105], [106, 143]]
    assert origin[-1]["end_clipped"]
    for end in (float("nan"), 10, -1):
        with pytest.raises(ValueError):
            temporal_captions([dict(start_time=0, end_time=end, description="事件")], 143)
    assert temporal_captions([dict(start_time=0, end_time=0.05, description="太短")], 143)[0] == []
    assert internal_split("母来源") == internal_split("母来源")


def test_weighted_draws_source_balance_and_ddp():
    names = ["motionmillion", "humanml3d", "kitml", "bones_seed"]
    datasets = [
        SimpleNamespace(
            dataset=n,
            split="train",
            sequence_mode="crop",
            sample_source_groups=lambda: ["parent_a"] * 5 + ["parent_b"],
        )
        for n in names
    ]
    probabilities = dict(zip(names, [0.05, 0.25, 0.10, 0.60]))

    def sampler(**kwargs):
        return BumiTextDistributedSampler(datasets, probabilities, 20000, **kwargs)

    single = list(sampler())
    counts = Counter(d.dataset_index for d in single)
    for index, name in enumerate(names):
        assert counts[index] / len(single) == pytest.approx(probabilities[name], abs=0.015)
    assert sum(d.record_index == 5 for d in single) / len(single) == pytest.approx(0.5, abs=0.015)
    shards = [list(sampler(rank=i, num_replicas=2)) for i in range(2)]
    assert sorted(shards[0] + shards[1], key=lambda x: x.draw_index) == single
    changed = sampler()
    changed.set_epoch(1)
    assert list(changed)[:20] != single[:20]


def test_crop_model_loss_contract_resume_and_export(crop_release, tmp_path):
    from gem.runtime.bumi_text_runtime import OnnxTextStep, load_checkpoint_step
    from tools.export.bumi_text import export, sample_inputs

    cfg = crop_config(crop_release)
    model = instantiate(cfg.model, _recursive_=False).cpu().train()
    batch = collate_fn(
        [dataset(crop_release, "humanml3d")[0], dataset(crop_release, "bones_seed")[0]],
        "train",
        cfg.data.collate_cfg,
    )
    model.prepare_batch(batch, "diffusion")
    model.create_condition_mask(batch, train=True)
    assert batch["L"] == 120
    result = model.pipeline(batch, train=True, mode="diffusion", global_step=10000)
    result["loss"].backward()
    assert torch.isfinite(result["loss"])
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    payload = dict(state_dict=model.state_dict())
    model.on_save_checkpoint(payload)
    assert inspect_payload(payload)["sequence"]["pad_to_frames"] == 120
    model.on_load_checkpoint(payload)
    bad = copy.deepcopy(payload)
    bad["bumi_text_contract"]["sampling"]["dataset_probabilities"]["motionmillion"] = 0.10
    with pytest.raises(ValueError, match="resume"):
        model.on_load_checkpoint(bad)
    checkpoint = tmp_path / "crop.ckpt"
    torch.save(payload, checkpoint)
    path = tmp_path / "onnx" / "crop.onnx"
    meta = export(checkpoint, path)
    assert meta["inputs"]["noisy_motion"] == (1, 120, 30)
    step = OnnxTextStep(path)
    original, _, _ = load_checkpoint_step(checkpoint, "cpu")
    inputs = sample_inputs(frames=45, tensor_frames=120)
    with torch.no_grad():
        for a, b in zip(original(*inputs), step(*inputs)):
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)


def test_four_dataset_lightning_train_validation_and_resume(crop_release, tmp_path):
    import pytorch_lightning as pl

    from tools.eval.evaluate_bumi_text import cohort

    cfg = crop_config(crop_release)
    cfg.training_budget.max_steps = 8
    cfg.training_budget.warmup_steps = cfg.training_budget.auxiliary_warmup_steps = 1
    cfg.data.text_sampling.samples_per_epoch = 4
    for split in ("train", "val"):
        cfg.data.loader_opts[split].batch_size = 2
        cfg.data.loader_opts[split].num_workers = 0
    dm = instantiate(cfg.data, _recursive_=False)
    assert isinstance(dm.train_dataloader().sampler, BumiTextDistributedSampler)

    def trainer(steps):
        return pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_steps=steps,
            max_epochs=-1,
            logger=pl.loggers.CSVLogger(tmp_path / "logs", name=f"steps{steps}"),
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            limit_val_batches=1,
            use_distributed_sampler=False,
            default_root_dir=tmp_path / "fit",
        )

    first = trainer(3)
    model = instantiate(cfg.model, _recursive_=False)
    first.fit(model, datamodule=dm)
    assert first.global_step == 3
    assert first.train_dataloader.sampler.epoch == 1
    checkpoint = tmp_path / "resume.ckpt"
    first.save_checkpoint(checkpoint)
    resumed = trainer(4)
    resumed.fit(
        instantiate(cfg.model, _recursive_=False),
        datamodule=instantiate(cfg.data, _recursive_=False),
        ckpt_path=checkpoint,
    )
    assert resumed.global_step == 4
    a = cohort(crop_release, tmp_path / "cohort_a.json", per_dataset=1, sequence_mode="crop")
    b = cohort(crop_release, tmp_path / "cohort_b.json", per_dataset=1, sequence_mode="crop")
    assert a == b and len(a["records"]) == 4
    assert max(row["frames"] for row in a["records"]) == 120


def test_four_conversion_keeps_sources_and_binds_event_text(tmp_path):
    from gem.runtime.bumi_text_contract import sha256_file
    from tools.data.bumi.text_windows import four_dataset_conversion

    temporal = tmp_path / "events.jsonl"
    temporal.write_text(
        json.dumps(
            dict(
                filename="motion",
                num_events=2,
                events=[
                    dict(start_time=0, end_time=5, description="第一事件"),
                    dict(start_time=5, end_time=20, description="第二事件"),
                ],
            )
        )
        + "\n"
    )
    releases = []
    for name in ("motionmillion", "humanml3d", "kitml", "bones_seed"):
        release = tmp_path / name
        (release / "manifests").mkdir(parents=True)
        report = release / "quality"
        report.mkdir()
        (report / "run.json").write_text(
            json.dumps(
                dict(
                    state="complete",
                    partial_scan=False,
                    fingerprint=name,
                    identity=dict(paths=dict(kinematics=str(KIN))),
                )
            )
        )
        row = dict(
            motion_id="motion",
            canonical_source_id=name + ":motion",
            frames=600,
            status="PASS",
            fps=30,
            captions=[dict(caption="完整动作描述")],
            split="train" if name == "humanml3d" else "unassigned",
            source_path=f"/source/{name}/motion.npz",
        )
        manifest = release / "manifests/pass.jsonl"
        manifest.write_text(json.dumps(row) + "\n")
        (release / "dataset_info.json").write_text(
            json.dumps(
                dict(
                    schema=f"genmo.{name}_umr_pass.v1",
                    quality_report=str(report),
                    quality_fingerprint=name,
                    manifests={"pass.jsonl": sha256_file(manifest)},
                )
            )
        )
        releases.append(release)
    output = tmp_path / "conversion.json"
    four_dataset_conversion(releases, temporal, output)
    payload = json.loads(output.read_text())
    assert payload["source_storage"] == "full" and len(payload["quality_reports"]) == 4
    bones = next(r for r in payload["records"] if r["dataset"] == "bones_seed")
    assert bones["frames"] == 600 and bones["qpos_path"] == "/source/bones_seed/motion.npz"
    assert bones["captions"] == ["第一事件", "第二事件"]
    assert bones["caption_intervals"] == [[0, 150], [150, 600]]
    assert bones["provenance"]["temporal_annotations"]["sha256"] == sha256_file(temporal)
    assert not payload["cross_dataset_lineage_verified"]
    with pytest.raises(FileExistsError):
        four_dataset_conversion(releases, temporal, output)
