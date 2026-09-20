"""BUMI 文本完整序列 CPU 功能测试。

使用真实fe934运动学JSON、合成qpos和明确标记的替身T5特征；Transformer缩为32D两层。
验证数据/训练/损失契约，不冒充真实数据、完整网络GPU性能或动作语义质量验收。
所有生成文件仅在 pytest tmp_path；测试源代码长期保留。
"""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from gem.datasets.pure_motion.bumi_text import BumiTextDataset, caption_hash, GROUND
from gem.datamodule.mocap_trainX_testY import collate_fn
from gem.runtime.bumi_text_contract import inspect_payload, sha256_file
from gem.utils.sequence_contract import validate_sequence_experiment
from tools.data.bumi.prepare_bumi_text import build, preflight, statistics, select_records

ROOT = Path(__file__).resolve().parents[2]
KIN = ROOT / "configs/bumi/bumi_kinematics_robot_retargeter_fe934_v1.json"
LENGTHS = [60, 97, 120, 183, 299, 300]


@pytest.fixture
def text_release(tmp_path):
    spec = json.loads(KIN.read_text())
    records = []
    original = []
    for i, frames in enumerate(LENGTHS + [60, 97]):
        qpos = np.tile(spec["default_qpos"], (frames, 1)).astype(np.float32)
        qpos[:, 0] = np.linspace(0, 0.3, frames)
        qpos[:, 7] += np.sin(np.linspace(0, 3, frames)) * 0.01
        path = tmp_path / f"motion{i}.npz"
        np.savez(path, qpos=qpos, fps=30, joint_names=np.array(spec["joint_order"]))
        captions = [f"test-only walking {i}", f"test-only stepping {i}"]
        embed_path = tmp_path / f"embed{i}.pt"
        torch.save(
            dict(
                motion_id=f"m{i}",
                captions=captions,
                encoder="t5-3b",
                max_text_len=150,
                embeddings=torch.stack([torch.ones(150, 1024), torch.ones(150, 1024) * 2]).half(),
                attention_mask=torch.stack([torch.arange(150) < 5, torch.arange(150) < 8]),
            ),
            embed_path,
        )
        refs = [
            dict(
                format="bumi_text_t5_v1",
                path=str(embed_path),
                sha256=sha256_file(embed_path),
                record_index=0,
                text_index=t,
                motion_id=f"m{i}",
                caption_sha256=caption_hash(c),
            )
            for t, c in enumerate(captions)
        ]
        records.append(
            dict(
                dataset="motionmillion" if i % 2 == 0 else "humanml3d",
                motion_id=f"m{i}",
                split="train" if i < 6 else "val",
                qpos_path=str(path),
                fps=30,
                captions=captions,
                caption_ids=[f"m{i}/0", f"m{i}/1"],
                embeddings=refs,
                ground_semantics=GROUND,
                ground_alignment=dict(
                    applied=False, offset_z=0.0, reference="synthetic-known-plane"
                ),
                provenance=dict(
                    source_id=f"source{i}",
                    canonical_source_id=f"synthetic/{i}",
                    interval_seconds=[0.0, frames / 30],
                    retargeter="synthetic-test",
                    retarget_version="1",
                ),
            )
        )
        original.append(torch.from_numpy(qpos))
    source = tmp_path / "conversion.json"
    source.write_text(
        json.dumps(
            dict(
                schema="genmo.bumi_text_conversion.v1",
                kinematics=dict(path=str(KIN), sha256=sha256_file(KIN)),
                records=records,
            )
        )
    )
    root = tmp_path / "release"
    build(source, root, records_per_shard=2)
    statistics(root, root / "stats.json")
    return root, original, records


def config(root):
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
    cfg.endecoder.kinematics_path = str(KIN)
    cfg.endecoder.stats_path = str(root / "stats.json")
    for values in [cfg.train_datasets, cfg.test_datasets]:
        for ds in values.values():
            ds.root = str(root)
    validate_sequence_experiment(cfg)
    return cfg


def test_full_frames_captions_cache_batch(text_release):
    root, original, _ = text_release
    ds = BumiTextDataset(root, "train", caption_sampling="first")
    samples = [ds[i] for i in range(6)]
    by_id = {f"m{i}": original[i] for i in range(6)}
    for sample in samples:
        f = sample["length"]
        torch.testing.assert_close(sample["qpos"][:f], by_id[sample["meta"]["motion_id"]])
        assert sample["meta"]["crop_start"] == 0
        assert sample["qpos"].shape == (300, 28)
        assert sample["mask"]["valid"].tolist() == [True] * f + [False] * (300 - f)
        assert sample["text_embed"].shape == (150, 1024)
        assert sample["text_attention_mask"].sum() == 5
    samples[0]["qpos"].zero_()
    samples[0]["text_embed"].zero_()
    assert ds[0]["qpos"].abs().sum() > 0 and ds[0]["text_embed"].sum() > 0
    a, b = [BumiTextDataset(root, "train", random_seed=33) for _ in range(2)]

    def draws(dataset):
        result = []
        for _ in range(20):
            sample = dataset[0]
            tid = sample["meta"]["text_index"]
            assert sample["text_embed"][0, 0] == tid + 1
            assert sample["text_attention_mask"].sum() == (5 if tid == 0 else 8)
            result.append(tid)
        return result

    aa, bb = draws(a), draws(b)
    assert aa == bb and set(aa) == {0, 1}
    batch = collate_fn([ds[i] for i in range(6)], "train", config(root).data.collate_cfg)
    for key in ["qpos", "music_embed", "audio_array", "music_array", "foot_contact"]:
        assert batch[key].shape[1] == 300
    report = preflight(root, limit=0)
    assert sorted(map(int, report["length_distribution"])) == LENGTHS
    assert report["crop_count"] == 0
    stats = json.loads((root / "stats.json").read_text())
    assert stats["contract_version"] == "genmo.bumi_qpos30_stats.v4"
    assert stats["root_height_reference_m"] == pytest.approx(0.48120910, abs=1e-8)
    assert stats["valid_element_counts"][0] == sum(LENGTHS) - len(LENGTHS)
    assert stats["valid_element_counts"][2] == sum(LENGTHS)


def test_small_real_robot_forward_backward_and_contract(text_release):
    root, _, _ = text_release
    cfg = config(root)
    ds = BumiTextDataset(root, "train", caption_sampling="first")
    model = instantiate(cfg.model, _recursive_=False).cpu().train()
    batch = collate_fn([ds[0], ds[3]], "train", cfg.data.collate_cfg)
    model.prepare_batch(batch, "diffusion")
    model.create_condition_mask(batch, train=True)
    assert model.body_model is None and not hasattr(model, "music_embedder")
    assert batch["encoded_text"].shape == (2, 150, 1024)
    result = model.pipeline(batch, train=True, mode="diffusion", global_step=10000)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    payload = {"state_dict": model.state_dict()}
    model.on_save_checkpoint(payload)
    assert inspect_payload(payload)["feature_dim"] == 30
    model.on_load_checkpoint(payload)
    with pytest.raises(ValueError, match='weights-only'):
        model.load_pretrained_model('unused.ckpt')
    bad = copy.deepcopy(payload)
    bad["bumi_text_contract"]["assets"]["stats"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="resume"):
        model.on_load_checkpoint(bad)
    bad = copy.deepcopy(payload)
    bad["genmo_sequence_contract"]["sequence_mode"] = "crop"
    with pytest.raises(ValueError):
        model.on_load_checkpoint(bad)


@pytest.mark.parametrize("frames", [60, 97, 183, 299])
def test_robot_loss_padding_and_boundary(text_release, frames):
    root, _, _ = text_release
    cfg = config(root)
    model = instantiate(cfg.model, _recursive_=False).cpu()

    def calculate(size):
        qpos = model.endecoder.kinematics.default_qpos[None, None].repeat(1, size, 1)
        qpos[0, :frames, 0] = torch.linspace(0, 0.3, frames)
        qpos[:, frames:] = qpos[:, frames - 1 : frames]
        valid = torch.arange(size)[None] < frames
        batch = dict(qpos=qpos, length=torch.tensor([frames]), mask={"valid": valid})
        enc = model.endecoder.encode_with_aux(batch)
        for name, attr in [
            ("target_x", "normalized_features"),
            ("target_physical_features", "physical_features"),
            ("target_qpos_canonical", "canonical_qpos"),
            ("target_body_link_pos_root", "target_body_link_pos_root"),
            ("target_foot_contact", "target_foot_contact"),
            ("target_foot_contact_mask", "target_foot_contact_mask"),
            ("target_contact_ground_height", "target_contact_ground_height"),
        ]:
            batch[name] = getattr(enc, attr)
        pred = enc.normalized_features.clone().detach()
        pred[:, :, 9] += 0.1
        pred.requires_grad_()
        dec = model.endecoder.decode(pred)
        qp = model.endecoder.compose_qpos(dec)
        fk = model.endecoder.kinematics.forward_kinematics(qp)
        result = model.pipeline.losses(
            batch,
            dict(
                pred_x=pred,
                static_conf_logits=torch.zeros(1, size, 2),
                t_weights=torch.tensor([2.0]),
            ),
            dec,
            qp,
            fk,
            global_step=10000,
        )
        result["loss"].backward()
        assert not pred.grad[:, frames:].any()
        assert not pred.grad[:, frames - 1, :2].any()
        assert torch.isfinite(pred.grad).all()
        return result, enc

    a, ea = calculate(frames)
    b, eb = calculate(300)
    torch.testing.assert_close(ea.target_foot_contact, eb.target_foot_contact[:, :frames])
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=2e-5, rtol=2e-5, msg=key)


def test_corruption_and_source_overlap(text_release):
    root, _, records = text_release
    bad = copy.deepcopy(records[0])
    bad["split"] = "val"
    accepted, report = select_records([records[0], bad])
    assert len(accepted) == 1 and accepted[0]["split"] == "val"
    assert report["excluded"][0]["reason"] == "train_overlaps_held_out"
    ds = BumiTextDataset(root, "train")
    ds[0]
    path = root / ds.manifest["shards"][0]["path"]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        ds[0]


def test_lightning_cpu_two_steps_and_full_resume(text_release, tmp_path):
    """实际Trainer的合成CPU微型测试，核对fit hooks/优化器/完整恢复；不使用GPU。"""
    import pytorch_lightning as pl

    cfg = config(text_release[0])
    cfg.training_budget.max_steps = 8
    cfg.training_budget.warmup_steps = 2
    cfg.training_budget.auxiliary_warmup_steps = 2
    for split in ("train", "val"):
        cfg.data.loader_opts[split].batch_size = 2
        cfg.data.loader_opts[split].num_workers = 0

    def trainer(steps):
        return pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_steps=steps,
            max_epochs=-1,
            logger=pl.loggers.CSVLogger(tmp_path / "cpu_logs", name=f"steps{steps}"),
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            limit_val_batches=0,
            use_distributed_sampler=False,
            default_root_dir=tmp_path / "cpu_fit",
        )

    model = instantiate(cfg.model, _recursive_=False)
    dm = instantiate(cfg.data, _recursive_=False)
    first = trainer(2)
    first.fit(model, datamodule=dm)
    assert first.global_step == 2
    checkpoint = tmp_path / "synthetic_cpu_resume.ckpt"
    first.save_checkpoint(checkpoint)
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["genmo_data_identity"]["train"][0]["manifest_sha256"]
    assert saved["optimizer_states"] and saved["lr_schedulers"]
    restored = instantiate(cfg.model, _recursive_=False)
    second = trainer(3)
    second.fit(restored, datamodule=instantiate(cfg.data, _recursive_=False), ckpt_path=checkpoint)
    assert second.global_step == 3
