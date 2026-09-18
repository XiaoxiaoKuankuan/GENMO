"""路线 A0 的 CPU 数据、注意力、监督和契约回归测试。

临时分片只含人工构造的动作/T5数组，不是正式MotionMillion制品。身体模型和相机
可在数据组批测试中替换为明确的小型替身；Transformer使用真实实现的小尺寸配置，
关闭dropout并开启非零attention门以检验padding不变性。这些测试不加载T5-3B、
不启动GPU/DDP、不代表真实模型质量或显存验收。真实环境命令见路线A0文档。
"""

from __future__ import annotations

import builtins
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from gem.datamodule.mocap_trainX_testY import collate_fn
from gem.datamodule.motionmillion_sampler import ShardAwareDistributedSampler
from gem.datasets.pure_motion.base_dataset import BaseDataset
from gem.datasets.pure_motion.motionmillion import MotionMillionDataset
from gem.gem import GEM
from gem.network.base_arch.transformer.encoder_rope import RoPEAttention
from gem.network.gem_denoiser import NetworkEncoderRoPE, valid_length_attention_mask
from gem.utils.masked_reduction import masked_reduce, temporal_valid_mask
from gem.utils.sequence_contract import (
    validate_generation_length,
    validate_resume_contract,
    validate_sequence_experiment,
)
from scripts.demo.demo_smpl_text import validate_text_generation_payload
from tools.data.motionmillion.common import SAMPLE_INDEX_DTYPE, MotionMillionError, sha256_file
from tools.data.motionmillion.preflight_motionmillion import sequence_statistics
from tools.eval.motionmillion_protocol import length_protocol, prediction_length
from tools.eval.summarize_motionmillion_metrics import choose_candidate, summarize

ROOT = Path(__file__).resolve().parents[1]
LENGTHS = (60, 97, 120, 183, 299, 300)


def full_config(overrides=()):
    OmegaConf.register_new_resolver("eval", builtins.eval, replace=True)
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        return compose(
            config_name="train", overrides=["exp=gem_smpl_motionmillion_text_fullseq", *overrides]
        )


@pytest.fixture()
def release(tmp_path):
    motions, embeddings = [], []
    for index, frames in enumerate(LENGTHS):
        pose = torch.arange(frames).float()[:, None].expand(-1, 66).clone() * 0.0001
        trans = torch.zeros(frames, 3)
        trans[:, 0] = torch.arange(frames) * 0.01
        trans[:, 1] = 1.0
        motions.append(
            {
                "motion_id": f"MotionGV/{index}",
                "pose": pose,
                "trans": trans,
                "beta": torch.zeros(10),
                "captions": ["walk forward", "a person walks"],
                "fps": 30,
                "source_up_axis": "y",
                "source_archive": "MotionGV/source.tar.gz",
                "source_archive_sha256": "a" * 64,
                "source_subset": "MotionGV",
                "source_member": f"{index}.npy",
                "source_sha256": "b" * 64,
                "source_text_member": f"{index}.txt",
                "source_text_sha256": "c" * 64,
                "split": "train",
            }
        )
        embeddings.append(
            {
                "motion_id": f"MotionGV/{index}",
                "offsets": torch.tensor([0, 3, 8]),
                "embeddings": torch.cat([torch.ones(3, 1024), torch.full((5, 1024), 2)]).half(),
            }
        )
    motion_root, embedding_root = tmp_path / "motion", tmp_path / "text"
    for root in (motion_root, embedding_root):
        (root / "manifests").mkdir(parents=True)
    torch.save(motions, motion_root / "shard.pth")
    torch.save(embeddings, embedding_root / "shard.pth")
    np.save(
        motion_root / "index.npy",
        np.array([(0, i, f, 0) for i, f in enumerate(LENGTHS)], dtype=SAMPLE_INDEX_DTYPE),
    )
    motion = {
        "schema_version": 1,
        "split": "train",
        "build_fingerprint": "f" * 64,
        "motion_frames": 120,
        "fps": 30,
        "source_up_axis": "y",
        "record_count": len(LENGTHS),
        "sample_count": len(LENGTHS),
        "sample_index_path": "index.npy",
        "shards": [
            {
                "shard_id": 0,
                "record_count": len(LENGTHS),
                "path": "shard.pth",
                "sha256": sha256_file(motion_root / "shard.pth"),
            }
        ],
    }
    embedding = {
        "schema_version": 1,
        "split": "train",
        "source_build_fingerprint": "f" * 64,
        "max_text_tokens": 150,
        "hidden_dim": 1024,
        "shards": [
            {
                "shard_id": 0,
                "record_count": len(LENGTHS),
                "path": "shard.pth",
                "sha256": sha256_file(embedding_root / "shard.pth"),
                "source_motion_path": "shard.pth",
                "source_motion_sha256": motion["shards"][0]["sha256"],
            }
        ],
    }
    for root, manifest in ((motion_root, motion), (embedding_root, embedding)):
        (root / "manifests/train.json").write_text(json.dumps(manifest))
    return motion_root, embedding_root, motions, embeddings


def make_dataset(release, monkeypatch, **kwargs):
    def metadata_only(self, cam_augmentation, limit_size=None):
        self.cam_augmentation, self.limit_size = cam_augmentation, limit_size
        self._load_dataset()
        self._get_idx2meta()

    monkeypatch.setattr(BaseDataset, "__init__", metadata_only)
    motion, text, *_ = release
    settings = dict(sequence_mode="full", pad_to_frames=300, caption_sampling="random")
    settings.update(kwargs)
    return MotionMillionDataset(
        motion_manifest_path=motion / "manifests/train.json",
        embedding_manifest_path=text / "manifests/train.json",
        **settings,
    )


@pytest.mark.parametrize("index,frames", list(enumerate(LENGTHS)))
def test_complete_frames_padding_caption_and_cache(release, monkeypatch, index, frames):
    torch.manual_seed(234)
    dataset = make_dataset(release, monkeypatch)
    original = release[2][index]
    hashes = [
        sha256_file(path)
        for root in release[:2]
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    samples = [dataset._load_data(index) for _ in range(16)]
    assert {s["text_index"] for s in samples} == {0, 1}
    for sample in samples:
        assert sample["valid_length"] == sample["source_frames"] == frames
        assert sample["crop_start"] == 0 and sample["sequence_mode"] == "full"
        assert sample["body_pose"].shape == (300, 63)
        assert torch.equal(sample["body_pose"][:frames], original["pose"][:, 3:])
        assert torch.equal(sample["transl"][:frames], original["trans"])
        assert torch.equal(
            sample["transl"][frames:], original["trans"][-1:].expand(300 - frames, -1)
        )
        assert sample["text_embed"].shape == (150, 1024)
        token_count = (3, 5)[sample["text_index"]]
        assert sample["caption"] == original["captions"][sample["text_index"]]
        assert sample["text_attention_mask"].shape == (150,)
        assert sample["text_attention_mask"].sum() == token_count
        assert (sample["text_embed"][:token_count] == sample["text_index"] + 1).all()
        sample["body_pose"].fill_(99)
        sample["transl"].fill_(99)
        sample["text_embed"].fill_(99)
    cached = next(iter(dataset._motion_cache.values()))[index]
    assert torch.equal(cached["pose"], original["pose"])
    assert torch.equal(cached["trans"], original["trans"])
    assert hashes == [
        sha256_file(path)
        for root in release[:2]
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def test_caption_seed_reproducibility_and_legacy(release, monkeypatch):
    torch.manual_seed(431)
    first = make_dataset(release, monkeypatch)
    sequence = [first._load_data(0)["text_index"] for _ in range(20)]
    torch.manual_seed(431)
    second = make_dataset(release, monkeypatch)
    assert sequence == [second._load_data(0)["text_index"] for _ in range(20)]
    fixed = make_dataset(release, monkeypatch, caption_sampling="first")
    assert {fixed._load_data(2)["text_index"] for _ in range(8)} == {0}
    legacy = make_dataset(
        release,
        monkeypatch,
        sequence_mode="crop",
        pad_to_frames=120,
        random_crop=False,
        caption_sampling=None,
    )
    sample = legacy._load_data(3)
    assert sample["crop_start"] == (183 - 120) // 2
    assert sample["valid_length"] == 120 and sample["text_index"] == 0
    with pytest.raises(ValueError, match="random_crop"):
        make_dataset(release, monkeypatch, random_crop=False)
    with pytest.raises(MotionMillionError, match="motion_frames"):
        make_dataset(release, monkeypatch, sequence_mode="crop", motion_frames=300)


@pytest.mark.parametrize(
    "problem",
    [
        "fingerprint",
        "sha",
        "split",
        "index",
        "length",
        "shape",
        "beta",
        "embedding",
        "caption_count",
    ],
)
def test_release_corruption_rejected(release, monkeypatch, problem):
    motion, text, records, embeddings = release
    if problem in {"fingerprint", "sha", "split"}:
        path = text / "manifests/train.json"
        manifest = json.loads(path.read_text())
        if problem == "fingerprint":
            manifest["source_build_fingerprint"] = "other"
        elif problem == "sha":
            manifest["shards"][0]["source_motion_sha256"] = "0" * 64
        else:
            manifest["split"] = "val"
        path.write_text(json.dumps(manifest))
    elif problem == "index":
        index = np.load(motion / "index.npy")
        index[0]["frames"] = 97
        np.save(motion / "index.npy", index)
    elif problem in {"length", "shape", "beta"}:
        if problem == "length":
            records[0]["pose"] = torch.zeros(301, 66)
            records[0]["trans"] = torch.zeros(301, 3)
        elif problem == "shape":
            records[0]["trans"] = torch.zeros(60, 4)
        else:
            records[0]["beta"][0] = float("nan")
        torch.save(records, motion / "shard.pth")
    else:
        if problem == "embedding":
            embeddings[0]["embeddings"][0, 0] = float("nan")
        else:
            embeddings[0]["offsets"] = torch.tensor([0, 8])
        torch.save(embeddings, text / "shard.pth")
    with pytest.raises(MotionMillionError):
        make_dataset(release, monkeypatch)._load_data(0)


def test_real_base_processing_uses_F_then_pads_all_fields(release, monkeypatch):
    dataset = make_dataset(release, monkeypatch)
    seen = []

    class BodyStub:
        def __call__(self, body_pose, betas, global_orient, transl):
            return torch.zeros(len(body_pose), 24, 3)

        def get_skeleton(self, betas):
            return torch.zeros(24, 3)

    class CameraStub:
        def __call__(self, joints, frames, **kwargs):
            seen.append((len(joints), frames))
            out = torch.eye(4).repeat(frames, 1, 1)
            out[:, 2, 3] = 4.0
            return out

    import gem.datasets.pure_motion.base_dataset as base

    monkeypatch.setattr(base, "CameraAugmentorV11", CameraStub)
    dataset.smplx = dataset.smplx_lite = BodyStub()
    batch_items = [dataset[i] for i in range(len(LENGTHS))]
    assert seen == [(f, f) for f in LENGTHS]
    for frames, sample in zip(LENGTHS, batch_items):
        assert sample["length"] == sample["valid_length"] == frames
        assert sample["mask"]["valid"].tolist() == [True] * frames + [False] * (300 - frames)
        for name, value in sample["mask"].items():
            if torch.is_tensor(value):
                assert value.shape == (300,) and not value[frames:].any(), name
        for group in ("smpl_params_w", "smpl_params_c"):
            for value in sample[group].values():
                assert value.shape[0] == 300
        for key in ("R_c2gv", "K_fullimg", "T_w2c", "cam_angvel", "cam_tvel", "f_imgseq", "kp2d"):
            assert sample[key].shape[0] == 300, key
        assert sample["meta"]["T_w2c"].shape[0] == 300
        assert torch.allclose(torch.linalg.det(sample["R_c2gv"]), torch.ones(300), atol=1e-5)
    cfg = full_config()
    batch = collate_fn(batch_items, "train", cfg.data.collate_cfg)
    assert batch["text_embed"].shape == (6, 150, 1024)
    for key in ("music_embed", "music_array", "music_beats", "audio_array", "use_det_kp"):
        assert batch[key].shape[:2] == (6, 300), key
    assert batch["length"].tolist() == list(LENGTHS)


def tiny_denoiser(mode="valid_length"):
    torch.manual_seed(51)
    model = NetworkEncoderRoPE(
        output_dim=151,
        xt_dim=151,
        latent_dim=32,
        num_layers=2,
        num_heads=4,
        dropout=0,
        max_len=120,
        attention_mode=mode,
        encoded_text_dim=1024,
        text_mask_prob=0,
        text_encoder_cfg={"mode": "all", "cross_attn_type": "mha"},
    )
    for name, parameter in model.named_parameters():
        if "gate_" in name:
            parameter.data.fill_(0.3)
    return model.eval()


@pytest.mark.parametrize("frames", LENGTHS)
def test_attention_padding_invariance_and_backward(frames):
    model = tiny_denoiser()
    valid_noise = torch.randn(1, frames, 151)
    valid_cond = torch.randn(1, frames, 32)
    text = torch.randn(1, 150, 1024)
    text_mask = torch.arange(150)[None] < 11

    def forward(noise, cond):
        return model(
            noise,
            torch.tensor([37]),
            y={
                "f_cond": cond,
                "length": torch.tensor([frames]),
                "encoded_text": text,
                "text_attention_mask": text_mask,
            },
            inputs={},
            sample_indices_dict={"betas": (126, 136)},
        )

    # 显式复制同一有效噪声，不依赖不同shape的randn恰好相等。
    padded = torch.cat([valid_noise, torch.randn(1, 300 - frames, 151)], 1).requires_grad_()
    cond = torch.cat([valid_cond, torch.randn(1, 300 - frames, 32)], 1)
    short, long = forward(valid_noise, valid_cond), forward(padded, cond)
    changed = padded.detach().clone()
    changed[:, frames:] = 700 * torch.randn_like(changed[:, frames:])
    changed_cond = cond.clone()
    changed_cond[:, frames:] = 500 * torch.randn_like(changed_cond[:, frames:])
    changed_out = forward(changed, changed_cond)
    for key in ("pred_x", "pred_cam", "static_conf_logits"):
        assert torch.isfinite(long[key]).all()
        torch.testing.assert_close(short[key], long[key][:, :frames], atol=3e-6, rtol=2e-5)
        torch.testing.assert_close(
            long[key][:, :frames], changed_out[key][:, :frames], atol=3e-6, rtol=2e-5
        )
    long["pred_x"][:, :frames].square().mean().backward()
    assert torch.isfinite(padded.grad).all()
    assert not padded.grad[:, frames:].any()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.blocks[0].attn.rope.encoding.shape[0] == 4096


def test_attention_masks_match_legacy_window_and_empty_rows():
    for frames in LENGTHS:
        mask = valid_length_attention_mask(torch.tensor([frames]), 300, 120)[0]
        expected = torch.zeros(frames, frames, dtype=torch.bool)
        if frames > 120:
            expected.fill_(True)
            for i in range(frames):
                left = min(frames - 120, max(0, i - 60))
                right = max(120, min(frames, i + 60))
                expected[i, left:right] = False
        assert torch.equal(mask[:frames, :frames], expected)
        assert mask[frames:].all()
        if frames < 300:
            assert mask[:frames, frames:].all()
    attention = RoPEAttention(16, 4, dropout=0)
    x = torch.randn(1, 300, 16, requires_grad=True)
    mask = valid_length_attention_mask(torch.tensor([60]), 300, 120)
    output = attention(x, attn_mask=mask, key_padding_mask=torch.arange(300)[None] >= 60)
    assert not output[:, 60:].any()
    output.square().sum().backward()
    assert torch.isfinite(x.grad).all()


def test_legacy_attention_values_unchanged_on_nonempty_rows():
    model = tiny_denoiser("legacy")
    new = tiny_denoiser("valid_length")
    new.load_state_dict(model.state_dict(), strict=True)
    kwargs = dict(
        timesteps=torch.tensor([23]),
        y={
            "f_cond": torch.randn(1, 120, 32),
            "length": torch.tensor([120]),
            "encoded_text": torch.randn(1, 150, 1024),
        },
        inputs={},
        sample_indices_dict={"betas": (126, 136)},
    )
    x = torch.randn(1, 120, 151)
    torch.testing.assert_close(
        model(x, **kwargs)["pred_x"], new(x, **kwargs)["pred_x"], rtol=0, atol=0
    )


def test_effective_element_loss_and_difference_boundaries():
    torch.manual_seed(9)
    pred = torch.randn(2, 97, 4, 3, requires_grad=True)
    mask = torch.ones(2, 97, 4, 1, dtype=torch.bool)
    mask[1, 60:] = False
    mask[0, :, 1] = False
    weight = torch.tensor([0.5, 2.0])
    loss = masked_reduce(pred.square(), mask, strategy="valid_per_sample", sample_weights=weight)
    long = torch.cat([pred.detach(), torch.randn(2, 203, 4, 3)], 1).requires_grad_()
    longmask = torch.cat([mask, torch.zeros(2, 203, 4, 1, dtype=torch.bool)], 1)
    repeated = masked_reduce(
        long.square(), longmask, strategy="valid_per_sample", sample_weights=weight
    )
    torch.testing.assert_close(loss, repeated)
    repeated.backward()
    assert not long.grad[~longmask.expand_as(long)].any()
    empty = masked_reduce(pred.square(), torch.zeros_like(mask), strategy="valid_per_sample")
    assert empty.requires_grad and empty == 0
    empty.backward()
    assert torch.isfinite(pred.grad).all() and not pred.grad.any()
    valid = torch.arange(300)[None] < 97
    assert temporal_valid_mask(valid, pad_last=True).sum() == 96
    assert temporal_valid_mask(valid, order=2, pad_last=True).sum() == 95
    assert not temporal_valid_mask(valid, pad_last=True)[0, 96:].any()
    assert torch.equal(masked_reduce(pred.square(), mask), (pred.square() * mask).mean())


def test_configuration_and_checkpoint_contract():
    cfg = full_config()
    contract = validate_sequence_experiment(cfg)
    assert contract["pad_to_frames"] == 300 and contract["attention_max_len"] == 120
    assert (
        cfg.data.loader_opts.train.batch_size
        * cfg.pl_trainer.devices
        * cfg.pl_trainer.accumulate_grad_batches
        == 2048
    )
    assert cfg.callbacks.ckpt_saver.every10000s_top100.save_on_train_end
    assert cfg.pretrain_ckpt is cfg.resume_mode is cfg.ckpt_path is None
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        old_cfg = compose(config_name="train", overrides=["exp=gem_smpl_motionmillion_text_only"])
    for key in ("latent_dim", "num_layers", "num_heads", "output_dim"):
        assert cfg.network.model_cfg.denoiser[key] == old_cfg.network.model_cfg.denoiser[key]
    assert cfg.pipeline.args.weights == old_cfg.pipeline.args.weights
    assert cfg.optimizer == old_cfg.optimizer
    assert validate_sequence_experiment(old_cfg) is None
    small = full_config(["training_budget.max_steps=8", "training_budget.warmup_steps=2"])
    assert validate_sequence_experiment(small) == contract
    for override in (
        "pl_trainer.use_distributed_sampler=true",
        "training_budget.max_steps=20",
        "pl_trainer.max_epochs=1",
    ):
        with pytest.raises(ValueError):
            validate_sequence_experiment(full_config([override]))
    state = {
        "embed_text.weight": torch.zeros(1),
        "text_encoder_layers.x": torch.zeros(1),
        "gate_cross_attn": torch.zeros(1),
    }
    saved = {"state_dict": state.copy()}
    model = SimpleNamespace(
        sequence_contract=contract,
        text_condition_enabled=True,
        max_text_len=150,
        pipeline=SimpleNamespace(
            args=SimpleNamespace(in_attr=[]), denoiser3d=SimpleNamespace(encoded_text_dim=1024)
        ),
        ignored_weights_prefix=[],
        _sequence_data_identity=lambda: {"train": [{"build_fingerprint": "fixture"}]},
    )
    GEM.on_save_checkpoint(model, saved)
    parsed = validate_text_generation_payload(saved, "fixture.ckpt")
    assert parsed["sequence_contract"] == contract
    GEM.on_load_checkpoint(model, saved)
    assert model._resume_data_identity == saved["genmo_data_identity"]
    GEM.on_fit_start(model)
    model._resume_data_identity = {"train": [{"build_fingerprint": "changed"}]}
    with pytest.raises(ValueError, match="数据身份"):
        GEM.on_fit_start(model)
    for old in (None, {**contract, "sequence_mode": "crop", "pad_to_frames": 120}):
        with pytest.raises(ValueError, match="resume"):
            validate_resume_contract(old, contract)
    validate_resume_contract(contract, contract)
    validate_resume_contract(None, None)
    old_payload = validate_text_generation_payload({"state_dict": state}, "legacy.ckpt")
    assert old_payload["max_text_len"] == 50 and "sequence_contract" not in old_payload
    for frames in (60, 97, 120, 183, 240, 299, 300):
        validate_generation_length(contract, frames)
    for frames in (1, 59, 301):
        with pytest.raises(ValueError):
            validate_generation_length(contract, frames)
    validate_generation_length(None, 120)


def test_demo_loader_uses_checkpoint_sequence_contract_over_yaml(monkeypatch):
    """真实demo配置组装，CUDA模型实例化/权重读取替换为显式CPU桩。"""
    import hydra.utils

    import gem.utils.net_utils
    from scripts.demo.demo_utils import load_model

    contract = validate_sequence_experiment(full_config())
    contract["attention_max_len"] = 300  # 模拟之后独立消融的契约，不能被当前120 YAML覆盖。
    captured = []
    model = SimpleNamespace()
    model.cuda = lambda: model
    model.eval = lambda: model

    def instantiate(cfg, **kwargs):
        captured.append(OmegaConf.to_container(cfg, resolve=True))
        return model

    monkeypatch.setattr(hydra.utils, "instantiate", instantiate)
    monkeypatch.setattr(gem.utils.net_utils, "load_pretrained_model", lambda *args: None)
    assert (
        load_model(
            "explicit-fixture.ckpt",
            text_max_len_override=150,
            exp_name_override="gem_smpl_motionmillion_text_only",
            sequence_contract_override=contract,
        )
        is model
    )
    cfg = captured[0]
    assert cfg["model_cfg"]["sequence_contract"] == contract
    assert cfg["model_cfg"]["text_encoder"]["max_text_len"] == 150
    assert cfg["pipeline"]["args"]["loss_reduction"] == "valid_per_sample"
    network = cfg["pipeline"]["args_denoiser3d"]["model_cfg"]["denoiser"]
    assert network["max_len"] == 300 and network["attention_mode"] == "valid_length"


def test_eval_protocol_and_preflight_lengths(tmp_path):
    fixed, matched = length_protocol(), length_protocol("gt")
    assert prediction_length({"frames": 97}, fixed) == 120
    assert prediction_length({"frames": 97}, matched) == 97
    with pytest.raises(ValueError, match="诊断"):
        prediction_length({"frames": 240}, matched)
    stats = sequence_statistics(LENGTHS, "full", 300)
    assert stats["valid_frames"] == sum(LENGTHS) and stats["crop_count"] == 0
    assert stats["padding_frames"] == 1800 - sum(LENGTHS)
    with pytest.raises(MotionMillionError):
        sequence_statistics([120, 183], "full", 120)
    identity = dict(
        checkpoint_sha256="a",
        experiment_config_sha256="b",
        dataset_release_fingerprint="c",
        evaluator_fingerprint="d",
        ddim_steps=50,
        cfg_scale=2.5,
    )
    metrics = dict(
        fid=1.0,
        diversity=2.0,
        r_precision_1=0.5,
        r_precision_2=0.6,
        r_precision_3=0.7,
        matching_score=3.0,
    )
    paths = []
    for seed, protocol in enumerate((fixed, matched)):
        p = tmp_path / f"{seed}.json"
        p.write_text(json.dumps({**identity, **metrics, "seed": seed, "length_protocol": protocol}))
        paths.append(p)
    with pytest.raises(ValueError, match="长度协议"):
        summarize(paths, required_runs=2)
    with pytest.raises(ValueError, match="协议"):
        choose_candidate([{"length_protocol": fixed}, {"length_protocol": matched}], 0.01)


def test_full_preflight_reads_synthetic_shards_and_rejects_index_drift(release):
    """实际preflight读取三个临时split并校验SHA；仍是合成制品，不是服务器数据。"""
    from tools.data.motionmillion.preflight_motionmillion import _audit_release

    motion, text, records, embeddings = release
    motion_manifest = json.loads((motion / "manifests/train.json").read_text())
    text_manifest = json.loads((text / "manifests/train.json").read_text())
    (motion / "dataset_release.json").write_text(
        json.dumps({"schema_version": 1, "build_fingerprint": "f" * 64})
    )
    (text / "embedding_release.json").write_text(
        json.dumps({"schema_version": 1, "source_build_fingerprint": "f" * 64})
    )
    for split in ("val", "test"):
        motions, texts = copy.deepcopy(records), copy.deepcopy(embeddings)
        for row, embedding in zip(motions, texts):
            row["split"] = split
            row["motion_id"] = f"{split}/{row['motion_id']}"
            embedding["motion_id"] = row["motion_id"]
        torch.save(motions, motion / f"{split}.pth")
        torch.save(texts, text / f"{split}.pth")
        mm, tm = copy.deepcopy(motion_manifest), copy.deepcopy(text_manifest)
        mm["split"] = tm["split"] = split
        for root, manifest in ((motion, mm), (text, tm)):
            manifest["shards"][0].update(
                path=f"{split}.pth", sha256=sha256_file(root / f"{split}.pth")
            )
        tm["shards"][0].update(
            source_motion_path=f"{split}.pth", source_motion_sha256=mm["shards"][0]["sha256"]
        )
        (motion / f"manifests/{split}.json").write_text(json.dumps(mm))
        (text / f"manifests/{split}.json").write_text(json.dumps(tm))
    report = _audit_release(
        motion, text, verify_sha256=True, max_shards=1, sequence_mode="full", pad_to_frames=300
    )
    for split in ("train", "val", "test"):
        row = report["splits"][split]
        assert row["records_checked"] == 6
        assert row["sequence_statistics"]["valid_frames"] == sum(LENGTHS)
        assert row["sequence_statistics"]["crop_count"] == 0
    index = np.load(motion / "index.npy")
    index[0]["frames"] = 61
    np.save(motion / "index.npy", index)
    with pytest.raises(MotionMillionError, match="frames"):
        _audit_release(
            motion, text, verify_sha256=True, max_shards=1, sequence_mode="full", pad_to_frames=300
        )


def test_sampler_equal_batches_epochs_and_no_double_sharding():
    class Dataset:
        def sample_shard_ids(self):
            return np.repeat(np.arange(16), 65)

        def __len__(self):
            return 16 * 65

    dataset = Dataset()
    ranks = []
    for rank in range(8):
        sampler = ShardAwareDistributedSampler(dataset, rank=rank, num_replicas=8)
        first = list(sampler)
        assert len(first) // 64 == 2
        sampler.set_epoch(1)
        assert list(sampler) != first
        sampler.set_epoch(0)
        assert list(sampler) == first
        ranks.append(set(first))
    assert sum(len(r) for r in ranks) == len(set.union(*ranks))
    assert validate_sequence_experiment(full_config()) is not None


def test_real_lightning_loader_preparation_keeps_rank_sampler():
    """实际DataModule/Lightning装配代码，rank上下文为CPU替身；没有启动DDP。"""
    from pytorch_lightning.trainer.connectors.data_connector import _DataConnector
    from pytorch_lightning.trainer.states import RunningStage

    from gem.datamodule.mocap_trainX_testY import DataModule

    class Dataset:
        def sample_shard_ids(self):
            return np.repeat(np.arange(16), 65)

        def __len__(self):
            return 16 * 65

    cfg = full_config()
    trainer = SimpleNamespace(
        world_size=8,
        global_rank=3,
        accumulate_grad_batches=4,
        _accelerator_connector=SimpleNamespace(use_distributed_sampler=False, is_distributed=True),
    )
    dataset = Dataset()
    dm = SimpleNamespace(
        trainset=dataset,
        trainsets=[dataset],
        trainer=trainer,
        shard_aware_sampling=cfg.data.shard_aware_sampling,
        balanced_sampling=None,
        loader_opts=cfg.data.loader_opts,
        collate_cfg=cfg.data.collate_cfg,
    )
    loader = DataModule.train_dataloader(dm)
    assert loader.sampler.rank == 3 and loader.sampler.num_replicas == 8
    assert len(loader) == 2 and loader.drop_last
    connector = object.__new__(_DataConnector)
    connector.trainer = trainer
    assert connector._prepare_dataloader(loader, shuffle=True, mode=RunningStage.TRAINING) is loader
    trainer._accelerator_connector.use_distributed_sampler = True
    assert connector._requires_distributed_sampler(loader)


@pytest.mark.skipif(
    not (ROOT / "inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz").is_file(),
    reason="本机缺少真实SMPL-X neutral人体资产",
)
def test_cpu_real_smpl_small_network_complete_training_and_loss_invariance(release):
    """真实人体资产/相机/151D/扩散/全部原loss，网络缩为32D两层，仅作CPU功能验收。"""
    from hydra.utils import instantiate

    from gem.pipeline.gem_pipeline import (
        compute_extra_global_loss,
        compute_extra_incam_loss,
        slice_valid_inputs,
    )

    cfg = full_config(
        [
            "network.model_cfg.denoiser.latent_dim=32",
            "network.model_cfg.denoiser.num_layers=2",
            "network.model_cfg.denoiser.num_heads=4",
        ]
    )
    motion, text, *_ = release
    dataset = MotionMillionDataset(
        motion_manifest_path=motion / "manifests/train.json",
        embedding_manifest_path=text / "manifests/train.json",
        sequence_mode="full",
        pad_to_frames=300,
        caption_sampling="first",
    )
    batch = collate_fn([dataset[1], dataset[3]], "train", cfg.data.collate_cfg)
    model = instantiate(cfg.model, _recursive_=False).cpu().train()
    model.prepare_batch(batch, "diffusion")
    batch = model.create_condition_mask(
        batch, cfg.model.model_cfg.condition_mask, "diffusion", train=True
    )
    assert batch["length"].tolist() == [97, 183]
    for key in ("f_cond", "f_uncond", "f_empty"):
        assert batch[key].shape == (2, 300, 32)
    assert not batch["target_x_mask"][0, 96:, 148:151].any()
    assert batch["target_x_mask"][0, 96, :148].all()
    outputs = model.pipeline(batch, train=True, mode="diffusion")
    expected_losses = {
        "simple_loss",
        "cr_j3d_loss",
        "transl_c_loss",
        "j2d_loss",
        "cr_vert_loss",
        "verts2d_loss",
        "transl_w_loss",
        "static_conf_loss",
    }
    assert expected_losses.issubset(outputs)
    for key in expected_losses | {"loss"}:
        assert torch.isfinite(outputs[key]), key
    outputs["loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)

    # 固定同一组有效预测/目标，对真实辅助loss逐项比较F输入与pad300输入。
    def detached(value):
        if isinstance(value, dict):
            return {k: detached(v) for k, v in value.items()}
        return value.detach().clone() if torch.is_tensor(value) else value

    long_batch = slice_valid_inputs(detached(batch), 0, 2, 300, 300)
    short_batch = slice_valid_inputs(detached(batch), 0, 2, 300, 97)
    long_output = slice_valid_inputs(detached(outputs), 0, 2, 300, 300)
    short_output = slice_valid_inputs(detached(outputs), 0, 2, 300, 97)
    for function in (compute_extra_incam_loss, compute_extra_global_loss):
        _, long_losses = function(long_batch, long_output, model.pipeline, "diffusion")
        _, short_losses = function(short_batch, short_output, model.pipeline, "diffusion")
        for key in long_losses:
            torch.testing.assert_close(
                long_losses[key], short_losses[key], rtol=2e-5, atol=1e-5, msg=key
            )


def test_production_release_available_or_explicit_skip():
    """只在真实本地分片存在时读取一条；缺失时不冒充核验过服务器制品。"""
    motion = ROOT / "inputs/MotionMillion/genmo_smpl_v1/manifests/train.json"
    text = ROOT / "inputs/MotionMillion/t5_3b_v1_fp16/manifests/train.json"
    if not motion.is_file() or not text.is_file():
        pytest.skip("本机没有正式motion/T5分片；源码支持复用，实际制品待服务器授权预检")
    dataset = MotionMillionDataset(
        motion_manifest_path=motion,
        embedding_manifest_path=text,
        sequence_mode="full",
        pad_to_frames=300,
        caption_sampling="first",
    )
    sample = dataset._load_data(0)
    assert sample["source_frames"] == sample["valid_length"] and sample["crop_start"] == 0


def test_padded_validation_dispatches_only_real_frames_to_postprocessing():
    """只测试真实 Pipeline 的分派边界；内层生成/后处理用显式替身记录输入。"""
    from gem.pipeline.gem_pipeline import Pipeline

    seen = []

    def forward(sample, **kwargs):
        frames = int(sample["length"][0])
        assert sample["motion"].shape == (1, frames, 151)
        assert sample["text_embed"].shape == (1, 150, 1024)
        assert sample["mask"]["valid"].all()
        assert sample["smpl_params_w"]["transl"].shape == (1, frames, 3)
        assert sample["B"] == 1 and sample["L"] == frames
        assert kwargs["postproc"] is True
        seen.append(frames)
        return {"pred_body_params_global": {"transl": sample["smpl_params_w"]["transl"]}}

    lengths = torch.tensor([97, 183])
    inputs = {
        "motion": torch.zeros(2, 300, 151),
        "length": lengths,
        "text_embed": torch.ones(2, 150, 1024),
        "mask": {"valid": torch.arange(300)[None] < lengths[:, None]},
        "smpl_params_w": {"transl": torch.randn(2, 300, 3)},
    }
    model = SimpleNamespace(args={"loss_reduction": "valid_per_sample"}, forward=forward)
    result = Pipeline.forward(model, inputs, train=False, postproc=True)
    assert seen == [97, 183]
    for i, frames in enumerate(seen):
        output = result["pred_body_params_global"]["transl"][i]
        assert output.shape == (300, 3)
        torch.testing.assert_close(output[:frames], inputs["smpl_params_w"]["transl"][i, :frames])
        torch.testing.assert_close(output[frames:], output[frames - 1].expand(300 - frames, -1))
    assert inputs["motion"].shape == (2, 300, 151)


def test_evaluation_gt_length_loop_and_report_identity(tmp_path, monkeypatch):
    """真实生成控制流，推理器和SMPL→272转换是CPU替身，不代表模型或evaluator验收。"""
    from tools.eval import generate_motionmillion_val_predictions as generate
    from tools.eval.run_motionmillion_official_metrics import run_metrics

    checkpoint = tmp_path / "fixture.ckpt"
    checkpoint.write_bytes(b"explicit fake checkpoint")
    t5 = tmp_path / "t5"
    t5.mkdir()
    (t5 / "fixture.bin").write_bytes(b"explicit fake T5")
    t5_release = tmp_path / "text_release.json"
    t5_release.write_text(
        json.dumps(
            {
                "resolved_revision": "fixture",
                "model_files": [{"path": "fixture.bin", "sha256": sha256_file(t5 / "fixture.bin")}],
            }
        )
    )
    eligibility = tmp_path / "eligibility.json"
    rows = [
        {"motion_id": f"fixture/{i}", "frames": f, "captions": ["walk", "walking"]}
        for i, f in enumerate((97, 183))
    ]
    eligibility.write_text(json.dumps({"dataset_root": str(tmp_path), "records": rows}))
    requests = []

    class EngineStub:
        def __init__(self, **kwargs):
            assert kwargs["max_frames"] == 183

        def initialize(self):
            pass

        def close(self):
            pass

        def generate(self, request):
            requests.append(request)
            path = tmp_path / f"{request.num_frames}.npz"
            np.savez(path, body_pose=np.zeros((request.num_frames, 63)))
            return {"ok": True, "motion_npz": str(path)}

    def export_stub(args, **kwargs):
        frames = len(np.load(args.input)["body_pose"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, np.zeros((frames, 272), np.float32))
        return {
            "frames": frames,
            "output": str(args.output),
            "output_sha256": sha256_file(args.output),
        }

    monkeypatch.setattr(generate, "ResidentTextMotionEngine", EngineStub)
    monkeypatch.setattr(generate, "EnDecoder", lambda **kwargs: SimpleNamespace(eval=lambda: None))
    monkeypatch.setattr(generate, "export_motion", export_stub)
    args = generate.build_parser().parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--eligibility",
            str(eligibility),
            "--t5-model",
            str(t5),
            "--t5-release",
            str(t5_release),
            "--output-root",
            str(tmp_path / "predictions"),
            "--seed",
            "42",
            "--length-mode",
            "gt",
        ]
    )
    report = generate.generate_predictions(args)
    assert [request.num_frames for request in requests] == [97, 183]
    assert [row["length"] for row in report["records"]] == [97, 183]
    assert report["identity"]["length_protocol"] == length_protocol("gt")
    for source, request in zip(rows, requests):
        index, caption, seed = generate.select_caption_and_seed(
            source["motion_id"], source["captions"], 42
        )
        assert (request.metadata["text_index"], request.prompt, request.seed) == (
            index,
            caption,
            seed,
        )
    args.resume = True
    generate.generate_predictions(args)
    assert len(requests) == 2
    args.length_mode = "fixed"
    with pytest.raises(ValueError, match="身份不一致"):
        generate.generate_predictions(args)

    # 评分前逐条检查长度；即使样本随后会被drop_last，也不能绕过协议检查。
    evaluator_identity = tmp_path / "evaluator.json"
    evaluator_identity.write_text(
        json.dumps({"status": "PASS", "evaluator_fingerprint": "fixture", "files": []})
    )
    predictions = Path(report["prediction_manifest"])
    bad_rows = copy.deepcopy(report["records"])
    bad_rows[-1]["length"] = 120
    predictions.write_text("\n".join(json.dumps(row) for row in bad_rows))
    with pytest.raises(ValueError, match="长度与协议不一致"):
        run_metrics(
            SimpleNamespace(
                length_mode="gt",
                num_frames=120,
                evaluator_identity=evaluator_identity,
                eligibility=eligibility,
                predictions=predictions,
                checkpoint=checkpoint,
                seed=42,
                ddim_steps=50,
                cfg_scale=2.5,
                batch_size=32,
            )
        )
