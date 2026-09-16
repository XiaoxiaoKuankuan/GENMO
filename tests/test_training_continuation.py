"""续训恢复与固定质量监控的 CPU 回归测试。

用真实 Lightning 小模型证明完整 checkpoint 恢复保留 AdamW 动量、全局步数，
新阶段 LR 不被旧调度器覆盖，阶段内再次恢复保持相同轨迹，并在非保存周期末尾
补存最终 checkpoint。其余测试覆盖固定 RNG 隔离、近地脚滑代理、配对改善与
退化早停、原子交付失败保留旧文件、Hydra 对既有训练契约的继承。
测试仅使用 pytest 的系统临时目录，不加载 SMPL/T5、不执行 GPU 训练。
"""

import copy
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from gem.callbacks.motionmillion_quality import (
    atomic_artifact,
    isolated_rng,
    motion_proxies,
    quality_decision,
)
from gem.callbacks.simple_ckpt_saver import SimpleCkptSaver
from gem.network.gem_diffusion import GEMDiffusion
from gem.utils.lr_scheduler import ContinuationWarmupCosineLR, LinearWarmupCosineAnnealingLR


def test_stage_lr_bootstrap_and_second_resume_preserve_moments():
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([parameter], lr=2e-4)
    (parameter.square().sum()).backward()
    optimizer.step()
    optimizer.param_groups[0]["lr"] = 2.2768240790631672e-6
    expected = copy.deepcopy(optimizer.state[parameter])
    scheduler = ContinuationWarmupCosineLR(
        optimizer,
        start_step=210000,
        stage_steps=20000,
        warmup_steps=1000,
        peak_lr=1e-5,
        min_lr=2e-6,
    )
    scheduler.load_state_dict({"last_epoch": 210000, "total_steps": 215000})
    assert scheduler.get_last_lr()[0] == pytest.approx(2.2768240790631672e-6)
    for key in ("exp_avg", "exp_avg_sq", "step"):
        torch.testing.assert_close(optimizer.state[parameter][key], expected[key], rtol=0, atol=0)
    for step, rate in [
        (210500, (2.2768240790631672e-6 + 1e-5) / 2),
        (211000, 1e-5),
        (230000, 2e-6),
        (240000, 2e-6),
    ]:
        scheduler.last_epoch = step
        assert scheduler.get_lr()[0] == pytest.approx(rate)
    scheduler.last_epoch = 215000
    saved = copy.deepcopy(scheduler.state_dict())
    restored = ContinuationWarmupCosineLR(
        optimizer,
        start_step=210000,
        stage_steps=20000,
        warmup_steps=1000,
        peak_lr=1e-5,
        min_lr=2e-6,
    )
    restored.load_state_dict(saved)
    assert restored.last_epoch == 215000
    assert restored.get_last_lr() == pytest.approx(scheduler.get_lr())
    with pytest.raises(ValueError, match="阶段"):
        restored.load_state_dict({**saved, "stage_signature": {}})


class TinyModel(pl.LightningModule):
    def __init__(self, continuation=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))
        self.continuation = continuation

    def training_step(self, batch, batch_idx):
        return ((self.weight * batch[0] - 0.2) ** 2).mean()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = (
            ContinuationWarmupCosineLR(
                optimizer, start_step=5, stage_steps=6, warmup_steps=2, peak_lr=1e-3, min_lr=1e-6
            )
            if self.continuation
            else LinearWarmupCosineAnnealingLR(
                optimizer, warmup_steps=2, total_steps=10, min_lr=1e-6
            )
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


def test_lightning_full_restore_and_nonperiodic_final_save(tmp_path):
    kwargs = dict(
        accelerator="cpu",
        devices=1,
        max_epochs=-1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    data = DataLoader(TensorDataset(torch.ones(40, 2)), batch_size=2)
    original = pl.Trainer(max_steps=5, enable_checkpointing=False, **kwargs)
    original.fit(TinyModel(), data)
    source = tmp_path / "source.ckpt"
    original.save_checkpoint(source)
    checkpoint = torch.load(source, weights_only=False)

    class AssertRestored(pl.Callback):
        def on_train_start(self, trainer, module):
            assert trainer.global_step == 5
            state = next(iter(trainer.optimizers[0].state.values()))
            expected = next(iter(checkpoint["optimizer_states"][0]["state"].values()))
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(state[key], expected[key], rtol=0, atol=0)
            assert (
                trainer.optimizers[0].param_groups[0]["lr"]
                == checkpoint["optimizer_states"][0]["param_groups"][0]["lr"]
            )

    saver = SimpleCkptSaver(tmp_path / "stage", every_n_steps=4, save_on_train_end=True)
    continued = pl.Trainer(max_steps=9, callbacks=[saver, AssertRestored()], **kwargs)
    continued.fit(TinyModel(True), data, ckpt_path=source)
    final = torch.load(tmp_path / "stage/last.ckpt", weights_only=False)
    assert final["global_step"] == 9
    assert (tmp_path / "stage/s000008.ckpt").is_file()
    assert (tmp_path / "stage/s000009.ckpt").is_file()
    assert final["lr_schedulers"][0]["stage_signature"]["start_step"] == 5
    again = pl.Trainer(max_steps=11, enable_checkpointing=False, **kwargs)
    again.fit(TinyModel(True), data, ckpt_path=tmp_path / "stage/last.ckpt")
    assert again.global_step == 11
    assert again.optimizers[0].param_groups[0]["lr"] == pytest.approx(1e-6)


def test_isolated_rng_is_deterministic_and_does_not_advance_training():
    def draw():
        return random.random(), np.random.rand(), torch.rand(1).item()

    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    expected = draw()
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    with isolated_rng(42, torch.device("cpu")):
        first = draw()
    with isolated_rng(42, torch.device("cpu")):
        assert draw() == first
    assert draw() == expected


def test_slide_proxy_measures_horizontal_contact_motion():
    joints = np.zeros((120, 22, 3))
    assert motion_proxies(joints)["slide_sum"] == 0
    joints[:, :, 0] = np.arange(120)[:, None] * 0.01
    proxy = motion_proxies(joints)
    assert proxy["slide_sum"] / proxy["contact_count"] == pytest.approx(0.3)
    joints[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="有限"):
        motion_proxies(joints)


def test_validation_loss_uses_train_noise_without_enabling_dropout():
    diffusion = GEMDiffusion.__new__(GEMDiffusion)
    torch.nn.Module.__init__(diffusion)
    diffusion.train_diffusion = SimpleNamespace(
        original_num_steps=1000, _scale_timesteps=lambda t: t
    )
    diffusion.regression_input_type = "zero"
    diffusion.args = SimpleNamespace(out_attr=[])

    class Denoiser(torch.nn.Module):
        def forward(self, x, t, **kwargs):
            assert not self.training
            assert t.item() == 999
            return {"pred_x": x}

    diffusion.denoiser = Denoiser()
    diffusion.eval()
    inputs = {
        "length": torch.tensor([3]),
        "motion": torch.zeros(1, 3, 2),
        "f_cond": torch.zeros(1, 3, 2),
        "f_empty": torch.zeros(1, 3, 2),
        "mask": {"valid": torch.ones(1, 3, dtype=torch.bool)},
        "sample_indices_dict": {},
    }
    with pytest.raises(AssertionError, match="training"):
        diffusion.forward_train(inputs, "regression")
    inputs["_validation_loss"] = True
    with pytest.raises(AssertionError, match="no_grad"):
        diffusion.forward_train(inputs, "regression")
    with torch.no_grad():
        assert diffusion.forward_train(inputs, "regression")["pred_x"].shape == (1, 3, 2)
    assert diffusion.training is False and diffusion.denoiser.training is False


def test_quality_requires_reliable_improvement_and_no_collapse():
    cfg = dict(
        patience=3,
        min_relative_improvement=0.002,
        min_diversity_ratio=0.9,
        max_slide_ratio=1.1,
        slide_tolerance=0.005,
    )
    baseline = dict(
        step=210000,
        loss=12.0,
        per_motion_loss=[12.0] * 32,
        diversity_m=0.1,
        foot_slide_proxy_mps=0.1,
        contact_fraction=0.2,
    )
    baseline.update(quality_decision([], baseline, **cfg))
    improved = {**baseline, "step": 215000, "loss": 11.8, "per_motion_loss": [11.8] * 32}
    assert quality_decision([baseline], improved, **cfg)["accepted"]
    history = [baseline]
    for i in range(3):
        collapsed = {**improved, "step": 215000 + i * 5000, "diversity_m": 0.01}
        decision = quality_decision(history, collapsed, **cfg)
        assert not decision["accepted"]
        history.append({**collapsed, **decision})
    assert history[-1]["stop"]


def test_atomic_failed_write_keeps_previous_and_cleans_temp(tmp_path, monkeypatch):
    target = tmp_path / "latest.npz"
    atomic_artifact(target, {"motion": np.ones(4)}, npz=True)
    previous = target.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("disk failure")

    monkeypatch.setattr(np, "savez_compressed", fail)
    with pytest.raises(OSError):
        atomic_artifact(target, {"motion": np.zeros(4)}, npz=True)
    assert target.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [target]


def test_continuation_hydra_preserves_training_contract():
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    with initialize_config_dir(
        version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "configs")
    ):
        cfg = compose(
            config_name="train",
            overrides=["exp=gem_smpl_motionmillion_continue", "resume_mode=/tmp/source.ckpt"],
        )
    assert cfg.pl_trainer.max_steps == 230000
    assert cfg.pl_trainer.max_epochs == -1
    assert cfg.pl_trainer.devices == 8 and cfg.data.loader_opts.train.batch_size == 256
    assert cfg.pl_trainer.precision == "bf16-mixed" and cfg.pl_trainer.gradient_clip_val == 0.5
    assert cfg.pl_trainer.use_distributed_sampler is False
    assert cfg.network.model_cfg.denoiser.text_mask_prob == 0.1
    assert cfg.pipeline.args.weights.cr_j3d == 500
    assert cfg.scheduler.scheduler.total_steps is None
    assert cfg.callbacks.quality_monitor.cohort_size == 32
    assert cfg.callbacks.ckpt_saver.every10000s_top100.save_on_train_end
