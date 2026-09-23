"""Stage 1 持久训练器的 CPU 逻辑集成回归，不替代服务器真实训练验收。

本文件复用原第 3 步合成数据夹具、真实 BumiEndecoder/FK、缩小的 GENMO Transformer 和
Stage1Actor，在系统临时测试目录内检查训练入口实际保存和加载 optimizer/scheduler/RNG。
不中断训练与中途恢复的最终权重、优化器及下一抽样位置必须逐元素一致；验证过程不得
消耗训练 RNG。另以两进程 CPU/Gloo 实际执行 DDP.forward、梯度累计 no_sync 和 rank0
写入，检查模型参数同步及各 rank 的采样流不同。小网络、合成数据和 CPU 结果只证明
执行器逻辑，不代表用户要求的完整模型、真实四库数据、单卡/八卡 CUDA 短程验证已完成。
通过验证报告落盘后注入中断，核验从同一 checkpoint 再次恢复会分配新的 attempt 名称，
并保留旧配置、事件日志和验证产物；存在更晚完整 checkpoint 时仍必须拒绝回退。

所有 stats、清单、checkpoint、TensorBoard 和子进程日志均在 pytest 的 tmp_path 内创建；
调用测试时必须使用显式系统临时 basetemp，并在命令结束后的 finally 清理该精确目录。
测试不读取服务器、不安装依赖、不写正式实验目录、不创建后台任务。
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from gem.closedloop import runner
from gem.closedloop.training import (
    build_stage1_losses,
    load_stage1_config,
)
from tests.closedloop.test_stage1_actor import actor_factory as actor_factory
from tests.closedloop.test_stage1_dataset import _write_dataset
from tests.closedloop.test_stage1_training import _data_config


def _configuration(tmp_path, actor_factory, *, max_steps=4, workers=2):
    reference, _ = actor_factory(history_steps=4, starts=(45,))
    train_root = _write_dataset(tmp_path / "runner_train_data", length=130)
    val_root = _write_dataset(tmp_path / "runner_val_data", length=130, split="val")
    data = _data_config(train_root, reference.endecoder.stats_path)
    data.datasets.train.synthetic.random_decision = True
    data.datasets.val = {"synthetic": OmegaConf.to_container(data.datasets.train.synthetic)}
    data.datasets.val.synthetic.root = str(val_root)
    data.datasets.val.synthetic.split = "val"
    data_path = tmp_path / "runner_data.yaml"
    OmegaConf.save(data, data_path)
    config = load_stage1_config(
        "configs/closedloop/stage1_train.yaml",
        [
            "model.latent_dim=32", "model.num_layers=1", "model.num_heads=4",
            "model.history_hidden_dim=16", "model.dropout=0.1", "model.music_mask_prob=0.1",
            "endecoder.allow_placeholder_stats=true", "runtime.device=cpu",
            "runtime.cpu_threads=1", "sample_contract.history_steps=4",
            "data_loader.batch_size=1", "data_loader.samples_per_epoch=4",
            f"data_loader.num_workers={workers}", f"train.max_steps={max_steps}",
            "trainer.gradient_accumulation=2", "trainer.precision=fp32",
            "trainer.require_warm_start=false", "trainer.log_every_steps=1",
            "trainer.validate_every_steps=1", "trainer.save_every_steps=2",
            "scheduler.total_steps=20", "scheduler.warmup_steps=2", "scheduler.min_lr_ratio=0.1",
            "validation.samples_per_source=1", "validation.seed=341", "validation.sample_steps=2",
        ],
    )
    config.dataset_config = str(data_path)
    return config, data


def _assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            _assert_tree_equal(a, b)
    else:
        assert left == right


def test_persistent_resume_matches_uninterrupted_across_epoch(tmp_path, actor_factory):
    config, _ = _configuration(tmp_path, actor_factory)
    full = runner.run_persistent(config, tmp_path / "uninterrupted")
    partial_config = copy.deepcopy(config)
    partial_config.train.max_steps = 2
    first = runner.run_persistent(partial_config, tmp_path / "resumed")
    resumed_config = copy.deepcopy(config)
    resumed_config.resume_checkpoint = first["last_checkpoint"]
    resumed = runner.run_persistent(resumed_config, tmp_path / "resumed")
    assert full["global_step"] == resumed["global_step"] == 4
    assert resumed["executed_optimizer_steps"] == 2
    assert resumed["resume"]["rng_restored"] is True
    assert resumed["resume"]["optimizer_restored"] is True
    assert resumed["resume"]["scheduler_restored"] is True
    assert resumed["resume"]["sampler_epoch"] == 1
    assert resumed["resume"]["sampler_offset"] == 0
    expected = torch.load(full["last_checkpoint"], map_location="cpu", weights_only=False)
    actual = torch.load(resumed["last_checkpoint"], map_location="cpu", weights_only=False)
    for key in ("state_dict", "optimizer_state_dict", "scheduler", "runtime_state"):
        _assert_tree_equal(expected[key], actual[key])
    expected_rows = [json.loads(row) for row in (tmp_path / "uninterrupted/train_metrics.jsonl").read_text().splitlines()]
    actual_rows = [json.loads(row) for row in (tmp_path / "resumed/train_metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in actual_rows] == [1, 2, 3, 4]
    for expected_row, actual_row in zip(expected_rows, actual_rows):
        for key in ("loss", "gradient_norm", "learning_rate"):
            assert expected_row[key] == actual_row[key]
        assert expected_row["ranks"][0]["sample_signature"] == actual_row["ranks"][0]["sample_signature"]


def test_monitor_preserves_training_rng_and_accepts_conditions_only(tmp_path, actor_factory):
    config, data = _configuration(tmp_path, actor_factory, workers=0)
    actor, _ = actor_factory(history_steps=4, starts=(45,))
    losses = build_stage1_losses(actor, config)
    loaders = runner.build_monitor_loaders(data, config.validation)
    actor.train()
    before = runner.capture_rng(torch.device("cpu"))
    report = runner.monitor(actor, losses, loaders, config.validation, torch.device("cpu"), step=0)
    after = runner.capture_rng(torch.device("cpu"))
    _assert_tree_equal(before, after)
    assert actor.training and losses.training
    assert report["sampling_received_target_fields"] is False
    assert report["completed_samples"] == 1
    sample = report["sources"]["synthetic"][0]
    assert sample["known_max_abs_error"] == sample["trace_known_max_abs_error"] == 0


def test_real_two_process_cpu_ddp_accumulation_and_rank_writes(tmp_path, actor_factory):
    config, _ = _configuration(tmp_path, actor_factory, max_steps=2, workers=0)
    path = tmp_path / "runner_cpu_ddp.yaml"
    OmegaConf.save(config, path)
    output = tmp_path / "cpu_ddp"
    code = (
        "import sys; from omegaconf import OmegaConf; "
        "from gem.closedloop.runner import run_persistent; "
        "run_persistent(OmegaConf.load(sys.argv[1]), sys.argv[2])"
    )
    command = [
        sys.executable, "-B", "-m", "torch.distributed.run", "--standalone",
        "--nnodes=1", "--nproc-per-node=2", "--no-python",
        sys.executable, "-B", "-c", code, str(path), str(output),
    ]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1"}
    process = subprocess.run(
        command, cwd=Path(__file__).resolve().parents[2], env=env,
        capture_output=True, text=True, timeout=120,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads((output / "reports/from_s000000_to_s000002.json").read_text())
    assert report["world_size"] == 2 and report["rank_weights_identical"] is True
    assert report["effective_global_batch"] == 4
    rows = [json.loads(row) for row in (output / "train_metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [1, 2]
    assert all(len(row["ranks"]) == 2 for row in rows)
    assert any(len({rank["sample_signature"] for rank in row["ranks"]}) == 2 for row in rows)
    assert len(list((output / "checkpoints").iterdir())) == 1
    assert len(list((output / "tensorboard").glob("events.*"))) == 1


def test_interrupted_resume_retries_without_overwriting_previous_attempt(
    tmp_path, actor_factory, monkeypatch
):
    config, _ = _configuration(tmp_path, actor_factory, workers=0)
    uninterrupted = runner.run_persistent(config, tmp_path / "reference")
    initial_config = copy.deepcopy(config)
    initial_config.train.max_steps = 2
    output = tmp_path / "retry"
    initial = runner.run_persistent(initial_config, output)
    resume_config = copy.deepcopy(config)
    resume_config.resume_checkpoint = initial["last_checkpoint"]
    base_session = "from_s000002_to_s000004"
    original_write = runner.write_json

    def fail_after_validation(path, value):
        original_write(path, value)
        if Path(path) == output / "validation" / base_session / "s000003.json":
            raise RuntimeError("simulated interruption before the next checkpoint")

    with monkeypatch.context() as patch:
        patch.setattr(runner, "write_json", fail_after_validation)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            runner.run_persistent(resume_config, output)
    assert not (output / "checkpoints/s000004.pt").exists()
    assert (output / "validation" / base_session / "s000003.json").exists()
    previous_files = {
        path.relative_to(output): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in output.rglob("*")
        if path.is_file() and path.name not in {"train_metrics.jsonl", "latest.json"}
    }
    previous_log = (output / "train_metrics.jsonl").read_bytes()
    resumed = runner.run_persistent(resume_config, output)
    session = f"{base_session}_attempt002"
    assert resumed["session"] == resumed["resume"]["session"] == session
    assert resumed["executed_optimizer_steps"] == 2
    assert (output / f"config_{session}.yaml").exists()
    assert (output / "reports" / f"resume_{session}.json").exists()
    assert (output / "reports" / f"{session}.json").exists()
    assert (output / "validation" / session / "s000003.json").exists()
    for relative, digest in previous_files.items():
        assert hashlib.sha256((output / relative).read_bytes()).hexdigest() == digest
    assert (output / "train_metrics.jsonl").read_bytes().startswith(previous_log)
    rows = [json.loads(row) for row in (output / "train_metrics.jsonl").read_text().splitlines()]
    assert [row["session"] for row in rows if row["step"] == 3] == [base_session, session]
    expected = torch.load(uninterrupted["last_checkpoint"], map_location="cpu", weights_only=False)
    actual = torch.load(resumed["last_checkpoint"], map_location="cpu", weights_only=False)
    for key in ("state_dict", "optimizer_state_dict", "scheduler", "runtime_state"):
        _assert_tree_equal(expected[key], actual[key])


def test_resume_session_names_advance_and_later_checkpoint_still_blocks_rollback(tmp_path):
    session = "from_s000010_to_s000015"
    contract = {"test_contract": 1}
    (tmp_path / "run_contract.json").write_text(json.dumps(contract))
    (tmp_path / f"config_{session}.yaml").write_text("previous: true\n")
    (tmp_path / "validation" / f"{session}_attempt002").mkdir(parents=True)
    assert runner.check_output_session(tmp_path, session, 10, True, contract) == f"{session}_attempt003"
    with pytest.raises(FileExistsError, match="already exists"):
        runner.check_output_session(tmp_path, session, 10, False, contract)
    with pytest.raises(ValueError, match="different training contract"):
        runner.check_output_session(tmp_path, session, 10, True, {"test_contract": 2})
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints/s000011.pt").write_bytes(b"complete checkpoint sentinel")
    with pytest.raises(ValueError, match="roll back newer complete checkpoints"):
        runner.check_output_session(tmp_path, session, 10, True, contract)
