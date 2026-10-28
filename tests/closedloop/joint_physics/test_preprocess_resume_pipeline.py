"""受控预处理恢复流水线的CPU安全回归，不发送真实非零信号或启动任何作业。

使用临时/proc身份与小型完整checkpoint验证PID复用拒绝、完整断点前禁止停止、
checkpoint缺rank或含NaN拒绝、benchmark失败后不启动预处理/恢复，以及精确清理
只作用于已核验benchmark分片。停止任务接口全部替换；pidfd ABI回退仅对测试进程自身
执行signal0能力检查，不递送终止信号，不操作现有训练任务。
"""

import copy
import errno
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import preprocess_then_resume_joint_physics as pipeline


def checkpoint_value(step=5000):
    import torch
    rng = torch.ones(10, dtype=torch.uint8)
    sampler = {"source_rng": {}, "time_rng": {}, "draws": 1, "order": [0], "cursor": 0, "identity": "source"}
    states = [{"data": {"rank": rank, "world_size": 8, "rng": {}, "draws": 1, "identity": "catalog"},
        "noise": rng, "window": {}, "physics_data": copy.deepcopy(sampler), "future_data": copy.deepcopy(sampler),
        "physics_noise": rng, "future_noise": rng,
        "rng": {"python": (), "numpy": (), "cpu": rng, "cuda_current": rng}} for rank in range(8)]
    return {"schema": pipeline.CHECKPOINT_SCHEMA, "global_step": step, "actor": {"weight": torch.ones(2)},
        "optimizer": {"state": {0: {"exp_avg": torch.zeros(2)}}, "param_groups": [{"lr": 1e-5}]},
        "scheduler": {"last_epoch": step}, "curriculum": {"step": step}, "rank_states": states,
        "contract": {"assets": {"sm_wm_sha256": "asset"}}, "counters": {},
        "config": {"training": {"max_steps": 200000}}, "sidecar_snapshot": [{"index": {}}],
        "initialization": {"mode": "random"}}


class TestPreprocessResumePipeline(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="joint-pipeline-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    @unittest.skipUnless(pipeline.sys.platform == "linux" and pipeline.platform.machine() == "x86_64", "仅已核验ABI")
    def test_syscall_fallback_opens_self_and_checks_signal_zero_only(self):
        with patch.object(pipeline.os, "pidfd_open", None, create=True), \
             patch.object(pipeline.signal, "pidfd_send_signal", None, create=True), \
             patch.object(pipeline.os, "kill") as naked_kill:
            result = pipeline.pidfd_preflight()
            self.assertEqual(result["own_pid"], os.getpid())
            self.assertEqual(result["signal_number"], 0)
            self.assertEqual(result["pidfd_open_backend"], "linux_x86_64_syscall434")
            self.assertEqual(result["pidfd_send_backend"], "linux_x86_64_syscall424")
            self.assertTrue(result["pidfd_closed"])
            naked_kill.assert_not_called()

    @unittest.skipUnless(pipeline.sys.platform == "linux" and pipeline.platform.machine() == "x86_64", "仅已核验ABI")
    def test_pidfd_fallback_propagates_errno_and_never_uses_kill(self):
        with patch.object(pipeline.os, "pidfd_open", None, create=True), \
             patch.object(pipeline.signal, "pidfd_send_signal", None, create=True), \
             patch.object(pipeline.os, "kill") as naked_kill:
            with self.assertRaises(OSError) as opened:
                pipeline.safe_pidfd_open(-1)
            self.assertEqual(opened.exception.errno, errno.EINVAL)
            with self.assertRaises(OSError) as sent:
                pipeline.safe_pidfd_send_signal(-1, 0)
            self.assertEqual(sent.exception.errno, errno.EBADF)
            with patch.object(pipeline.platform, "machine", return_value="aarch64"):
                with self.assertRaises(RuntimeError):
                    pipeline.safe_pidfd_open(os.getpid())
            naked_kill.assert_not_called()

    def test_proc_full_identity_detects_pid_reuse_and_argv_change(self):
        directory = self.root/"42"
        directory.mkdir()
        fields = ["R", "1"]+["0"]*18
        fields[19] = "987"
        (directory/"stat").write_text("42 (python worker) "+" ".join(fields))
        (directory/"cmdline").write_bytes(b"python\0--output\0/run\0")
        (directory/"cwd").symlink_to(self.root)
        identity = pipeline.process_identity(42, self.root)
        self.assertEqual(identity["starttime"], 987)
        reader = lambda pid: pipeline.process_identity(pid, self.root)
        self.assertTrue(pipeline.require_same_process(identity, reader=reader))
        fields[19] = "988"
        (directory/"stat").write_text("42 (python worker) "+" ".join(fields))
        with self.assertRaises(RuntimeError):
            pipeline.require_same_process(identity, reader=reader)
        fields[19] = "987"
        (directory/"stat").write_text("42 (python worker) "+" ".join(fields))
        (directory/"cmdline").write_bytes(b"python\0--output\0/another-run\0")
        with self.assertRaises(RuntimeError):
            pipeline.require_same_process(identity, reader=reader)

    def test_missing_complete_checkpoint_never_signals(self):
        with patch.object(pipeline, "signal_exact_pidfd") as sender:
            with self.assertRaises(FileNotFoundError):
                pipeline.terminate_verified({"pid": 42}, self.root/"missing-proof.json")
            sender.assert_not_called()

    def test_finite_checkpoint_proof_and_reject_missing_rank_nan(self):
        import torch
        value = checkpoint_value()
        path, proof = self.root/"joint_00005000.pt", self.root/"checkpoint_verified.json"
        torch.save(value, path)
        result = pipeline.verify_checkpoint(path, 5000, proof)
        self.assertTrue(result["all_floating_tensors_finite"])
        self.assertEqual(pipeline.require_checkpoint_proof(proof)["global_step"], 5000)
        bad = copy.deepcopy(value)
        bad["rank_states"].pop()
        with self.assertRaises(ValueError):
            pipeline.validate_checkpoint_payload(bad, 5000)
        bad = copy.deepcopy(value)
        bad["actor"]["weight"][0] = float("nan")
        with self.assertRaises(ValueError):
            pipeline.validate_checkpoint_payload(bad, 5000)
        with patch.object(pipeline, "process_identity", return_value={"pid": 42, "starttime": 2}), \
             patch.object(pipeline, "signal_exact_pidfd") as sender:
            with self.assertRaises(RuntimeError):
                pipeline.terminate_verified({"pid": 42, "starttime": 1}, proof)
            sender.assert_not_called()

    def test_benchmark_failure_prevents_preprocessing_and_resume(self):
        run, sidecar = self.root/"run", self.root/"sidecar"
        run.mkdir(); sidecar.mkdir()
        (run/"config.json").write_text(json.dumps({"data": {"sidecar": ["old1", "old2", str(sidecar)]},
                                                "prepare": {"exclude_sidecars": ["old1", "old2"]},
                                                "training": {"max_steps": 200000}}))
        (run/"joint_00005000.pt").write_bytes(b"mock checkpoint verified by patched verifier")
        args = SimpleNamespace(run=run, sidecar=sidecar, repo=self.root, torchrun_pid=42, collector_pid=None,
            checkpoint_step=5000, max_steps=200000, exit_timeout=1, gpu_idle_timeout=1, gpus=list(map(str, range(8))),
            python="python", poll_seconds=.01, expected_sources=4287, resume_output=self.root/"resume")
        journal = pipeline.Journal(self.root)
        with patch.object(pipeline, "capture_task", return_value={"pid": 42}), \
             patch.object(pipeline, "capture_workers", return_value=[{"pid": i} for i in range(8)]), \
             patch.object(pipeline, "require_same_process", return_value=True), \
             patch.object(pipeline, "verify_checkpoint", return_value={"training_max_steps": 200000}), \
             patch.object(pipeline, "terminate_verified", return_value="sigterm_sent"), \
             patch.object(pipeline, "wait_exited"), patch.object(pipeline, "require_idle_gpus"), \
             patch.object(pipeline, "run_stage", side_effect=RuntimeError("benchmark failed")) as stage:
            with self.assertRaisesRegex(RuntimeError, "benchmark failed"):
                pipeline.execute(args, journal)
            self.assertEqual(stage.call_count, 1)
            self.assertEqual(stage.call_args.args[1], "benchmark")
            self.assertFalse(args.resume_output.exists())

    def test_cleanup_verifies_all_shards_before_removing_any(self):
        benchmark = self.root/"benchmark"
        worker = benchmark/"envs_256"/"worker_00"
        worker.mkdir(parents=True)
        paths = [worker/"shard_000.pt", worker/"shard_001.pt"]
        for path in paths:
            path.write_bytes(b"temporary benchmark")
        sha = hashlib.sha256(paths[0].read_bytes()).hexdigest()
        index = {"shards": {paths[0].name: sha, paths[1].name: "wrong"}}
        (worker/"index.json").write_text(json.dumps(index))
        with self.assertRaises(ValueError):
            pipeline.clean_benchmark_shards(benchmark)
        self.assertTrue(all(path.exists() for path in paths))
        index["shards"][paths[1].name] = sha
        (worker/"index.json").write_text(json.dumps(index))
        self.assertEqual(len(pipeline.clean_benchmark_shards(benchmark)), 2)
        self.assertTrue((worker/"index.json").exists())
        self.assertFalse(any(path.exists() for path in paths))


if __name__ == "__main__":
    unittest.main()
