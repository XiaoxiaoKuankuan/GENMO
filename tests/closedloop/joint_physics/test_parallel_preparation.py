"""预处理协调器的纯CPU控制合同测试，不加载SM/WM或启动实际GPU任务。

覆盖物理GPU编号/UUID与compute进程关联、忙卡拒绝、来源排除及固定轮流分配、
恢复进度的新增吞吐口径，以及JSON原子写入不覆盖已有证据。真实Newton吞吐和
环境规模选择由服务器完整采集+前瞻benchmark单独验收，本测试不替代动力学结果。
"""

import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.prepare_joint_physics_parallel import (assigned_sources, atomic_json,
    episode_counts, parse_gpus, progress_payload, records_from_verified_shards, require_idle_gpus)
from tools.benchmark_joint_physics_preparation import select_result, summarize_trial


class TestParallelPreparation(unittest.TestCase):
    def test_legacy_converted_v2_counts_come_from_real_shard_once(self):
        import torch
        with tempfile.TemporaryDirectory(prefix="joint-parallel-test-") as temp:
            output = Path(temp)
            episodes = [{"source_key": key, "split": "train", "control_steps": steps,
                "action": torch.zeros(steps, 21), "termination_reason": reason,
                "offline_requests": {"frames": torch.tensor([0, 3]), "valid": torch.tensor([True, False])}}
                for key, steps, reason in (("a", 100, "explicit_control_budget"),
                                           ("b", 50, "sm_physical_termination"))]
            torch.save(episodes, output/"shard.pt")
            # 原转换器只保存前瞻统计，故意放入不真实数字，必须完全忽略该摘要。
            index = {"split": "train", "sources": {key: {"shard": "shard.pt", "episode": i}
                for i, key in enumerate(("a", "b"))}, "shards": {"shard.pt": "already_verified"},
                "episodes": [{"source": "a", "valid_requests": 999}]}
            def validator(episode, index, horizon):
                self.assertEqual(horizon, 32)
                return (None, None if bool(episode["offline_requests"]["valid"][index]) else "invalid")
            with patch.object(torch, "load", wraps=torch.load) as loader:
                records = records_from_verified_shards(index, output, validator)
                self.assertEqual(loader.call_count, 1)
            counts = episode_counts({"episodes": records})
            self.assertEqual(counts["sources"], 2)
            self.assertEqual(counts["control_steps"], 150)
            self.assertEqual(counts["valid_future_requests"], 2)
            self.assertEqual(counts["legal_h32_windows"], 2)
            self.assertEqual(counts["explicit_control_budget_count"], 1)
            self.assertEqual(counts["failed_termination_count"], 1)

    def test_gpu_busy_is_rejected_by_uuid_and_other_gpu_is_allowed(self):
        def runner(command, **kwargs):
            output = "0, GPU-a\n1, GPU-b\n" if "--query-gpu=" in command[1] else "GPU-b, 77, train.py\n"
            return subprocess.CompletedProcess(command, 0, output, "")
        self.assertEqual(require_idle_gpus(["0"], runner=runner), {"0": "GPU-a"})
        with self.assertRaisesRegex(RuntimeError, "77"):
            require_idle_gpus(["1"], runner=runner)
        with self.assertRaises(RuntimeError):
            require_idle_gpus(["8"], runner=runner)

    def test_gpu_ids_and_deterministic_assignments(self):
        self.assertEqual(parse_gpus("0,1,7"), ["0", "1", "7"])
        with self.assertRaises(Exception):
            parse_gpus("0,0")
        assignments = assigned_sources(list("abcdef"), {"b"}, 2, 4)
        self.assertEqual(assignments, {"0": ["a", "e"], "1": ["c", "d"]})

    def test_eight_ranks_rotate_across_four_interleaved_datasets(self):
        keys = [f"dataset{i % 4}:source{i}" for i in range(64)]
        assignment = assigned_sources(keys, set(), 8)
        flattened = [key for values in assignment.values() for key in values]
        self.assertEqual(set(flattened), set(keys))
        self.assertEqual(len(flattened), len(set(flattened)))
        for values in assignment.values():
            self.assertEqual({key.split(":")[0] for key in values}, {f"dataset{i}" for i in range(4)})
        self.assertEqual(assigned_sources(keys, set(), 1)["0"], keys)

    def test_resume_rates_count_only_newly_published_sources(self):
        def episode(key):
            return {"source_key": key, "control_steps": 100, "termination_reason": "sm_physical_termination",
                    "offline_preprocessing": {"valid_requests": 10, "legal_windows": {"32": 3}}}
        index = {"sources": dict(a={}, b={}), "episodes": [episode("a"), episode("b")]}
        manifest = {"assignments": {"0": ["a", "b", "c"]}}
        process = SimpleNamespace(pid=42, poll=lambda: None)
        value = progress_payload(index, manifest, {"a"}, 2., {0: process})
        self.assertEqual(value["completed_sources"], 2)
        self.assertEqual(value["remaining_sources"], 1)
        self.assertEqual(value["source_per_second"], .5)
        self.assertEqual(value["control_steps_per_second"], 50.)
        self.assertEqual(value["future_requests_per_second"], 5.)
        self.assertEqual(value["total"]["legal_h32_windows"], 6)
        self.assertEqual(value["eta_seconds"], 2.)

    def test_atomic_manifest_cannot_overwrite_previous_identity(self):
        with tempfile.TemporaryDirectory(prefix="joint-parallel-test-") as temp:
            path = Path(temp)/"manifest.json"
            atomic_json(path, {"workers": 8}, replace=False)
            with self.assertRaises(FileExistsError):
                atomic_json(path, {"workers": 1}, replace=False)
            self.assertEqual(json.loads(path.read_text()), {"workers": 8})
            self.assertEqual(list(Path(temp).glob("*.partial")), [])

    def test_scale_selection_excludes_failed_and_uses_complete_process_time(self):
        with tempfile.TemporaryDirectory(prefix="joint-parallel-test-") as temp:
            output = Path(temp)
            (output/"batch_manifest.json").write_text(json.dumps({"assignments": {"0": ["same_source"]}}))
            (output/"summary.json").write_text(json.dumps({"status": "completed", "frozen_unchanged": True,
                "complete_index_verified": True, "session": {"sources": 1, "control_steps": 300,
                "valid_future_requests": 12, "legal_h32_windows": 8},
                "worker_summaries": {"0": {"wall_seconds": 1., "peak_memory_bytes": 1000}}}))
            first, sources = summarize_trial(output, 256, 0, 4., None)
            self.assertTrue(first["eligible"])
            self.assertEqual(first["process_future_requests_per_second"], 3.)
            failed, _ = summarize_trial(output, 512, 1, 1., sources)
            different, _ = summarize_trial(output, 1024, 0, 1., ["another_source"])
            self.assertFalse(failed["eligible"])
            self.assertFalse(different["eligible"])
            self.assertEqual(select_result([first, failed, different])["num_envs"], 256)
            self.assertIsNone(select_result([failed]))


if __name__ == "__main__":
    unittest.main()
