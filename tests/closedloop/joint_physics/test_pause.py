"""协作暂停请求与凭据的CPU合同回归，不创建GENMO模型或使用GPU。

测试请求只在optimizer检查边界被rank0接受、错误run路径与半写JSON不会误停训练、
非rank0不读取控制文件，以及仅已有checkpoint可发布不覆盖旧结果的暂停凭据。
纯Python测试可直接通过unittest执行；分布式同步仍由训练已有日志all_reduce承担，
本测试不把文件合同验证宣称为当前运行中的旧进程已支持暂停。
"""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from gem.closedloop.joint_physics.pause import PauseControl


def _pause_rank(rank, output, rendezvous):
    import torch
    from torch import distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://"+rendezvous, rank=rank, world_size=2)
    try:
        control = PauseControl(output, rank)
        decisions = []
        for step in (9, 10):
            # 与trainer一样复用一次日志张量归约，数据值保持全rank平均。
            values = torch.tensor([2.+rank*2, float(control.local_flag(step))])
            dist.all_reduce(values)
            values = (values/2).tolist()
            control.accept_global_flag(values.pop())
            assert values == [3.]
            decisions.append(control.requested)
        Path(output, f"rank{rank}.json").write_text(json.dumps(decisions))
        dist.barrier()
    finally:
        dist.destroy_process_group()


class TestPauseControl(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="joint-pause-test-")
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.path = self.output / "pause-request.json"

    def request(self, **overrides):
        self.path.write_text(json.dumps({"output": str(self.output), "reason": "完成预处理后恢复", **overrides}))

    def test_no_request_and_check_interval(self):
        control = PauseControl(self.output, 0)
        self.assertEqual(control.local_flag(10), 0)
        self.request()
        self.assertEqual(control.local_flag(11), 0)
        self.assertEqual(control.local_flag(20), 1)
        self.assertEqual(control.request["reason"], "完成预处理后恢复")
        self.path.unlink()
        self.assertEqual(control.local_flag(30), 1)

    def test_rank_one_never_reads_but_accepts_global_decision(self):
        self.request()
        control = PauseControl(self.output, 1)
        self.assertEqual(control.local_flag(10), 0)
        control.accept_global_flag(1/8)
        self.assertTrue(control.requested)
        self.assertIsNone(control.request)
        with self.assertRaises(RuntimeError):
            control.publish(10, self.output / "joint_00000010.pt", 8)

    def test_reject_wrong_run_empty_reason_and_partial_json(self):
        for value in ({"output": "/another/run", "reason": "pause"},
                      {"output": str(self.output), "reason": " "},
                      {"output": ".", "reason": "pause"}, []):
            self.path.write_text(json.dumps(value))
            control = PauseControl(self.output, 0)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(control.local_flag(10), 0)
                self.assertIsNotNone(control.last_error)
        self.path.write_text('{"reason":')
        control = PauseControl(self.output, 0)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(control.local_flag(10), 0)
        self.request()
        self.assertEqual(control.local_flag(20), 1)

    def test_publish_requires_saved_checkpoint_and_preserves_old_marker(self):
        self.request()
        control = PauseControl(self.output, 0)
        control.accept_global_flag(control.local_flag(20))
        checkpoint = self.output / "joint_00000020.pt"
        with self.assertRaises(ValueError):
            control.publish(20, checkpoint, 8)
        checkpoint.write_bytes(b"mock complete checkpoint; no model quality claim")
        result = control.publish(20, checkpoint, 8)
        self.assertEqual(result["global_step"], 20)
        self.assertEqual(json.loads((self.output / "paused.json").read_text()), result)
        with self.assertRaises(FileExistsError):
            control.publish(20, checkpoint, 8)
        self.assertEqual(list(self.output.glob("*.partial")), [])

    def test_two_rank_existing_metric_reduce_delivers_same_pause_decision(self):
        from torch import multiprocessing as mp
        self.request()
        mp.spawn(_pause_rank, args=(str(self.output), str(self.output / "rendezvous")), nprocs=2, join=True)
        for rank in range(2):
            self.assertEqual(json.loads((self.output / f"rank{rank}.json").read_text()), [False, True])


if __name__ == "__main__":
    unittest.main()
