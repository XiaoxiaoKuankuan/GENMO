"""验证第二阶段八卡入口在成功与受控拒绝时均销毁通信组。

本文件通过替代 GPU 资源查询、CUDA 初始化及通信器，只执行真实入口的生命周期
控制逻辑，不启动训练、物理环境或后台进程。测试覆盖成功结束时保留最终 barrier，
以及 KL 等受控拒绝返回非零状态时跳过额外 barrier、按序销毁 NCCL 与默认 Gloo。
这样可以避免将真实拒绝误报为成功，也防止异常退出遗留通信资源的警告。
"""

import pytest

from gem.closedloop.dppo import parallel_training
from tools import train_closedloop_stage10_8gpu as entry


@pytest.mark.parametrize('exit_code', [0, 1])
def test_parallel_entry_releases_groups_after_controlled_result(monkeypatch, tmp_path, exit_code):
    events = []
    for key, value in {'RANK': '0', 'LOCAL_RANK': '0', 'WORLD_SIZE': '8'}.items():
        monkeypatch.setenv(key, value)
    config = {
        'stage10': {'version': entry.training.VERSION_V2,
                    'distributed': {'world_size': 8, 'backend': 'nccl', 'collection': 'all_ranks'},
                    'limits': {'accepted_iterations': 2}},
        'runtime': {'torch_threads': 1},
    }
    monkeypatch.setattr(entry.training, 'configuration', lambda _: config)
    monkeypatch.setattr(entry.training, 'runtime_preflight', lambda *a, **k: {'ready': True})
    monkeypatch.setattr(entry, '_available_gpus', lambda: list(range(8)))
    monkeypatch.setattr(entry.dist, 'init_process_group', lambda *a, **k: events.append('init'))
    monkeypatch.setattr(entry.dist, 'broadcast_object_list', lambda *a, **k: None)
    monkeypatch.setattr(entry.dist, 'new_group', lambda **k: 'tensor_group')
    monkeypatch.setattr(entry.dist, 'barrier', lambda: events.append('barrier'))
    monkeypatch.setattr(entry.dist, 'destroy_process_group',
                        lambda group='default': events.append(('destroy', group)))
    monkeypatch.setattr(entry.torch.cuda, 'set_device', lambda _: None)
    monkeypatch.setattr(entry.torch, 'set_num_threads', lambda _: None)
    original_tensor = entry.torch.tensor
    monkeypatch.setattr(entry.torch, 'tensor', lambda value, **_: original_tensor(value))

    class FakeCollectives:
        def __init__(self, *args, **kwargs):
            pass

        def sum_tensor(self, value):
            return 36.0

    monkeypatch.setattr(entry, 'DistributedCollectives', FakeCollectives)
    monkeypatch.setattr(parallel_training, 'run_parallel', lambda *a: exit_code)
    assert entry.main(['--output-dir', str(tmp_path / 'never_created')]) == exit_code
    assert events == ['init', *(['barrier'] if exit_code == 0 else []),
                      ('destroy', 'tensor_group'), ('destroy', 'default')]
    assert not (tmp_path / 'never_created').exists()
