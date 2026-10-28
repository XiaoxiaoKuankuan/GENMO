"""正式八组资源等待与配置矩阵的CPU回归，不启动GPU、tmux或真实训练。

通过假的nvidia-smi与子进程验证：忙卡不抢占、卡释放后只启动一次、设备UUID改变
拒绝误启动、查询失败继续等待、训练失败保留退出码且不重启、已有日志不覆盖。
测试同时核对八配置只比较H8/H16、学习率和噪声，所有冻结资产与奖励/约束保持一致。
所有状态与锁文件写入pytest独立临时目录，测试结果不能作为真实物理收益证据。
"""

import importlib.util
import json
from pathlib import Path

import pytest
import yaml


REPO = Path(__file__).resolve().parents[3]
MODULE_SPEC = importlib.util.spec_from_file_location('stage2_launch_queue', REPO/'tools/launch_stage2_alternating.py')
launcher = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(launcher)


def fixture_spec(tmp_path):
    spec = dict(root=str(tmp_path), gpu=7, uuid='GPU-fixture', cwd=str(tmp_path),
                lock_dir=str(tmp_path/'locks'), output=str(tmp_path/'gpu7'),
                command=['env', 'CUDA_VISIBLE_DEVICES=7', 'fixture_python', '--output', str(tmp_path/'gpu7')])
    path = tmp_path/'wait.json'
    path.write_text(json.dumps(spec))
    return path, spec


def mock_processes(monkeypatch, states, *, code=0, uuid='GPU-fixture'):
    events, states = [], iter(states)
    def query(command, **kwargs):
        if '--query-gpu=index,uuid' in command:
            return '7, '+uuid+'\n'
        result = next(states)
        events.append(('query', result))
        if isinstance(result, BaseException):
            raise result
        return result
    class Child:
        pid = 12345
        def wait(self):
            events.append(('wait_child', code))
            return code
    def spawn(command, **kwargs):
        events.append(('spawn', command))
        assert kwargs['stderr'] == launcher.subprocess.STDOUT
        return Child()
    monkeypatch.setattr(launcher.subprocess, 'check_output', query)
    monkeypatch.setattr(launcher.subprocess, 'Popen', spawn)
    monkeypatch.setattr(launcher.time, 'sleep', lambda seconds: events.append(('sleep', seconds)))
    return events


def test_busy_gpu_waits_then_starts_once_and_records_real_pid(tmp_path, monkeypatch):
    path, spec = fixture_spec(tmp_path)
    events = mock_processes(monkeypatch, ['GPU-fixture, 90\n', 'GPU-other, 91\n'])
    assert launcher._wait_and_run(path, .01) == 0
    assert [event[0] for event in events] == ['query', 'sleep', 'query', 'spawn', 'wait_child']
    assert events[3][1] == spec['command']
    state = json.loads((tmp_path/'queue_gpu7.json').read_text())
    assert state['phase'] == 'completed' and state['training_pid'] == 12345
    assert (tmp_path/'gpu7.exit').read_text() == '0\n'


def test_gpu_query_failure_never_means_idle(tmp_path, monkeypatch):
    path, _ = fixture_spec(tmp_path)
    error = launcher.subprocess.CalledProcessError(1, ['nvidia-smi'])
    events = mock_processes(monkeypatch, [error, ''])
    assert launcher._wait_and_run(path, .01) == 0
    assert [x[0] for x in events].count('sleep') == 1
    assert [x[0] for x in events].count('spawn') == 1


def test_wrong_gpu_identity_refuses_launch(tmp_path, monkeypatch):
    path, _ = fixture_spec(tmp_path)
    events = mock_processes(monkeypatch, [], uuid='GPU-changed')
    with pytest.raises(RuntimeError, match='UUID'):
        launcher._wait_and_run(path)
    assert not events
    assert json.loads((tmp_path/'queue_gpu7.json').read_text())['phase'] == 'queue_error'


def test_training_failure_is_not_restarted(tmp_path, monkeypatch):
    path, _ = fixture_spec(tmp_path)
    events = mock_processes(monkeypatch, [''], code=3)
    assert launcher._wait_and_run(path) == 3
    assert [x[0] for x in events].count('spawn') == 1
    assert (tmp_path/'gpu7.exit').read_text() == '3\n'
    assert json.loads((tmp_path/'queue_gpu7.json').read_text())['phase'] == 'failed'


def test_existing_log_is_never_overwritten(tmp_path):
    path, _ = fixture_spec(tmp_path)
    log = tmp_path/'gpu7.log'
    log.write_text('preserve_existing_training')
    with pytest.raises(FileExistsError):
        launcher._wait_and_run(path)
    assert log.read_text() == 'preserve_existing_training'


def test_formal_matrix_has_eight_independent_parameter_combinations():
    configs = [yaml.safe_load((REPO/f'configs/closedloop/stage2_formal_20261028/gpu{i}.yaml').read_text())
               for i in range(8)]
    combinations = {(c['bptt_steps'], c['training']['adapter_lr'], c['alternating']['action_noise']) for c in configs}
    assert combinations == {(h, lr, noise) for h in (8, 16) for lr in (1e-5, 3e-5) for noise in (0., .02)}
    for c in configs:
        assert c['vector']['target_accepted_updates'] == 1000 and c['vector']['num_envs'] == 2048
        assert c['training']['wm_fit_window'] == 8 and c['training']['seed'] == 42
        assert c['alternating']['critic_enabled'] is False and c['alternating']['root_orientation'] is True
        assert c['alternating']['root_xy'] is False and c['alternating']['soft_joint_limit'] is False
        assert c['rewards'] == configs[0]['rewards']
        for name in ('supermimic', 'genmo', 'tasks'):
            assert c[name] == configs[0][name]
        assert c['alternating']['projection'] == configs[0]['alternating']['projection']
        assert c['alternating']['precision'] == configs[0]['alternating']['precision']
        assert c['alternating']['evaluation_updates'] == [0,5,10,20,50,100,200,400,700,1000]
