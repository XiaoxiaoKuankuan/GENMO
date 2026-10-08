"""正式训练自动接续入口的CPU门控测试。

所有GPU查询、子进程训练和代码核验都用受控替身，不连接服务器、不创建真实训练。
测试确保占用/高显存/查询失败不被解释为空闲，空闲计数遇忙归零；只有两次验收、
真实批量配置、严格概率门槛及审计全部通过后才可调用正式入口。使用临时目录验证
顺序和失败停止，不降低生产启动器的检查，也不更改任何正式学习配置。
"""
import copy
import json
from pathlib import Path
import subprocess

import pytest
import yaml

from scripts import start_stage10_when_idle as launch


@pytest.mark.parametrize('pids,memory,expected', [('', 4, True), ('123\n123', 4, False), ('', 512, False)])
def test_gpu_query_requires_no_compute_and_low_memory(monkeypatch, pids, memory, expected):
    def output(command, **kwargs):
        return pids if '--query-compute-apps=pid' in command else '\n'.join(f'{i}, {memory}' for i in range(8))
    monkeypatch.setattr(subprocess, 'check_output', output)
    assert launch.gpu_snapshot()['idle'] is expected


def test_failed_gpu_query_is_not_idle(monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.TimeoutExpired('nvidia-smi', 20)
    monkeypatch.setattr(subprocess, 'check_output', fail)
    with pytest.raises(subprocess.TimeoutExpired):
        launch.gpu_snapshot()


def test_idle_count_resets_on_busy_and_pending_cancel_stops(tmp_path, monkeypatch):
    states = iter([True, False, True, True, True])
    seen = []
    def snapshot():
        idle = next(states)
        seen.append(idle)
        return dict(idle=idle)
    monkeypatch.setattr(launch, 'gpu_snapshot', snapshot)
    monkeypatch.setattr(launch.time, 'sleep', lambda *_: None)
    runner = launch.Launcher(tmp_path, dict(max_wait_seconds=60))
    runner.wait_idle('waiting')
    assert len(seen) == 5
    (tmp_path/'cancel').touch()
    with pytest.raises(RuntimeError, match='cancelled'):
        runner.wait_idle('waiting')
    assert len(seen) == 5


@pytest.mark.parametrize('fault', [None, 'audit', 'microbatch', 'cfg', 'probability', 'early_end'])
def test_workflow_never_launches_formal_after_failed_acceptance(tmp_path, monkeypatch, fault):
    config = yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/closedloop/stage10_8gpu_server1_v2.yaml').read_text())
    cfg = tmp_path/'formal.yaml'
    cfg.write_text(yaml.safe_dump(config))
    request = dict(max_wait_seconds=60, config=str(cfg), formal_run=str(tmp_path/'formal'),
                   genmo_repo=str(tmp_path), capacity_reference=str(tmp_path/'reference'))
    runner = launch.Launcher(tmp_path, request)
    phases = []
    monkeypatch.setattr(runner, 'wait_idle', lambda phase, **kw: phases.append(phase))
    monkeypatch.setattr(launch, 'verify_sources', lambda *_: phases.append('verify_sources'))
    def command(phase, args, **kwargs):
        phases.append(phase)
        if phase == 'acceptance_first':
            assert args[-1] == '1'
        if phase == 'acceptance_resume':
            assert args[-4:] == ['--stop-after-iteration', '2', '--resume', 'latest']
            profile = dict(microbatch=2, cfg_batch=True,
                probability_tolerances=dict(logprob=1e-4, ratio=1e-3, gaussian=1e-8))
            if fault == 'microbatch':profile['microbatch'] = 1
            if fault == 'cfg':profile['cfg_batch'] = False
            if fault == 'probability':profile['probability_tolerances']['logprob'] = .1
            for number in (1, 2):
                path = tmp_path/'acceptance'/'sessions'/str(number)/'summary.json'
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(dict(status='passed', final_state=dict(iteration=number,
                    execution_profile=copy.deepcopy(profile)))))
        elif phase == 'acceptance_audit':
            launch.write_json(tmp_path/'acceptance_audit.json', dict(status='passed',
                failed_checks=1 if fault == 'audit' else 0, not_run_checks=0))
        elif phase == 'formal_training':
            assert args[-2:] == ['--stop-after-iteration', '10000']
            root = Path(request['formal_run'])
            root.mkdir()
            launch.write_json(root/'latest.json', dict(iteration=500 if fault == 'early_end' else 10000))
    monkeypatch.setattr(runner, 'command', command)
    if fault:
        with pytest.raises(RuntimeError):runner.execute()
        assert ('formal_training' in phases) == (fault == 'early_end')
    else:
        runner.execute()
        assert phases.index('acceptance_resume') < phases.index('acceptance_audit') < phases.index('formal_training')
        assert phases.index('waiting_formal_gpu') < phases.index('capacity_recheck') < phases.index('formal_training')
        assert json.loads((tmp_path/'status.json').read_text())['phase'] == 'completed'
