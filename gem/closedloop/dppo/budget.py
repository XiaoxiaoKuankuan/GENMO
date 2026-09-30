"""第九步有限验收的持久化运行预算。

生成请求和控制步在执行副作用之前预占并立即原子写盘；异常或进程崩溃不退还未知
消耗。正常返回的提前终止只结算有确切证据的未执行步。不同模式和 resume 共享同一
账本，校准、warmup、拒绝、延迟等待及诊断均计入预算，不能重启程序绕过全轮上限。
"""
from __future__ import annotations

import json
import os
from pathlib import Path


class BudgetExceeded(RuntimeError):
    pass


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class RunBudget:
    def __init__(self, path, *, generations=256, control_steps=10000, iterations=3):
        self.path = Path(path)
        self.limits = dict(generations=int(generations), control_steps=int(control_steps),
                           physics_steps=4 * int(control_steps), iterations=int(iterations))
        if any(v <= 0 for v in self.limits.values()):
            raise ValueError('Budget limits must be positive')
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state['limits'] != self.limits:
                raise ValueError('Cannot change an existing run budget')
        else:
            self.state = dict(limits=self.limits, used={key: 0 for key in self.limits}, phases={})
            self._save()

    def _save(self):
        atomic_json(self.path, self.state)

    def reserve(self, phase, **amounts):
        for key, count in amounts.items():
            if key not in self.limits or type(count) is not int or count < 0:
                raise ValueError('Invalid budget reservation')
            if self.state['used'][key] + count > self.limits[key]:
                raise BudgetExceeded(f'{key} budget exhausted in {phase}')
        entries = self.state['phases'].setdefault(phase, {})
        phase_cap = {'calibration':150,'main':6500,'resume':1800,
                     'comparison_A':300,'comparison_B':300,'comparison_C':300,'diagnostic':650}.get(phase)
        if phase_cap is not None and entries.get('control_steps',0)+amounts.get('control_steps',0)>phase_cap:
            raise BudgetExceeded(f'{phase} control allocation exhausted')
        for key, count in amounts.items():
            self.state['used'][key] += count
            entries[key] = entries.get(key, 0) + count
        self._save()

    def settle_control(self, phase, requested, result):
        if not result.get('physics_count_exact', True):
            return
        actual = int(result['executed_control_steps'])
        physics = result.get('executed_physics_steps')
        if physics is None or not (0 <= actual <= requested and 4 * actual <= physics <= 4 * requested):
            raise ValueError('Cannot settle inconsistent execution counts')
        refunds = dict(control_steps=requested-actual, physics_steps=4*requested-int(physics))
        for key, count in refunds.items():
            self.state['used'][key] -= count
            self.state['phases'][phase][key] -= count
        self._save()

    def state_dict(self):
        return json.loads(json.dumps(self.state))
