"""独立评估/MC世界启动重试的纯CPU验证：瞬时失败换新目录重试，持续失败才抛出。

替换单次启动函数与等待，不启动Isaac；检查重试标签互不复用（目录不冲突），成功后返回
世界句柄，三次失败的错误信息包含每次原因。
"""
from __future__ import annotations

import pytest

from gem.closedloop.dppo import vector_evaluation


def test_isolated_world_start_retries_with_fresh_labels(monkeypatch):
    calls = []
    def once(context, label):
        calls.append(label)
        if len(calls) < 3:
            raise RuntimeError('remote ground USD unavailable')
        return 'world'
    monkeypatch.setattr(vector_evaluation, '_open_isolated_world_once', once)
    monkeypatch.setattr(vector_evaluation.time, 'sleep', lambda seconds: None)
    assert vector_evaluation.open_isolated_world(None, 'eval_x') == 'world'
    assert calls == ['eval_x', 'eval_x_retry1', 'eval_x_retry2']


def test_isolated_world_start_raises_after_three_failures(monkeypatch):
    def fail(context, label):
        raise RuntimeError('down '+label)
    monkeypatch.setattr(vector_evaluation, '_open_isolated_world_once', fail)
    monkeypatch.setattr(vector_evaluation.time, 'sleep', lambda seconds: None)
    with pytest.raises(RuntimeError, match='failed to start.*down eval_y.*down eval_y_retry2'):
        vector_evaluation.open_isolated_world(None, 'eval_y')
