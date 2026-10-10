"""将GPU批量连续奖励接入原因果奖励流，并保留周期性标量审计。

冻结GMT在同一物理控制点计算七项连续奖励，本适配核对版本、配置SHA、环境、
episode和时间戳后消费完整score/raw/normalized证据。音乐节拍、配对活动窗、
事件惩罚、权重、积分及功率/参考一致性诊断继续使用原ExecutionReward实现。
不重新定义奖励，也不把不同环境的窗口拼在一起。首个控制步及每100步用独立
标量奖励对象重算全部结果，保持atol=1e-8、rtol=1e-7；异常立即失败而非降级。
GPU前一步PD目标通过预热自然建立，CPU副本同步维护以支持审计和显式episode reset。
旧CPU路径不导入本适配；使用者必须显式绑定新的奖励计算合同。
"""
from __future__ import annotations
from collections.abc import Mapping
import copy
import time
import numpy as np
from .rewards import ExecutionReward
from .vector_reward_math import VECTOR_REWARD_VERSION, reward_config_sha


def compare_reward_evidence(actual, expected, path='reward'):
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise AssertionError(f'{path}: reward evidence fields differ')
        for key in expected:
            compare_reward_evidence(actual[key], expected[key], f'{path}.{key}')
    elif isinstance(expected, (list, tuple)) and expected and isinstance(expected[0], Mapping):
        if len(actual) != len(expected):
            raise AssertionError(f'{path}: reward evidence length differs')
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare_reward_evidence(left, right, f'{path}.{index}')
    elif (expected is None or isinstance(expected, str) or
          isinstance(expected, (list, tuple)) and expected and isinstance(expected[0], str)):
        if actual != expected:
            raise AssertionError(f'{path}: reward metadata differs')
    else:
        np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-7, err_msg=path)


class VectorExecutionReward(ExecutionReward):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_sha256 = reward_config_sha(self.config)
        self.control_count = self.scalar_audits = 0
        self.vector_version = VECTOR_REWARD_VERSION
        self.columnar_seconds = 0.
        self.columnar_fallbacks = 0
        self.last_columnar_fallback = None
        self.columnar_timings = {}

    def evaluate_batch(self, trace):
        """完整列式数值消费，逐行仅组装兼容证据；周期标量审计不省略。"""
        from gem.runtime.trajectory_blocks import ColumnarTrace
        from .columnar_reward import evaluate_columns
        if not isinstance(trace, ColumnarTrace):
            return [self.evaluate_step(row) for row in trace]
        if not len(trace): return []
        started = time.perf_counter()
        timings={}
        try:
            results, new = evaluate_columns(self, trace,timings=timings)
        except (ValueError, KeyError, IndexError, TypeError) as error:
            # 原错误证据及逐步状态推进语义保持不变，不吞掉不合法奖励。
            self.columnar_fallbacks += 1
            self.last_columnar_fallback = str(error)
            return [self.evaluate_step(row) for row in trace]
        assembled=time.perf_counter()
        for i, (row, result) in enumerate(zip(trace, results)):
            if self.control_count % 100 == 0:
                reference = object.__new__(ExecutionReward)
                reference.__dict__ = copy.deepcopy(self.__dict__)
                compare_reward_evidence(result, reference.evaluate_step(row))
                self.scalar_audits += 1
            self.window.append(new[i])
            self._last_tick, self._episode = row['tick'], row['episode_id']
            self._previous_target = np.array(row['joint_position_target'], copy=True)
            self.control_count += 1
        self.columnar_seconds += time.perf_counter()-started
        timings['independent_audit_and_state_commit_seconds']=time.perf_counter()-assembled
        for key,value in timings.items():self.columnar_timings[key]=self.columnar_timings.get(key,0.)+value
        return results

    def evaluate_step(self, trace):
        packet = trace.get('reward_primitives', {})
        if (packet.get('schema') != VECTOR_REWARD_VERSION or
                packet.get('config_sha256') != self.config_sha256 or
                any(packet.get(key) != trace.get(key) for key in ('env_id', 'episode_id', 'tick')) or
                not all(key in trace for key in ('env_id', 'episode_id', 'tick')) or
                set(packet.get('components', {})) != {'track', 'stable', 'alive', 'cmd', 'torque', 'contact', 'joint_limit'}):
            raise ValueError('GPU reward identity/configuration/fields mismatch')
        reference = None
        if self.control_count % 100 == 0:
            reference = object.__new__(ExecutionReward)
            reference.__dict__ = copy.deepcopy(self.__dict__)
        result = super().evaluate_step(trace)
        if reference is not None:
            compare_reward_evidence(result, reference.evaluate_step(trace))
            self.scalar_audits += 1
        self.control_count += 1
        return result

    @staticmethod
    def _component(row, name):
        item = row['reward_primitives']['components'][name]
        return float(item['score']), bool(item['valid']), item['raw'], item['normalized']

    def _track(self, row): return self._component(row, 'track')
    def _stable(self, row): return self._component(row, 'stable')
    def _alive(self, row): return self._component(row, 'alive')
    def _torque(self, row): return self._component(row, 'torque')
    def _contact(self, row): return self._component(row, 'contact')
    def _joint_limit(self, row): return self._component(row, 'joint_limit')

    def _cmd(self, row):
        result = self._component(row, 'cmd')
        self._previous_target = np.array(result[2]['joint_position_target_rad'], copy=True)
        return result
