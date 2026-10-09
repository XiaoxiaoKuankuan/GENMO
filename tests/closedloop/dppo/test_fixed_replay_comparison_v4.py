"""真实物理重放比较器的严格范围测试。

完整参考里包含字符串计划ID、浮点状态和内部schema；比较器必须支持这些数组，
只排除已声明墙钟统计，不能忽略真实部署到达时间或模拟tick。细小状态变化也须
报告，不通过增加数值容差使重放通过。本测试只验证比较器，真实八卡重放另验收。
"""
import numpy as np
from tools.replay_stage10_runtime_v4 import differences, Recorder
from gem.runtime.closedloop_protocol import RemoteError
import pytest


def test_reference_string_arrays_and_dtype_preserved():
    a={'plan_ids':np.array(['a','b']), 'x':np.array([1.,2.],dtype=np.float32)}
    assert differences(a,a)==[]
    assert differences(a,dict(a,plan_ids=np.array(['a','c'])))
    assert differences(a,dict(a,x=a['x'].astype(np.float64)))


def test_wall_clock_exclusion_does_not_hide_critical_arrival_or_state():
    a=dict(schema='genmo.gmt_execution_feedback.v2',tick=600,critical_ready_seconds=.4,
           state=np.array([1.]),step_seconds=.1)
    assert differences(a,dict(a,step_seconds=.2))==[]
    assert differences(a,dict(a,critical_ready_seconds=.5))
    assert differences(a,dict(a,state=np.array([1.+1e-12])))
    assert differences(a,dict(a,tick=612))
    b=dict(a,schema='genmo.gmt_execution_feedback.columns.v3',trace_encoding_seconds=.1)
    assert differences(a,b)==[]
    assert differences(dict(schema='physical.v1'),dict(schema='physical.v2'))


def test_recorder_preserves_acknowledged_business_rejection():
    class Backend:
        MUTATIONS={'commit_plan'}
        last_envelope=dict(result=None,error=dict(code='late_plan',message='late_plan',type='ReferenceRejected'))
        last_call_timing=dict(journal_seconds=.1)
        def call(self,*a,**k):raise RemoteError(self.last_envelope['error'])
    recorder=Recorder(Backend())
    with pytest.raises(RemoteError):recorder.call('commit_plan',prepared_plan_id='p')
    assert recorder.records==[dict(method='commit_plan',payload=dict(prepared_plan_id='p'),result=None,
        remote_error=Backend.last_envelope['error'],timing=Backend.last_call_timing)]
