"""GPU共享世界的唯一关闭责任、真实回执及冻结/ACK退出审计回归。

在服务器1执行无物理替身，模拟世界RPC线程已成功关闭服务但进程尚未被wait回收的
边界，确认Workers不会再次发送close造成Broken pipe，也不丢掉第一次冻结回执。
覆盖进程已经退出、仍待回收、重复登记、缺失回执、参数变化和未ACK故障。真实
Isaac进程关闭与正式保存/退出/恢复仍由八卡有限训练另外验证；本文件不代表物理验收。
"""
from copy import deepcopy
from types import SimpleNamespace as NS
import pytest
from tools.eval.run_closedloop_baseline import Workers
from gem.closedloop.dppo.vector_collector import validate_vector_close


def acknowledgement():
    return dict(closed=True,policy_unchanged=True,runtime_parameters_unchanged=True,
        gmt_parameters_frozen=True,actual_module_sha256='same',initial_module_sha256='same',
        pending_env_ids=[],deferred_reset_env_ids=[],restart_required=False,
        execution_journal=dict(backend_session_id='world',executed_seq=2,acked_seq=2,outstanding_seq=None),
        lane_journals=[dict(backend_session_id='lane',executed_seq=3,acked_seq=3,outstanding_seq=None)])


@pytest.mark.parametrize('already_exited',[False,True])
def test_external_close_is_not_sent_twice_and_retains_original_reply(tmp_path,already_exited):
    calls=[]
    class Client:
        def close(self):calls.append('socket_close')
        def call(self,*args,**kwargs):raise AssertionError('Duplicate close RPC')
    class Process:
        returncode=0 if already_exited else None
        def poll(self):return self.returncode
        def wait(self,timeout):calls.append('wait');self.returncode=0
    worker=Workers(dict(runtime=dict(rpc_timeout_s=1)),tmp_path)
    worker.entries=[dict(name='gmt',client=Client(),proc=Process(),log=NS(close=lambda:None))]
    reply=acknowledgement()
    worker.record_external_close('gmt',reply)
    validate_vector_close(reply)
    with pytest.raises(ValueError,match='unrecorded'):worker.record_external_close('gmt',reply)
    worker.close()
    assert calls==['socket_close','wait']
    assert worker.shutdown['gmt']==dict(reply,process_exit_code=0)
    assert 'close_error' not in worker.shutdown['gmt']


@pytest.mark.parametrize('fault',['policy','runtime','module','parameters','pending','deferred','restart','world_ack','lane_ack','missing_lanes'])
def test_incomplete_or_changed_vector_close_is_rejected(fault):
    result=deepcopy(acknowledgement())
    if fault=='policy':result['policy_unchanged']=False
    elif fault=='runtime':result['runtime_parameters_unchanged']=False
    elif fault=='module':result['actual_module_sha256']='changed'
    elif fault=='parameters':result['gmt_parameters_frozen']=False
    elif fault=='pending':result['pending_env_ids']=[0]
    elif fault=='deferred':result['deferred_reset_env_ids']=[0]
    elif fault=='restart':result['restart_required']=True
    elif fault=='world_ack':result['execution_journal']['acked_seq']=1
    elif fault=='lane_ack':result['lane_journals'][0]['outstanding_seq']=4
    else:result['lane_journals']=[]
    with pytest.raises(RuntimeError):validate_vector_close(result)


def test_missing_external_close_cannot_be_reported_as_success(tmp_path):
    worker=Workers({},tmp_path)
    worker.entries=[dict(name='gmt',client=None)]
    try:
        with pytest.raises(ValueError,match='acknowledgement'):worker.record_external_close('gmt',{})
        assert worker.shutdown=={}
    finally:worker.temp.cleanup()
