"""GPU多环境正式接口的配置、环境证据引用与双层持久化计时契约测试。

锁定全局160条、八卡、冻结GMT后端和原DPPO超参数，拒绝旧槽位布局、未声明延迟
协议及多于真实转移的生产环境分配。验证新环境目录的原链引用绑定准确槽位，不能
引用相邻环境或逃出rank目录。同步N=1评估transport单独检查世界journal耗时传递，
防止评估重新引入审计I/O影响模拟到达时刻的问题。全部测试在服务器1执行。
"""
from pathlib import Path
from types import SimpleNamespace
import copy
import pytest
from tools.train_closedloop_stage10 import configuration
from gem.closedloop.dppo.parallel_support import validate_v2_configuration
from gem.closedloop.dppo.rollout_storage import _valid_raw_relative
from gem.closedloop.dppo.vector_evaluation import SingleVectorLaneTransport


def test_production_vector_config_preserves_learning_and_storage():
    config=configuration(Path(__file__).resolve().parents[3]/'configs/closedloop/stage10_8gpu_server1_gpu_vectorized.yaml')
    s=config['stage9']; assert config['runtime']['num_envs']==8
    assert (s['actor_lr'],s['rollout_upper_steps'],s['rollout_upper_steps_per_rank'],s['denoising_steps'])==(5e-9,160,20,20)
    assert (s['kl_stop_joint'],s['kl_soft_stop_joint'],s['ppo_epochs'],s['max_actor_optimizer_steps'])==(.03,.015,2,4)
    assert config['stage10']['storage']['checkpoint_every_iterations']==500
    for update in ({'num_envs':32},{'prefix_deadline_contract':'unknown'},{'vector_batch_wait_s':float('nan')}):
        broken=copy.deepcopy(config);broken['runtime'].update(update)
        with pytest.raises(ValueError):validate_v2_configuration(broken)


def test_vector_raw_references_bind_environment_slot():
    assert _valid_raw_relative(Path('raw_samples/chain.pt'),{})
    assert _valid_raw_relative(Path('env003/raw_samples/chain.pt'),{'collector_env_slot':3})
    for path in ('env004/raw_samples/chain.pt','env003/../raw_samples/chain.pt','/raw_samples/chain.pt'):
        assert not _valid_raw_relative(Path(path),{'collector_env_slot':3})


def test_eval_transport_carries_world_audit_time():
    def call(method,**payload):
        assert method=='exchange'
        row=payload['requests'][0]
        assert row['env_id']==0
        return {'replies':[{'request_id':row['request_id'],'ok':True,'result':{'tick':600}}]}
    world=SimpleNamespace(call=call,last_call_timing={'journal_seconds':.4})
    client=SingleVectorLaneTransport(world)
    assert client.call('snapshot')=={'tick':600}
    assert client.last_call_timing['nested_journal_seconds']==.4
