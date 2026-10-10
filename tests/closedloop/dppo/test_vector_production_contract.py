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


def test_native_isaac_device_binding_preserves_training_permutation(monkeypatch):
    from gem.closedloop.dppo.vector_devices import bind_vector_device,VECTOR_DEVICE_CONTRACT
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','2,0,1,3,4,5,6,7')
    cfg={'runtime':{}}
    env=bind_vector_device(cfg,0)
    assert cfg['runtime']==dict(physics_device='cuda:2',vector_device_contract=VECTOR_DEVICE_CONTRACT)
    assert env==dict(CUDA_VISIBLE_DEVICES=None,CUDA_DEVICE_ORDER='PCI_BUS_ID')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,0,1,3,4,5,6,7')
    with pytest.raises(ValueError):bind_vector_device(cfg,0)


def test_production_vector_config_preserves_learning_and_storage():
    config=configuration(Path(__file__).resolve().parents[3]/'configs/closedloop/stage10_8gpu_server1_gpu_vectorized.yaml')
    s=config['stage9']; assert config['runtime']['num_envs']==8
    assert (s['actor_lr'],s['rollout_upper_steps'],s['rollout_upper_steps_per_rank'],s['denoising_steps'])==(5e-9,160,20,20)
    assert (s['kl_stop_joint'],s['kl_soft_stop_joint'],s['ppo_epochs'],s['max_actor_optimizer_steps'])==(.03,.015,2,4)
    assert config['stage10']['storage']['checkpoint_every_iterations']==500
    for update in ({'num_envs':32},{'prefix_deadline_contract':'unknown'},{'vector_batch_wait_s':float('nan')},
                   {'vector_device_contract':None},{'vector_fragment_contract':None}):
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


def test_gae_values_belong_to_detached_buffer_not_mutable_collection_rows():
    import torch
    from tests.closedloop.dppo.test_data_learning import transition
    from gem.closedloop.dppo.buffer import RolloutBuffer
    from gem.closedloop.dppo.vector_runtime import freeze_buffer_targets
    class Critic(torch.nn.Module):
        def forward(self, context, remaining):
            return context['music_features'].flatten(1).mean(1)+1.
    original=[transition(),transition()]
    buffer=RolloutBuffer(2)
    for slot,row in enumerate(original):
        row.identity.update(env_id=slot,episode_id=f'episode{slot}')
        row.truncated=True
        row.metadata.update(remaining_music_seconds=5.,next_remaining_music_seconds=4.96)
        buffer.append(row)
    for row in original:row.context['music_features'].fill_(100.)
    targets=freeze_buffer_targets(buffer,[1,1],Critic(),'cpu',critic_version=7,batch_size=32,
        gamma_upper=.99,lambda_upper=.95)
    assert all(r.metadata['value_snapshot_version']==7 and r.old_value==1. for r in buffer.transitions)
    assert all('value_snapshot_version' not in r.metadata for r in original)
    assert torch.isfinite(targets['returns']).all()


def test_vector_metrics_count_shared_generation_once_and_all_rewards():
    from gem.closedloop.dppo.vector_metrics import attach_vector_metrics
    rows=[SimpleNamespace(reason=None,metadata=dict(timing={'critical_ready_seconds':.7},
        reward_details=[{'components':{'track':{'integrated_reward':.04},'music':.01}}])) for _ in range(8)]
    rows[-1].reason='rpc_timeout';rows[-1].metadata['rejection']='late_plan'
    report=attach_vector_metrics(rows,{'batches':[{'generation_seconds':.6,'components':{
        'denoising_seconds':.5,'interval_scope':'current_stream_cuda_events'}}]})
    assert report['rejections']==report['timeouts']==1
    assert report['reward_component_sums']['track']==pytest.approx(.32)
    assert report['generation_timing_totals']['critical_ready_seconds']==pytest.approx(5.6)
    assert report['generation_batch_wall_seconds']==.6
    assert report['actor_phase_totals']['denoising_seconds']==.5
    assert report['actor_timing_interval_scopes']==['current_stream_cuda_events']
    assert 'interval_scope' not in report['actor_phase_totals']
