"""GPU奖励证据消费器的独立因果状态、身份隔离与强制审计测试。

使用原奖励夹具构造权威分项，检查GPU消费器与完整标量奖励的窗口、前一步目标、
音乐及积分一致。故意交换环境、episode、时间或配置SHA时必须立即拒绝；首步和
第101步注入数值错误必须被严格审计发现。测试仅在服务器1运行，不用这些合成
结果代替八卡真实GPU计算/物理闭环；真实GPU公式另由固定证据和有限训练验收。
"""
import copy
import pytest
from gem.closedloop.dppo.rewards import ExecutionReward
from gem.closedloop.dppo.vector_reward_math import VECTOR_REWARD_VERSION,reward_config_sha
from gem.closedloop.dppo.vector_reward_adapter import VectorExecutionReward,compare_reward_evidence
from test_data_learning import actual_step,music_fixture,target_activity_fixture


def calculators():
    return [cls(music_features=music_fixture(),target_activity=target_activity_fixture)
            for cls in (ExecutionReward,ExecutionReward,VectorExecutionReward)]


def packet(row,oracle):
    result={}
    for name in ('track','stable','alive','cmd','torque','contact','joint_limit'):
        score,valid,raw,normalized=getattr(oracle,'_'+name)(row)
        result[name]=dict(score=score,valid=valid,raw=raw,normalized=normalized)
    return dict(schema=VECTOR_REWARD_VERSION,config_sha256=reward_config_sha(oracle.config),
        **{key:row[key] for key in ('env_id','episode_id','tick')},components=result)


@pytest.mark.parametrize("lazy",[False,True])
def test_causal_stream_and_periodic_full_audit(lazy):
    oracle,reference,vector=calculators()
    for index in range(102):
        row=actual_step(index,speed=1.);row['env_id']=3
        row['reward_primitives']=packet(row,oracle)
        if lazy:
            from gem.runtime.trajectory_blocks import pack_trace,unpack_trace
            row=unpack_trace(pack_trace([row]),readonly_views=True,lazy=True)[0]
        compare_reward_evidence(vector.evaluate_step(row),reference.evaluate_step(row))
    assert vector.scalar_audits==2 and vector.control_count==102


@pytest.mark.parametrize('key,value',[('env_id',4),('episode_id','different'),('tick',999),
    ('config_sha256','bad'),('schema','unknown')])
def test_binding_mismatch_fails_before_reward_state_changes(key,value):
    oracle,_,vector=calculators();row=actual_step();row['env_id']=3
    row['reward_primitives']=packet(row,oracle);row['reward_primitives'][key]=value
    with pytest.raises(ValueError,match='identity'):vector.evaluate_step(row)
    assert vector.control_count==0 and vector._last_tick is None


def test_first_step_numeric_drift_is_not_silently_accepted():
    oracle,_,vector=calculators();row=actual_step();row['env_id']=3
    row['reward_primitives']=packet(row,oracle)
    row['reward_primitives']['components']['track']['score']-=.01
    with pytest.raises(AssertionError):vector.evaluate_step(row)


def test_cmd_warmup_seed_is_retained_in_first_audit():
    oracle,reference,vector=calculators();row=actual_step(speed=1.);row['env_id']=3
    target=row['joint_position_target'].copy()+.003
    for calculator in (oracle,reference,vector):calculator.seed_previous_target(target)
    row['reward_primitives']=packet(row,oracle)
    compare_reward_evidence(vector.evaluate_step(row),reference.evaluate_step(row))


def test_runtime_reward_import_does_not_load_training_dataset():
    import subprocess
    import sys
    code = '''
import importlib.abc, sys
class NoTrainingLogging(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'colorlog':
            raise ImportError('Frozen runtime must not require training logging')
guard=NoTrainingLogging();sys.meta_path.insert(0,guard)
from gem.closedloop.dppo.vector_reward_math import VectorRewardMath
assert 'gem.closedloop.stage1_dataset' not in sys.modules
sys.meta_path.remove(guard)
from gem.closedloop import BumiClosedLoopStage1Dataset
from gem.closedloop.stage1_dataset import BumiClosedLoopStage1Dataset as direct
assert direct is BumiClosedLoopStage1Dataset
'''
    subprocess.run([sys.executable,'-B','-c',code],check=True)


@pytest.mark.parametrize('block_size',[1,7,25])
def test_columnar_complete_reward_and_causal_state(block_size):
    from gem.runtime.trajectory_blocks import pack_trace, ColumnarTrace
    oracle,reference,vector=calculators()
    data=[]
    for index in range(103):
        row=actual_step(index,speed=1.+.1*(index%9));row['env_id']=3
        row['reward_primitives']=packet(row,oracle);data.append(row)
    for begin in range(0,len(data),block_size):
        batch=data[begin:begin+block_size]
        results=vector.evaluate_batch(ColumnarTrace([pack_trace(batch)]))
        for row,result in zip(batch,results):
            compare_reward_evidence(result,reference.evaluate_step(row))
    assert vector.columnar_seconds>0 and vector.scalar_audits==2
    assert vector.control_count==103 and vector._last_tick==reference._last_tick
    for actual,expected in zip(vector.window,reference.window):
        assert actual[0]==expected[0]
        compare_reward_evidence(actual[1],expected[1])
        compare_reward_evidence(actual[2],expected[2])
    compare_reward_evidence(vector._previous_target,reference._previous_target)


def test_columnar_rejects_swapped_identity_without_state_change():
    from gem.runtime.trajectory_blocks import pack_trace, ColumnarTrace
    oracle,_,vector=calculators();row=actual_step();row['env_id']=3
    row['reward_primitives']=packet(row,oracle);row['reward_primitives']['episode_id']='wrong'
    with pytest.raises(ValueError,match='identity'):
        vector.evaluate_batch(ColumnarTrace([pack_trace([row])]))
    assert vector._last_tick is None and vector.control_count==0


def test_world_timing_counts_replaced_reward_instances_without_negative_deltas():
    from gem.closedloop.dppo.vector_reward_adapter import capture_columnar_timings
    from gem.runtime.trajectory_blocks import pack_trace, ColumnarTrace
    histories = []
    # 模拟同一slot换episode：第二个对象从零计数，旧对象仍有大量累计历史。
    with capture_columnar_timings() as total:
        for instance in range(2):
            oracle, reference, vector = calculators()
            vector.columnar_timings['reward_field_assembly_seconds'] = 1000.
            before = dict(vector.columnar_timings)
            row = actual_step(0, speed=1.); row['env_id'] = 3
            row['reward_primitives'] = packet(row, oracle)
            result = vector.evaluate_batch(ColumnarTrace([pack_trace([row])]))[0]
            compare_reward_evidence(result, reference.evaluate_step(row))
            histories.append({k: v-before.get(k, 0.) for k, v in vector.columnar_timings.items()})
    assert all(value >= 0 for value in total.values())
    for key in total:
        assert total[key] == pytest.approx(sum(row.get(key, 0.) for row in histories), abs=1e-12)
    assert total['reward_field_assembly_seconds'] < 1000.


def test_columnar_timing_scope_is_restored_after_failure():
    from gem.closedloop.dppo.vector_reward_adapter import capture_columnar_timings
    with capture_columnar_timings() as outer:
        with pytest.raises(RuntimeError):
            with capture_columnar_timings() as inner:
                _, _, vector = calculators()
                vector._record_columnar_timings({'test_seconds': 2.})
                raise RuntimeError('injected world failure')
        vector._record_columnar_timings({'test_seconds': 3.})
    vector._record_columnar_timings({'test_seconds': 5.})
    assert inner == {'test_seconds': 2.}
    assert outer == {'test_seconds': 3.}


@pytest.mark.parametrize('damage',['nan_duration','integer_consistency','extra_component'])
def test_columnar_faults_keep_scalar_invalid_evidence_between_audits(damage):
    from gem.runtime.trajectory_blocks import pack_trace,ColumnarTrace
    oracle,_,vector=calculators()
    first=actual_step(0);first['env_id']=3;first['reward_primitives']=packet(first,oracle)
    vector.evaluate_step(first)
    reference=copy.deepcopy(vector)
    row=actual_step(1);row['env_id']=3;row['reward_primitives']=packet(row,oracle)
    if damage=='nan_duration':row['physics_substeps'][0]['dt_s']=float('nan')
    elif damage=='integer_consistency':row['reference_consistency']['valid']=1
    else:row['reward_primitives']['components']['extra']={}
    if damage=='extra_component':
        with pytest.raises(ValueError,match='identity'):reference.evaluate_step(row)
        with pytest.raises(ValueError,match='identity'):vector.evaluate_batch(ColumnarTrace([pack_trace([row])]))
    else:
        expected=reference.evaluate_step(row)
        actual=vector.evaluate_batch(ColumnarTrace([pack_trace([row])]))[0]
        compare_reward_evidence(actual,expected)
        assert not actual['transition_valid']
    assert vector.columnar_fallbacks==1
