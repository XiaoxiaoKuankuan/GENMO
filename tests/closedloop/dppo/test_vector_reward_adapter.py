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


def test_causal_stream_and_periodic_full_audit():
    oracle,reference,vector=calculators()
    for index in range(102):
        row=actual_step(index,speed=1.);row['env_id']=3
        row['reward_primitives']=packet(row,oracle)
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
