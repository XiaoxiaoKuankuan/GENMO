"""GPU向量产物审计的正例和污染反例，全部在服务器1运行。

两个环境可以具有相同decision编号，但每条链必须绑定自身lane；测试故意交换身份、
重复batch槽位、伪造padding、缺失ACK及削减采样池，确保审计明确拒绝，而非为新拓扑
跳过检查。checkpoint另覆盖独立种子、累计已接受转移数、合同和完整来源排列。
使用最小只读字典夹具，不生成模型、物理结果或冒充真实吞吐验收。
"""
import pytest
from tools.eval.audit_stage10_vector_helpers import audit_vector_rows,audit_vector_checkpoint,is_vector


def fixture():
    identity=dict(base_seed=42,execution_contract={'runtime':dict(backend='gpu_vectorized.v1',num_envs=2,
        vector_fragment_contract='available_reference_fragment_boundary.v3')},
        training_contract={'rollout_upper_steps_per_rank':20},performance_contract={'numerical_layout':'sample_matrix_bmm_fp32.v1'},
        dataset={'sample_counts':{'train':{'Mine':2}}},sampling=dict(source_probabilities=[1.],random_start=True))
    rows=[dict(identity=dict(env_id=slot,backend_session_id=f'lane{slot}',decision_id=i,policy_version=0),count=25,
               terminated=False,truncated=i==9) for slot in range(2) for i in range(10)]
    collection=dict(schema='genmo.gpu_vector_collector.v1',real_environment_batch=True,allocated_envs=2,active_envs=2,
        total_transitions=20,fragment_lengths=[10,10],fragment_contract='available_reference_fragment_boundary.v3',
        batches=[dict(environment_slots=[0,1],effective_rows=2,padding_rows=0) for _ in range(10)])
    frozen=dict(execution_journal={'backend_session_id':'world'},gmt_parameters_frozen=True,
        actual_module_sha256='same',initial_module_sha256='same',pending_env_ids=[],
        lane_journals=[dict(backend_session_id=f'lane{i}',executed_seq=20,acked_seq=20,outstanding_seq=None) for i in range(2)])
    states=[]
    for slot in range(2):
        states.append(dict(seed=42+100003*slot,transitions=10,
            execution=dict(decision=10,attempt=13,episode_count=2,latency_budget_s=1.,policy_version=0,iteration=0),
            sampler=dict(split='train',catalog_identity=identity['dataset'],source_probabilities=[1.],random_start=True,
                         orders={'Mine':[1,0]},cursors={'Mine':1})))
    local=dict(iteration=1,vector_collector=dict(schema='genmo.gpu_vector_collector.boundary.v1',num_envs=2,
        fragment_contract='available_reference_fragment_boundary.v3',numerical_layout='sample_matrix_bmm_fp32.v1',
        restore_environment='fresh_PhysX_episodes_preserve_rng_cursors_and_spent_budget',states=states))
    return identity,rows,collection,frozen,local


def test_independent_numbering_and_full_checkpoint_pass():
    identity,rows,collection,frozen,local=fixture()
    assert is_vector(identity) and not is_vector({})
    audit_vector_rows(rows,collection,frozen,identity)
    result=audit_vector_checkpoint(local,rows,identity,0)
    assert result['full_pool_permutations_verified'] and len(result['environments'])==2


@pytest.mark.parametrize('fault',['lane','env','decision','padding','repeat_batch','quota','ack','world_alias','zero_step'])
def test_vector_rollout_contamination_fails(fault):
    identity,rows,collection,frozen,local=fixture()
    if fault=='lane':rows[0]['identity']['backend_session_id']='lane1'
    if fault=='env':rows[0]['identity']['env_id']=1
    if fault=='decision':rows[1]['identity']['decision_id']=0
    if fault=='padding':collection['batches'][0]['padding_rows']=1
    if fault=='repeat_batch':collection['batches'][0]['environment_slots']=[0,0]
    if fault=='quota':collection['fragment_lengths']=[9,11]
    if fault=='ack':frozen['lane_journals'][0]['acked_seq']=19
    if fault=='world_alias':frozen['lane_journals'][0]['backend_session_id']='world'
    if fault=='zero_step':rows[0]['count']=0
    with pytest.raises((AssertionError,ValueError)):audit_vector_rows(rows,collection,frozen,identity)


@pytest.mark.parametrize('fault',['missing','contract','seed','decisions','transitions','pool','cursor','version'])
def test_vector_checkpoint_contamination_fails(fault):
    identity,rows,collection,frozen,local=fixture()
    states=local['vector_collector']['states']
    if fault=='missing':states.pop()
    if fault=='contract':local['vector_collector']['fragment_contract']='old'
    if fault=='seed':states[0]['seed']=0
    if fault=='decisions':states[0]['execution']['decision']=9
    if fault=='transitions':states[0]['transitions']=9
    if fault=='pool':states[0]['sampler']['orders']['Mine']=[1]
    if fault=='cursor':states[0]['sampler']['cursors']['Mine']=3
    if fault=='version':states[0]['execution']['policy_version']=1
    with pytest.raises((AssertionError,ValueError)):audit_vector_checkpoint(local,rows,identity,0)
