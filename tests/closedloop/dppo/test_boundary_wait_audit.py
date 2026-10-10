"""服务器1八rank执行的有限轮末等待独立审计测试。

用户已授权有上限的边界等待，但等待必须进入真实转移、奖励、物理计数和恢复身份。
本文件使用小型只读证据夹具，精确比较逐控制步FP32奖励，覆盖尾段物理失败惩罚、
零等待以及汇总伪造。旧世界合同仍禁止额外drain；新合同不能只靠标签绕过检查。
测试不运行物理或训练，不能代替同机真实1024/2048/4096/8192条闭环验收。
"""
import copy
from types import SimpleNamespace
import pytest
import torch
from tools.eval.audit_stage10_vector_helpers import audit_boundary_wait_metadata, audit_vector_rows, audit_vector_checkpoint
from tests.closedloop.dppo.test_vector_artifact_audit import fixture


def tail_item(count=50, failure=False):
    tail = dict(boundary_wait_contract='bounded_reference_wait.v1', wait_limit_controls=100,
        executed_control_steps=count, executed_physics_steps=4*count, budget_reserved_controls=100,
        requested_control_steps=125, begin_tick=300, end_tick=300+12*count,
        maximum_supported_decision_tick=1800, execution_sequence=12, wait_wall_seconds=1.25,
        wait_limit_reached=False, reward_sum=.13*count-(5. if failure and count else 0.))
    values = [.13]*count
    if failure and count: values[-1]-=5.
    rewards=torch.tensor([.21]*25+values, dtype=torch.float32)
    item=SimpleNamespace(rewards=rewards,executed_control_steps=25+count,control_tick_begin=0,
        control_tick_end=tail['end_tick'],metadata=dict(fragment_tail=tail,
        reward_details=[dict(reward=.21,transition_valid=True)]*25+
                      [dict(reward=.13,transition_valid=True)]*count,
        terminal_snapshot=dict(terminated=failure)))
    identity=dict(execution_contract={'runtime':dict(vector_boundary_wait_contract='bounded_reference_wait.v1',
        vector_boundary_wait_max_control_steps=100)},reward=dict(failure_penalty=5.))
    return item,identity


@pytest.mark.parametrize('count,failure',[(0,False),(50,False),(50,True),(100,False)])
def test_boundary_exact_steps_and_physical_failure_penalty(count,failure):
    item,identity=tail_item(count,failure)
    assert audit_boundary_wait_metadata(item,identity)==item.metadata['fragment_tail']


@pytest.mark.parametrize('fault',['reward','reward_sum','physics','end','reservation','cap','supported','wall','flag','detail'])
def test_boundary_wait_metadata_tampering_rejected(fault):
    item,identity=tail_item();tail=item.metadata['fragment_tail']
    if fault=='reward': item.rewards[-1]+=.01
    if fault=='reward_sum': tail['reward_sum']+=.01
    if fault=='physics': tail['executed_physics_steps']-=4
    if fault=='end': tail['end_tick']-=12
    if fault=='reservation': tail['budget_reserved_controls']=101
    if fault=='cap': tail['wait_limit_controls']=200
    if fault=='supported': tail['maximum_supported_decision_tick']=600
    if fault=='wall': tail['wait_wall_seconds']=-1.
    if fault=='flag': tail['wait_limit_reached']=True
    if fault=='detail': item.metadata['reward_details'][-1]=dict(reward=.13,transition_valid=False)
    with pytest.raises(ValueError): audit_boundary_wait_metadata(item,identity)


def bounded_fixture():
    identity,rows,collection,frozen,local=fixture()
    _,bound=tail_item()
    identity['execution_contract']['runtime'].update(bound['execution_contract']['runtime'],
        vector_collection_contract='genmo.world_batched_flow.v1')
    collection.update(schema='genmo.world_batched_flow.v1',fragment_contract='genmo.world_batched_flow.v1',
        normal_boundary_resets=0,boundary_wait_contract='bounded_reference_wait.v1',boundary_wait_max_controls=100,
        administrative_drain_controls=100,administrative_drain_physics_steps=400,
        administrative_drain_reward=13.,administrative_wait_max_wall_seconds=1.25,administrative_wait_limit_reached=0)
    for row in (rows[9],rows[19]):
        item,check_identity=tail_item()
        row['audited_fragment_tail']=audit_boundary_wait_metadata(item,check_identity)
        row['count']=75
    local['vector_collector'].update(schema='genmo.gpu_vector_collector.boundary.v2',
        fragment_contract='genmo.world_batched_flow.v1',boundary_wait_contract='bounded_reference_wait.v1',
        boundary_wait_max_controls=100)
    return identity,rows,collection,frozen,local


def test_bounded_world_aggregate_and_checkpoint():
    identity,rows,collection,frozen,local=bounded_fixture()
    audit_vector_rows(rows,collection,frozen,identity)
    audit_vector_checkpoint(local,rows,identity,0)
    local['vector_collector']['boundary_wait_max_controls']=200
    with pytest.raises(ValueError,match='bounded wait'): audit_vector_checkpoint(local,rows,identity,0)


@pytest.mark.parametrize('fault',['controls','physics','reward','wall','tail','extra_tail','reset','old_contract'])
def test_bounded_world_does_not_hide_wait_or_relax_legacy(fault):
    identity,rows,collection,frozen,_=bounded_fixture()
    if fault=='controls':collection['administrative_drain_controls']=0
    if fault=='physics':collection['administrative_drain_physics_steps']=0
    if fault=='reward':collection['administrative_drain_reward']=0.
    if fault=='wall':collection['administrative_wait_max_wall_seconds']=0.
    if fault=='tail':rows[9].pop('audited_fragment_tail')
    if fault=='extra_tail':rows[0]['audited_fragment_tail']=copy.deepcopy(rows[9]['audited_fragment_tail'])
    if fault=='reset':collection['normal_boundary_resets']=1
    if fault=='old_contract':identity['execution_contract']['runtime'].pop('vector_boundary_wait_contract')
    with pytest.raises(ValueError):audit_vector_rows(rows,collection,frozen,identity)
