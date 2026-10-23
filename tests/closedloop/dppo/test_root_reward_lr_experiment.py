"""根位置/姿态新奖励及单卡完整旧行为批次重放的回归测试。

输入为当前连续滚动配置和模拟的多rank真实数据结构；测试只重新分配计算所有权，
确保行为链对象、有效标记与已经全局归一化的优势不被复制改写或重新标准化。
同时核对10倍、50倍学习率可通过实际配置加载，组倍率、噪声核、GMT权重和其他
六项奖励保持原值。文件仅写入pytest临时目录，不启动正式训练或物理仿真。
"""
import copy
from pathlib import Path

import pytest
import torch
import yaml

from gem.closedloop.dppo.training_config import configuration
from gem.closedloop.dppo.rewards import ROOT_RELAXED_REWARD_VERSION,ROOT_RELAXED_REWARD_STD
from tools.compare_stage2_optimizer_batches import combine_rollout_shards


def shards():
    manifest=[dict(owner_rank=r,local_index=i,valid=i!=1,has_free=True)
              for r,n in enumerate((3,2)) for i in range(n)]
    return [dict(kernel={'same':True},global_manifest=manifest,transitions=[object() for _ in range(n)],
        targets=dict(advantages=torch.arange(n,dtype=torch.float64)+10*r,
                     valid=torch.tensor([i!=1 for i in range(n)]),advantages_normalized=True,
                     advantage_global_count=3,advantage_global_mean=1.2))
        for r,n in enumerate((3,2))]


def test_complete_single_device_reassignment_preserves_behavior_and_advantages():
    parts=shards();result=combine_rollout_shards(parts)
    assert result['transitions']==sum([p['transitions'] for p in parts],[])
    assert torch.equal(result['targets']['advantages'],torch.tensor([0.,1.,2.,10.,11.],dtype=torch.float64))
    assert result['targets']['advantage_global_count']==3
    assert [m['local_index'] for m in result['global_manifest']]==list(range(5))
    assert all(m['owner_rank']==0 for m in result['global_manifest'])
    assert result['targets']['valid'].tolist()==[True,False,True,True,False]
    assert parts[1]['global_manifest'][3]['owner_rank']==1


def test_incomplete_or_inconsistent_shards_are_rejected():
    parts=shards()
    with pytest.raises(ValueError,match='All original ranks'):combine_rollout_shards(parts[:1])
    parts[1]['targets']['advantage_global_mean']=1.3
    with pytest.raises(ValueError,match='normalization'):combine_rollout_shards(parts)


@pytest.mark.parametrize('factor',[10,50])
def test_requested_lr_and_reward_configuration_preserve_other_contracts(tmp_path,factor):
    path=Path(__file__).resolve().parents[3]/'configs/closedloop/stage2_continuous_rolling.yaml'
    original=yaml.safe_load(path.read_text());changed=copy.deepcopy(original)
    t=changed['stage10']['training'];t['actor_lr']=1e-6*factor;t['trust_region']['maximum_lr']=t['actor_lr']
    target=tmp_path/'candidate.yaml';target.write_text(yaml.safe_dump(changed))
    c=configuration(target);stage=c['stage10']
    assert stage['reward']['version']==ROOT_RELAXED_REWARD_VERSION
    assert c['stage9']['reward']['tracking']['reward_std']==ROOT_RELAXED_REWARD_STD
    assert c['stage9']['actor_lr']==pytest.approx(1e-6*factor)
    for field in ('actor_lr_multipliers','std_schedule','frozen_prefix_steps','ppo_steps_per_chain','bc_enabled'):
        assert c['stage9'][field]==original['stage10']['training'][field]
    assert stage['reward']['tracking']['weights']==original['stage10']['reward']['tracking']['weights']
