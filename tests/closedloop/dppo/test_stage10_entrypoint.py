"""第十步正式采集边界、配置接线与价值归档的CPU回归。

使用真实UpperTransition、RolloutBuffer和分块RolloutWriter，外部物理环境用明确
替身提供已经完成的区间；验证前三个episode不会被验收特例截短、行政批次边界保留
下一状态，以及写盘前已有冻结Critic的价值。另核验gamma/lambda可调时真正改变
固定目标。所有文件写pytest临时目录，不加载大模型、GPU或Isaac，不声称动力学成功。
"""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch
import yaml

from gem.closedloop.dppo.run_management import RolloutWriter
from gem.closedloop.dppo.trainer import fixed_targets, critic_update
from tests.closedloop.dppo.test_data_learning import transition
from tools import train_closedloop_stage10 as entry


class Sampler:
    def __init__(self, split='train'):
        self.split, self.calls, self.executions = split, 0, []

    def next_task(self):
        self.calls += 1
        return dict(sample=dict(dataset='AIST++', row=dict(sample_id=str(self.calls), split=self.split),
                                manifest_sha256='a'*64), music=np.zeros((120,35)), music_start_frame=30)

    def record_execution(self, task, controls):
        self.executions.append((task['sample']['row']['sample_id'], controls))

    def coverage(self):
        return {'controls':sum(v for _,v in self.executions)}


class Environment:
    def __init__(self, *, bad_version=False):
        self.config={'stage9':{'seed':42}}
        self.episode_count=self.iteration=self.policy_version=0
        self.starts=[]
        self.bad_version=bad_version

    def reset_task(self, sample, music, *, seed, phase, music_start_frame):
        assert len(music)==120 and phase=='train'
        self.episode_count+=1
        self.step_count=0
        self.starts.append(music_start_frame)

    def step(self):
        row=transition()
        row.identity.update(episode_id=str(self.episode_count),decision_id=self.step_count,
                            policy_version=self.policy_version+int(self.bad_version))
        row.control_tick_begin=600+24*self.step_count
        row.control_tick_end=row.control_tick_begin+24
        self.step_count+=1
        row.truncated=self.step_count==3
        return row


def test_full_collection_keeps_normal_first_three_episodes_and_values(tmp_path):
    sampler,env=Sampler(),Environment()
    writer=RolloutWriter(tmp_path/'rollout',policy_version=0,chunk_size=3)
    def values(row):
        row.old_value=4.;row.next_value=5.
    buffer,report=entry.collect_rollout(env,sampler,8,writer,value_snapshot=values)
    assert env.starts==[30,30,30]
    assert [r.identity['episode_id'] for r in buffer.transitions]==['1']*3+['2']*3+['3']*2
    assert not buffer.transitions[0].truncated
    assert buffer.transitions[-1].truncated and buffer.transitions[-1].next_context is not None
    assert report['control_steps']==16 and report['full_train_pool']
    assert sampler.coverage()=={'controls':16}
    for path in (tmp_path/'rollout').glob('chunk_*/transition_*.pt'):
        row=torch.load(path,weights_only=False)
        assert row.old_value==4. and row.next_value==5.
        assert row.metadata['training_task']['split']=='train'
        assert row.metadata['training_task']['music_start_frame']==30
    manifest=json.loads((tmp_path/'rollout/manifest.json').read_text())
    assert manifest['transition_count']==8 and len(manifest['chunks'])==3


@pytest.mark.parametrize('kind',['split','version'])
def test_collection_rejects_eval_or_wrong_policy_before_training(tmp_path,kind):
    sampler=Sampler('val' if kind=='split' else 'train')
    writer=RolloutWriter(tmp_path/'rollout',policy_version=0)
    with pytest.raises(ValueError):
        entry.collect_rollout(Environment(bad_version=kind=='version'),sampler,2,writer)
    assert not (tmp_path/'rollout/manifest.json').exists()


class ConstantCritic(torch.nn.Module):
    def __init__(self):
        super().__init__();self.value=torch.nn.Parameter(torch.tensor(2.))

    def forward(self,context,remaining):
        return self.value.expand(len(remaining))


def test_explicit_gamma_changes_fixed_returns_and_does_not_mutate_target():
    row=transition()
    row.truncated=True
    row.metadata.update(remaining_music_seconds=10.,next_remaining_music_seconds=9.96)
    critic=ConstantCritic()
    a= fixed_targets([copy.deepcopy(row)],critic,'cpu',gamma_upper=.5,lambda_upper=.7)
    b= fixed_targets([copy.deepcopy(row)],critic,'cpu',gamma_upper=.99,lambda_upper=.95)
    gamma=.5**(1/25)
    assert a['returns'][0]==pytest.approx(.1+gamma*.2+gamma**2*2,abs=1e-7)
    assert abs(float(a['returns'][0]-b['returns'][0]))>1e-3
    assert not a['returns'].requires_grad


@pytest.mark.parametrize('bad',[0,-1,float('nan')])
def test_critic_clip_validation(bad):
    critic=ConstantCritic()
    with pytest.raises(ValueError,match='clip'):
        critic_update(critic,torch.optim.AdamW(critic.parameters()),[],{},grad_clip_norm=bad)


def test_config_accepts_real_hyperparameters_and_rejects_subset(tmp_path):
    original=entry.ROOT/'configs/closedloop/stage10_prepare_server1.yaml'
    config=yaml.safe_load(original.read_text())
    config['stage10']['training'].update(gamma_upper=.98,lambda_upper=.9,critic_grad_clip_norm=.5)
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(config))
    parsed=entry.configuration(path)
    assert parsed['stage9']['gamma_upper']==.98
    assert parsed['stage9']['bc_prefix']['max_frames']==30
    config['stage10']['dataset']['selection_path']='obsolete-selection.json'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError,match='subset'):
        entry.configuration(path)


def test_resume_budget_cannot_rewind_or_expand_limits():
    a={'limits':{'optimizer_attempts':9},'used':{'optimizer_attempts':3}}
    entry.validate_resume_budget(a,{'limits':a['limits'],'used':{'optimizer_attempts':4}})
    with pytest.raises(ValueError):
        entry.validate_resume_budget(a,{'limits':a['limits'],'used':{'optimizer_attempts':2}})
    with pytest.raises(ValueError):
        entry.validate_resume_budget(a,{'limits':{'optimizer_attempts':10},'used':a['used']})
