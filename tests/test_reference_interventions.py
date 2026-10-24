"""验证实验C跨计划参考连续性及原30→50Hz重采样下的边界。

纯轨迹测试反复把已提交前缀接入下一滚动块，检查前缀严格不变、速度/加速度增量
受限，覆盖高度、三轴姿态和腿幅。不启动物理仿真，不把FK合法性当作跟踪收益。
"""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from gem.closedloop.dppo.reference_interventions import desired_delta,bridge_delta,apply_delta,check_proposal,project_free_joints

P=Path(__file__).resolve().parents[2]/'legged_lab_gmt/scripts/motion/bumi4340_reference_kinematics.py'
spec=importlib.util.spec_from_file_location('reference_resample_test',P);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


class Kinematics:
    joint_limits=torch.tensor([[-1.,1.]]*21)
    def forward_body_positions(self,q):
        result=q[:,:3,None].transpose(1,2).repeat(1,22,1)
        result[:,1:,2]+=q[:,7:]*.05
        return result


@pytest.mark.parametrize('variant',['height_plus','yaw_plus','pitch_plus','roll_plus','leg_minus','smooth05'])
def test_repeated_plan_prefix_and_consumed_continuity(variant):
    baseline=np.zeros((400,28));baseline[:,3]=1;baseline[:,7:]=.2
    committed=baseline.copy();standing=baseline[0].copy();standing[7:]=0
    names=[f'leg_{i}' for i in range(21)];kin=Kinematics()
    for start in range(0,270,9):
        base=baseline[start:start+60];prefix=committed[start:start+10].copy()
        desired=desired_delta(base,600+start*20,variant,standing,names)
        proposal=apply_delta(base,bridge_delta(desired,base,prefix));proposal[:10]=prefix
        free=np.ones((60,30),bool);free[:10]=False
        legal,metrics=check_proposal(base,proposal,free,kin,mod.resample)
        assert legal,metrics
        np.testing.assert_array_equal(proposal[:10],prefix)
        committed[start:start+60]=proposal
    actual=mod.resample(committed[:270]);delta=actual[:,:3]-mod.resample(baseline[:270])[:,:3]
    assert np.max(np.abs(np.diff(delta,axis=0)*50))<.25
    assert np.max(np.abs(np.diff(delta,n=2,axis=0)*2500))<12


def test_hard_limit_or_large_splice_is_rejected():
    base=np.zeros((60,28));base[:,3]=1;new=base.copy();new[20:,7]=1.1
    free=np.ones((60,30),bool);free[:10]=False
    legal,_=check_proposal(base,new,free,Kinematics(),mod.resample)
    assert not legal
    new=base.copy();new[20:,2]=.1
    legal,_=check_proposal(base,new,free,Kinematics(),mod.resample)
    assert not legal


def test_joint_projection_anticipates_limit_without_clipping_kink():
    base=np.zeros((60,28));base[:,3]=1.;base[:,7]=.98;base[30,7]=1.03
    free=np.ones((60,30),bool);free[:10]=False
    new,count=project_free_joints(base,base.copy(),free,Kinematics.joint_limits.numpy())
    assert count==1 and new[10:,7].max()<=1.
    np.testing.assert_array_equal(new[:10],base[:10])
    legal,metrics=check_proposal(base,new,free,Kinematics(),mod.resample)
    assert legal,metrics


def test_immutable_prefix_is_not_optimized_and_global_audit_still_catches_old_jump():
    from tools.summarize_decisive_physics import reference_continuity
    base=np.zeros((60,28));base[:,3]=1.;proposal=base.copy();proposal[:4,7]=.04
    free=np.ones((60,30),bool);free[:10]=False;free[9,:2]=True
    new,_=project_free_joints(base,proposal,free,Kinematics.joint_limits.numpy())
    np.testing.assert_array_equal(new[:10],proposal[:10])
    legal,_=check_proposal(base,new,free,Kinematics(),mod.resample)
    assert legal  # 本次自由区与接缝合法，历史跳变仍不能从前缀中改掉。
    a,b=mod.resample(base),mod.resample(new)
    audit=reference_continuity({('task',i*12):q for i,q in enumerate(a)},
        {('task',i*12):q for i,q in enumerate(b)},Kinematics())
    assert audit['maxima']['joint_velocity_delta_max']>.6
