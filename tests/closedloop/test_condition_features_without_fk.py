"""在线前缀去除冗余FK后的精确数值与梯度回归，只在服务器1运行。

对照原完整encode，覆盖不同前缀长度、真实非单位输入四元数、批次、FP32/FP64，
要求条件和anchor逐元素相等，梯度保持原门槛。通过禁止调用FK的替身保证新入口
确实不重复计算机器人几何，而Stage1完整几何编码仍使用原实现。
"""
from pathlib import Path
import pytest
import torch
from gem.robots.bumi.feature_codec import BumiMotionFeatureCodec
from gem.robots.bumi.kinematics import BumiKinematics


@pytest.mark.parametrize('length',[1,6,18,30,60,100])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_condition_features_match_full_codec_exactly(length,dtype,monkeypatch):
    root=Path(__file__).resolve().parents[2]
    codec=BumiMotionFeatureCodec(BumiKinematics(root/'configs/bumi/bumi_kinematics_robot_retargeter_fe934_v1.json'))
    q=torch.randn(2,length,28,generator=torch.Generator().manual_seed(19),dtype=dtype,requires_grad=True)
    expected=codec.encode(q)
    gradient=torch.autograd.grad(expected.physical_features.square().sum(),q)[0]
    def forbidden(*args,**kwargs):raise AssertionError('Condition path must not compute unused FK')
    monkeypatch.setattr(codec.kinematics,'forward_kinematics',forbidden)
    features,anchor=codec.encode_condition_features(q)
    torch.testing.assert_close(features,expected.physical_features,atol=0,rtol=0)
    for name,value in vars(anchor).items():torch.testing.assert_close(value,getattr(expected.anchor,name),atol=0,rtol=0)
    actual=torch.autograd.grad(features.square().sum(),q)[0]
    torch.testing.assert_close(actual,gradient,atol=1e-7,rtol=1e-6)
