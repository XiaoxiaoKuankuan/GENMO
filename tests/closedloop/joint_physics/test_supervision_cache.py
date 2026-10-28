"""CPU只读监督缓存的数学等价与可恢复读取回归。

使用真实codec/FK和临时配对数据，在起点、中段及右侧halo不足时核对全部条件、标签、
Stage1各项损失及学生预测梯度。缓存不含学生或物理模型输出，不运行Newton；临时文件
由pytest独立目录管理。另比较未缓存与持久缓存读取，确保窗口anchor没有变成全曲anchor。
"""
import torch

from gem.closedloop.joint_physics.supervision_cache import prepare_sequence
from gem.closedloop.joint_physics.data import _CachedStage1Dataset, collate_joint_samples
from gem.closedloop.losses import Stage1BumiLosses
from tests.closedloop.test_stage1_dataset import _write_dataset, KINEMATICS_PATH


def test_precomputed_windows_history_gt_loss_and_gradient(tmp_path, actor_factory, monkeypatch):
    actor, _ = actor_factory(motion_frames=60)
    root = _write_dataset(tmp_path/'cache_data', length=180)
    dataset = _CachedStage1Dataset(sequence_cache_bytes=10**8, root=str(root),
        dataset_name='test_bumi', split='train', kinematics_path=str(KINEMATICS_PATH),
        motion_frames=60, prefix_min_frames=12, prefix_max_frames=24, history_steps=50,
        joint_limit_tolerance=.0001)
    starts = (0, 3, 75, 177)
    expected = [dataset.get_window(0, f) for f in starts]
    sequence = prepare_sequence(dataset, 0, actor.endecoder, chunk_size=17)
    dataset.reader.prepared = (tmp_path, {})
    monkeypatch.setattr(dataset.reader, 'load_aligned_sequence', lambda row: sequence)
    actual = [dataset.get_window(0, f) for f in starts]
    for first, second in zip(expected, actual):
        for key, value in first.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, second[key], atol=1e-6, rtol=1e-6)
            else:
                assert value == second[key]
    old, new = collate_joint_samples(expected), collate_joint_samples(actual)
    assert '_target_body' not in old and '_target_body' in new
    from tests.closedloop.test_stage1_losses import _weights
    objective = Stage1BumiLosses(actor.endecoder, _weights())
    pred = torch.randn(4,60,30, requires_grad=True)
    contact = torch.randn(4,60,2, requires_grad=True)
    loss1, terms1 = objective(old, pred, contact, 20000)
    grad1 = torch.autograd.grad(loss1, (pred, contact), retain_graph=True)
    loss2, terms2 = objective(new, pred, contact, 20000)
    grad2 = torch.autograd.grad(loss2, (pred, contact))
    for key in terms1:
        torch.testing.assert_close(terms1[key], terms2[key], atol=2e-6, rtol=2e-6)
    for first, second in zip(grad1, grad2):
        torch.testing.assert_close(first, second, atol=2e-6, rtol=2e-6)
