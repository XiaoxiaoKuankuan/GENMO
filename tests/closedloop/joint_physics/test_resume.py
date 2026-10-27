"""完整联合训练断点的下一批次/下一更新复现测试。

使用既有Stage1生成的显式临时数据与缩小的原Transformer，真实执行本训练器、AdamW和
原损失；保存全部游标/RNG后比较不中断与恢复的第二次更新。此处只验证恢复协议，真实
完整架构/原配对数据的恢复另在Server-1有界验收，不把小网络当作正式资产结果。
"""

import copy

from omegaconf import OmegaConf
import torch

from gem.closedloop.joint_physics import config, data
from gem.closedloop.joint_physics.checkpoint import save_trainer
from gem.closedloop.joint_physics.trainer import JointTrainer
from tests.closedloop.test_stage1_dataset import _write_dataset, KINEMATICS_PATH


def test_checkpoint_reproduces_next_batch_and_update(tmp_path, monkeypatch):
    root = _write_dataset(tmp_path/"data", length=180)
    cfg = config.read_config(config.ROOT/"configs/closedloop/joint_physics_data_only.yaml")
    cfg["stage1"]["model"].update(latent_dim=32, num_layers=2, num_heads=4, history_hidden_dim=24)
    cfg["training"].update(batch_size=2, max_steps=10, lr_warmup_steps=0, diagnostics_every=10)
    cfg["data"]["sidecar"] = None
    dc = OmegaConf.create({"datasets": {"train": {"fixture": {
        "_target_": "gem.closedloop.stage1_dataset.BumiClosedLoopStage1Dataset", "root": str(root),
        "dataset_name": "test_bumi", "kinematics_path": str(KINEMATICS_PATH), "split": "train", "motion_frames": 60,
        "prefix_min_frames": 12, "prefix_max_frames": 24, "history_steps": 50, "joint_limit_tolerance": .0001}}},
        "sample_contract": {"history_steps": 50}, "train_sampling_reference": {"test_bumi": 1.},
        "qpos30_stats": {"path": cfg["stage1"]["endecoder"]["stats_path"]}})
    monkeypatch.setattr(config, "data_config", lambda unused: dc)
    monkeypatch.setattr(data, "data_config", lambda unused: dc)
    first = JointTrainer(cfg, "cpu")
    first.update()
    path = tmp_path/"joint_00000001.pt"
    save_trainer(first, path)
    expected = first.update()
    expected_weights = copy.deepcopy(first.actor.state_dict())
    restored = JointTrainer(cfg, "cpu", resume=path)
    observed = restored.update()
    for key in ("sample_ids", "timesteps", "data_loss", "gradient_norm", "supervised_updates"):
        assert observed[key] == expected[key]
    for key, tensor in restored.actor.state_dict().items():
        torch.testing.assert_close(tensor, expected_weights[key], rtol=0, atol=0)
    assert {id(p) for group in restored.optimizer.param_groups for p in group["params"]} == {id(p) for p in restored.actor.parameters() if p.requires_grad}
    assert restored.counters["physics_updates"] == 0
