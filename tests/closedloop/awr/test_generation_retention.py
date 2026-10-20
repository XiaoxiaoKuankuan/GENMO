"""验证固定生成保留指标只使用有效未知坐标和真实Contact标签。

合成已知/未知坐标、padding和Contact真值，检查已知前缀不增加未知生成误差，
对比Contact BCE来自原标签而非生成概率；测试不冒称真实模型或物理评价结果。
"""
import importlib.util
import math
from pathlib import Path

import pytest
import torch


def test_generation_retention_masks_and_true_contact():
    path = Path(__file__).resolve().parents[3]/"tools"/"validate_stage2_awr_generation.py"
    spec = importlib.util.spec_from_file_location("generation_retention", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    known = torch.zeros(1, 60, 30, dtype=torch.bool)
    known[:, :10] = True
    valid = torch.ones_like(known)
    valid[:, 50:] = False
    physical = torch.zeros(1, 60, 30)
    physical[:, :10] = 100.
    batch = dict(known_qpos30_mask=known, target_qpos30_valid=valid,
        target_qpos30=torch.zeros_like(physical), known_qpos30=physical.clone(),
        target_contact=torch.ones(1, 60, 2),
        target_contact_valid=torch.ones(1, 60, 2, dtype=torch.bool))
    result = dict(qpos30=physical, normalized=physical.clone(),
                  contact_logits=torch.full((1, 60, 2), 2.),
                  contact=torch.zeros(1, 60, 2))  # 故意与真值不同，禁止用这个输出作标签。
    result["normalized"][:, 50:] = 999.
    metrics = tool.clip_metrics(batch, result, torch.zeros_like(physical))
    assert metrics["normalized_unknown_qpos30_mse"] == 0.
    assert metrics["unknown_joint_rmse_rad"] == 0.
    assert metrics["unknown_joint_velocity_rms_rad_s"] == 0.
    assert metrics["true_contact_bce"] == pytest.approx(math.log1p(math.exp(-2)), abs=1e-7)
    result["qpos30"][:, :10] = 101.
    with pytest.raises(ValueError, match="known physical prefix"):
        tool.clip_metrics(batch, result, torch.zeros_like(physical))
