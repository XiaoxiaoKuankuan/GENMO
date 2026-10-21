"""离线有限复验的数据顺序、物理身份与先保留后强化门禁测试。

只用CPU人工身份检查复用64集时不会按来源重排/补样，不会把不同奖励或部署时钟
对应的交叉优势标签用于训练，以及生成/物理任一门禁失败时不得进入AWR。真实模型
和PhysX表现由独立服务器有限实验验证，本文件不把接口测试称作物理质量证明。
"""
import copy

import pytest

from tools.diagnose_stage2_awr_distillation import (
    refit_snapshot_entries, require_unweighted_refit, validate_refit_contract,
)


def test_refit_uses_complete_snapshot_order():
    rows = [("a", "source-z", "sha-a"), ("b", "source-a", "sha-b"), ("c", "source-a", "sha-c")]
    result = refit_snapshot_entries(dict(episode_ids=["b", "a"]), rows)
    assert result == [dict(id="b", source="source-a", sha="sha-b"), dict(id="a", source="source-z", sha="sha-a")]
    for ids in ([], ["a", "a"], ["missing"], [str(i) for i in range(65)]):
        with pytest.raises(ValueError):
            refit_snapshot_entries(dict(episode_ids=ids), rows)


def test_refit_rejects_mismatched_physical_labels():
    config = dict(assets={"actor": "immutable"}, reward={"version": "no_activity"},
                  environment={"control": 50}, termination={"kind": "physical"},
                  timing={"deployment_profile": {"samples_seconds": [.32]}},
                  awr={"configuration_assets": {"controller": "frozen"}})
    sampler = {"guidance_scale": 2.5}
    contract = {key: copy.deepcopy(config[key]) for key in ("assets", "reward", "environment", "termination", "timing")}
    contract.update(schema="genmo.awr.replay_physics.v1", horizon_frames=60,
                    configuration_assets=config["awr"]["configuration_assets"], sampler=sampler,
                    source_commits={"genmo": "original-collection-source"})
    validate_refit_contract(contract, config, sampler)
    for key in ("assets", "reward", "environment", "termination", "timing", "configuration_assets", "sampler", "horizon_frames"):
        changed = copy.deepcopy(contract); changed[key] = "different"
        with pytest.raises(ValueError, match=key):
            validate_refit_contract(changed, config, sampler)
    assert contract["source_commits"]["genmo"] == "original-collection-source"


@pytest.mark.parametrize("generation,physical", [(False, True), (True, False), (False, False), (None, True), (True, None)])
def test_refit_blocks_awr_without_both_retention_results(generation, physical):
    with pytest.raises(ValueError, match="before AWR"):
        require_unweighted_refit(dict(generation_probe_passed=generation), dict(accepted=physical))
    require_unweighted_refit(dict(generation_probe_passed=True), dict(accepted=True))
