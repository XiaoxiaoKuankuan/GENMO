"""验证AWR物理配置和列式回执存储的边界。

从已批准基线配置读取实际诊断开关，检查AWR保留GMT原接触/跟踪子步来源；
删除关键开关应在创建PhysX前拒绝。列式回执以Mapping接口进入独立存储，
验证复制、二进制往返及原数据隔离，不将其当成PPO概率链记录。
"""
from collections import UserDict
from pathlib import Path
import numpy as np
import pytest
import yaml

from gem.closedloop.awr.config import load_config, physical_config
from gem.closedloop.awr.io import cpu_copy, read_tree, write_tree


def test_original_physical_diagnostics_are_required(tmp_path):
    root = Path(__file__).resolve().parents[3]
    original = yaml.safe_load((root / "configs/closedloop/stage2_full_chain.yaml").read_text())
    awr = load_config(root / "configs/closedloop/stage2_awr.yaml")
    assert physical_config(awr, run_id="contract")["diagnostics"] == original["diagnostics"]
    del awr["diagnostics"]["reward_substeps"]
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(yaml.safe_dump(awr))
    with pytest.raises(ValueError, match="substep diagnostics"):
        load_config(invalid)


def test_lazy_mapping_is_copied_and_roundtrips(tmp_path):
    original = UserDict({"control": UserDict({"qpos": np.arange(30, dtype=np.float64)})})
    copied = cpu_copy(original)
    original["control"]["qpos"][0] = -1
    assert copied["control"]["qpos"][0] == 0
    write_tree(tmp_path / "episode", original)
    restored = read_tree(tmp_path / "episode")
    np.testing.assert_array_equal(restored["control"]["qpos"], original["control"]["qpos"])
