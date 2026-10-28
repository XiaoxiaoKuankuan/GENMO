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


def test_control_configuration_content_changes_cannot_reuse_identity(tmp_path):
    from gem.closedloop.awr.config import configuration_assets
    first, second = tmp_path / "agent.yaml", tmp_path / "isaac.json"
    first.write_text("control: frozen\n")
    second.write_text('{"dt": 0.005}\n')
    config = dict(paths=dict(gmt_agent_config=str(first), isaac_contract=str(second)))
    identity = configuration_assets(config)
    assert configuration_assets(config, expected=identity) == identity
    first.write_text("control: changed\n")
    with pytest.raises(ValueError, match="contents changed"):
        configuration_assets(config, expected=identity)
from gem.closedloop.awr.io import cpu_copy, read_tree, write_tree


def test_original_physical_diagnostics_are_required(tmp_path):
    root = Path(__file__).resolve().parents[3]
    original = yaml.safe_load((root / "tests/closedloop/fixtures/dppo_contract.yaml").read_text())
    awr = load_config(root / "tests/closedloop/fixtures/awr_contract.yaml")
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


def test_plan_id_object_strings_are_binary_and_exact(tmp_path):
    plans = np.asarray(["bootstrap:e:0", "生成计划:1"], dtype=object)
    write_tree(tmp_path / "plans", {"origins": plans})
    restored = read_tree(tmp_path / "plans")["origins"]
    assert restored.dtype == plans.dtype
    np.testing.assert_array_equal(restored, plans)
    with pytest.raises(TypeError, match="Object arrays"):
        write_tree(tmp_path / "unsupported", {"objects": np.asarray([object()], dtype=object)})


def test_columnar_physics_keeps_exact_rows_without_expanding_25_copies(tmp_path):
    from gem.runtime.trajectory_blocks import pack_trace, ColumnarTrace
    original = [dict(tick=600+12*i, body=np.arange(42, dtype=np.float32).reshape(14, 3)+i,
                     nested=dict(force=np.full((2, 3), i, dtype=np.float64)),
                     source="original_physics", valid=True) for i in range(25)]
    block = pack_trace(original)
    rows = list(ColumnarTrace([block]))
    copied = cpu_copy(rows)
    block["columns"]["fields"]["body"]["values"].fill(-1)
    write_tree(tmp_path / "columns", copied)
    restored = read_tree(tmp_path / "columns")
    for expected, actual in zip(original, restored):
        assert actual["tick"] == expected["tick"] and actual["source"] == expected["source"]
        np.testing.assert_array_equal(actual["body"], expected["body"])
        np.testing.assert_array_equal(actual["nested"]["force"], expected["nested"]["force"])
        assert not actual["body"].flags.writeable
    assert len({id(row._fields) for row in restored}) == 1
    import json
    manifest = json.loads((tmp_path / "columns/manifest.json").read_text())
    assert len(manifest["column_fields"]) == 1
    assert manifest["schema"] == "genmo.awr.array_tree.v2"


def test_current_consumption_and_reward_reference_are_distinct(episode_factory):
    from gem.closedloop.awr.collection import finalize_episode
    _, _, make = episode_factory
    episode = make(lengths=(2, 1))
    tick = 600
    for row in episode.rows:
        row.metadata.update(consumed_plan_first_gmt_input_tick={"p0": 600, "p1": 624}, execution_trace=[],
                            published=dict(effective_tick=row.control_tick_begin))
        for _ in range(row.executed_control_steps):
            plans = ["p0"] * 11+["p0" if tick < 624 else "p1"]*10
            plans[10] = "bootstrap:test" if tick == 600 else "p0"
            row.metadata["execution_trace"].append(dict(consumed_plan_ids=plans,
                consumed_reference_ticks=np.arange(tick-120, tick+121, 12), control_tick_begin=tick,
                tick=tick+12, episode_id=row.identity["episode_id"],
                reference_plan_id="p0" if tick < 624 else "p1", reference_tick=tick+12))
            tick += 12
    task = dict(sample=dict(dataset="source", group_id="g0", row=dict(source_audio_sha256="audio-g0")))
    finalized = finalize_episode(episode.rows, task, episode.contract)
    assert finalized.rows[0].execution["actually_consumed_seconds"] == .04
    assert finalized.rows[1].execution["actually_consumed_seconds"] == 0.
    assert finalized.rows[0].execution["reward_reference_ticks"].tolist() == [612, 624]
    assert finalized.rows[1].execution["reward_reference_ticks"].tolist() == [636]


def test_physical_metrics_preserve_original_units_and_each_control():
    from gem.closedloop.awr.physical_metrics import PhysicalMetrics, summarize_metrics
    from gem.closedloop.common.rewards import TRACKING_FUNCTIONS
    primitive = dict(track=dict(normalized=dict(scores={k: .75 for k in TRACKING_FUNCTIONS})),
        contact=dict(normalized=dict(slide=.2, bad_contact=.1), raw=dict(samples=[
            dict(foot_net_contact_forces_w_n=[[0, 0, 10], [0, 0, 0]], foot_contact_force_threshold_n=1.,
                 slide_proxy=dict(left=[.2, .4]), support_selected=dict(left=[True, False]))])))
    trace = [dict(tick=612+12*i, actual_qpos=[i*.1, 0, 1.], reward_primitives=dict(components=primitive),
                  errors=dict(root_position_error_m=.3, joint_position_rmse_rad=.1)) for i in range(2)]
    diagnostics = PhysicalMetrics()
    diagnostics.observe({}, dict(metadata=dict(generated=dict(qpos_world=np.zeros((60, 28))), execution_trace=trace)))
    report = summarize_metrics([dict(awr_physical_metrics=diagnostics.result())])
    assert report["actual_control_samples"] == 2
    assert set(report["tracking_eight_mean"].values()) == {.75}
    assert report["supported_sphere_speed_mean_m_s"] == .2
    assert report["foot_contact_fraction"] == .5
    assert report["actual_root_path_length_m_per_episode"] == [.1]
