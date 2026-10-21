"""检查CFG训练/采样一致性、冻结教师约束和无活动门控奖励。

使用真实小型Stage1网络，在相同x_t、t、条件下独立计算两个分支及CFG组合，检查
Replay梯度确实通过两分支、前缀与padding隔离、原Stage1监督行为不变。奖励夹具
比较同一真实字段格式的标量/列式路径，只删除活动乘数，其余项逐项保持一致。
所有测试是数学和接口回归，不能替代s595000更新及PhysX完整音乐验收。
"""
import copy
import pytest
import torch

from tests.closedloop.test_stage1_actor import actor_factory, _conditions
from gem.closedloop.awr.losses import weighted_denoising
from gem.closedloop.common.rewards import ExecutionReward, NO_ACTIVITY_REWARD_VERSION


def test_guided_training_is_deployed_predictor_and_has_both_gradients(actor_factory):
    actor, batch = actor_factory(motion_frames=60)
    conditions = _conditions(batch)
    times = torch.tensor([23, 654]); noise = torch.randn_like(batch["target_qpos30"])
    out = actor.denoising_forward(conditions, batch["target_qpos30"], batch["target_qpos30_valid"],
                                 timesteps=times, noise=noise, guidance_scale=2.5)
    adapted = actor.adapt_conditions(conditions)
    cond = actor._denoise(out["xt"], times, adapted, actor.encode_conditions(adapted))
    uncond = actor._denoise(out["xt"], times, adapted,
        actor.encode_conditions(adapted, torch.ones(2, dtype=torch.bool)))
    expected = actor._constrain(2.5*cond["pred_x_start"]-1.5*uncond["pred_x_start"], adapted)
    torch.testing.assert_close(out["pred_x_start"], expected, atol=1e-6, rtol=1e-6)
    out["conditional_x_start"].retain_grad(); out["unconditional_x_start"].retain_grad()
    mask = batch["target_qpos30_valid"] & ~conditions["known_qpos30_mask"]
    weighted_denoising(out["pred_x_start"], out["target_x_start"], mask, torch.ones(2))[0].backward()
    first, second = out["conditional_x_start"].grad, out["unconditional_x_start"].grad
    assert first[mask].norm() > 0 and second[mask].norm() > 0
    torch.testing.assert_close(first * -1.5, second * 2.5)
    assert (first[~mask] == 0).all() and (second[~mask] == 0).all()
    with pytest.raises(ValueError, match="drop music"):
        actor.denoising_forward(conditions, batch["target_qpos30"], batch["target_qpos30_valid"],
                                 guidance_scale=2.5, music_dropout=True)


def test_frozen_reference_survives_resume_and_does_not_replace_replay(episode_factory):
    from gem.closedloop.awr.dataset import ReplayDataset
    from gem.closedloop.awr.distributed import Distributed
    from gem.closedloop.awr.trainer import ActorTrainer
    from tests.closedloop.awr.test_training import ArithmeticBC
    actor, _, make = episode_factory
    data = ReplayDataset([make()], unit_weights=True)
    config = dict(epochs=5, global_batch=2, microbatch=1, bc_global_batch=2, bc_microbatch=1,
                  lr=1e-5, warmup_steps=1, reference_weight=2., max_optimizer_steps=2)
    trainer = ActorTrainer(actor, data, ArithmeticBC(), config, Distributed(), bc_weight=.1)
    initial = copy.deepcopy(actor.state_dict())
    trainer.backward_awr([0, 1], step=0)
    assert trainer.last_reference_loss == 0.
    assert any(p.grad is not None and p.grad.norm() > 0 for p in actor.parameters())
    assert all(p.grad is None and not p.requires_grad for p in trainer.reference_actor.parameters())
    trainer.optimizer.zero_grad(set_to_none=True)
    saved = []
    trainer.run(on_step=lambda current, _: saved.append(current.state_dict()) if current.steps == 1 else None)
    expected = copy.deepcopy(actor.state_dict())
    assert any(not torch.equal(expected[k], v) for k, v in initial.items())
    resumed = ActorTrainer(actor, data, ArithmeticBC(), config, Distributed(), bc_weight=.1)
    resumed.load_state_dict(saved[0]); resumed.run()
    for key, value in expected.items():
        torch.testing.assert_close(actor.state_dict()[key], value, atol=0, rtol=0)
        torch.testing.assert_close(resumed.reference_actor.state_dict()[key], initial[key], atol=0, rtol=0)


def test_no_activity_multiplier_scalar_and_columnar():
    from tests.closedloop.dppo.test_data_learning import actual_step, target_activity_fixture
    from gem.closedloop.common.vector_reward_math import VECTOR_REWARD_VERSION, reward_config_sha
    from gem.closedloop.common.vector_reward_adapter import VectorExecutionReward, compare_reward_evidence
    from gem.runtime.trajectory_blocks import pack_trace, ColumnarTrace
    new = dict(version=NO_ACTIVITY_REWARD_VERSION)
    old = ExecutionReward(target_activity=target_activity_fixture)
    scalar = ExecutionReward(new, target_activity=target_activity_fixture)
    oracle = ExecutionReward(new, target_activity=target_activity_fixture)
    vector = VectorExecutionReward(new, target_activity=target_activity_fixture)
    rows, expected = [], []
    for i in range(30):
        row = actual_step(i, speed=.01); row["env_id"] = 3
        left, right = old.evaluate_step(row), scalar.evaluate_step(row)
        assert 0 <= left["components"]["track"]["gate"] < 1.
        assert right["components"]["track"]["gate"] == 1.
        assert right["activity"] == left["activity"]
        delta = .02 * 2.5 * right["components"]["track"]["score"] * (1-left["components"]["track"]["gate"])
        assert right["reward"]-left["reward"] == pytest.approx(delta, abs=1e-14)
        for key in left["components"].keys()-{"track"}:
            compare_reward_evidence(right["components"][key], left["components"][key])
        evidence = oracle.evaluate_step(row)
        row["reward_primitives"] = dict(schema=VECTOR_REWARD_VERSION,
            config_sha256=reward_config_sha(oracle.config), env_id=row["env_id"],
            episode_id=row["episode_id"], tick=row["tick"], components={
                key: {field: evidence["components"][key][field] for field in ("score", "valid", "raw", "normalized")}
                for key in ("track", "stable", "alive", "cmd", "torque", "contact", "joint_limit")})
        rows.append(row); expected.append(right)
    actual = vector.evaluate_batch(ColumnarTrace([pack_trace(rows[:25]), pack_trace(rows[25:])]))
    for left, right in zip(actual, expected):
        compare_reward_evidence(left, right)
    assert vector.scalar_audits == 1 and vector.columnar_fallbacks == 0
    from gem.closedloop.common.evaluation import _decision_evidence
    evaluated = _decision_evidence(dict(rewards=[r["reward"] for r in actual],
        executed_control_steps=len(actual), metadata=dict(reward_details=actual,
        latency_seconds=.1, generated=dict(prefix_frames=17, generation_horizon_frames=60,
                                           joint_limit_diagnostics={}))), 0)
    assert evaluated["reward_sum"] == pytest.approx(sum(r["reward"] for r in expected))


def test_retention_accepts_unchanged_but_rejects_more_violent_motion():
    from gem.closedloop.awr.retention import retention_comparison
    task = dict(task_id="one", source="s", group_id="g", seconds=10., reward_sum=20.,
                physical_failure=False, generated=dict(generated_joint_speed_rms_rad_s=1.,
                generated_joint_acceleration_rms_rad_s2=10., generated_joint_jerk_rms_rad_s3=100.))
    report = dict(plan_identity="p", physics_identity="world", sampler_identity="ddim",
        task_summaries=[task], aggregate=dict(overall=dict(music_completion_count=1,
        physical_failure_count=0, executed_seconds=10., reward_sum=20.,
        activity=dict(actual_activity_rad_s=dict(mean=1.)))), physical_metrics=dict(overall=dict(
        original_errors_mean=dict(root_position_error_m=.1, joint_position_rmse_rad=.1))))
    limits = dict(min_duration_ratio=.99, min_return_ratio=.99, min_activity_ratio=.9,
                  max_motion_ratio=1.1, max_error_ratio=1.1)
    assert retention_comparison(report, report, limits)["accepted"]
    changed = copy.deepcopy(report)
    changed["task_summaries"][0]["generated"]["generated_joint_acceleration_rms_rad_s2"] *= 1.2
    assert not retention_comparison(report, changed, limits)["accepted"]
    changed = copy.deepcopy(report); changed["task_summaries"][0]["physical_failure"] = True
    assert not retention_comparison(report, changed, limits)["accepted"]


def test_new_run_rejects_training_cfg_mismatch(tmp_path):
    from pathlib import Path
    import yaml
    from gem.closedloop.awr.config import load_config
    path = Path(__file__).resolve().parents[3]/"configs/closedloop/stage2_awr.yaml"
    config = load_config(path)
    config["awr"]["actor"]["denoising_guidance_scale"] = 1.
    wrong = tmp_path/"wrong.yaml"; wrong.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="CFG"):
        load_config(wrong)


def test_registered_episode_receipt_skips_full_rescan_but_read_still_checks(tmp_path, episode_factory, monkeypatch):
    from gem.closedloop.awr.replay import EpisodeReplay
    import gem.closedloop.awr.replay as module
    _, _, make = episode_factory
    ep = make(); replay = EpisodeReplay(tmp_path/"replay", capacity=2, contract=ep.contract)
    publication = replay.publish(ep); assert replay.register(publication)
    def forbidden(*args, **kwargs):
        raise RuntimeError("full payload read")
    monkeypatch.setattr(module, "read_tree", forbidden)
    assert replay.register(publication) is False
    with pytest.raises(RuntimeError, match="full payload read"):
        replay.read(ep.episode_id)
    replay.close()
