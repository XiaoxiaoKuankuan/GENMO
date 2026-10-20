"""AWR测试数据工厂：使用真实Stage1条件和明确合成的连续物理反馈。

仅缩小网络，保持60帧qpos30、50步历史与真实归一化接口；Tracking额外通道为
可识别的合成值，奖励与终止由测试显式指定。所有磁盘产物只写pytest临时目录。
"""
import copy
import pytest
import torch
from tests.closedloop.test_stage1_actor import actor_factory, _conditions
from tests.closedloop.dppo.test_asymmetric_critic import tracking_context
from gem.closedloop.awr.records import AWRTransition, AWRCompleteEpisode
from gem.closedloop.awr.returns import mc_returns


@pytest.fixture
def episode_factory(actor_factory):
    actor, batch = actor_factory(starts=(45,), motion_frames=60)
    context = _conditions(batch)
    critic = tracking_context(context)
    context = {k: critic[k] for k in context}

    def make(*, episode_id="e", group="g0", source="source", lengths=(2, 1), rewards=None, event=0.):
        rows, tick = [], 600
        for index, count in enumerate(lengths):
            c = {k: v.clone() for k, v in context.items()}
            privileged = {k: v.clone() for k, v in critic.items()}
            r = torch.tensor(rewards[index] if rewards is not None else [float(index+1)]*count, dtype=torch.float64)
            row = AWRTransition(
                identity=dict(backend_session_id="synthetic", env_id=0, episode_id=episode_id,
                              decision_id=index, plan_id=f"p{index}", policy_version=0, actor_fingerprint="test"),
                control_tick_begin=tick, control_tick_end=tick+12*count, executed_control_steps=count,
                rewards=r, terminated=index == len(lengths)-1, truncated=False,
                reason="music_end" if index == len(lengths)-1 else None,
                execution=dict(committed=True, observed_by_gmt=True, target_matches_commit=True),
                context=c, target_qpos30=batch["target_qpos30"][0].clone(),
                target_valid=batch["target_qpos30_valid"][0].clone(),
                free_mask=batch["target_qpos30_valid"][0] & ~c["known_qpos30_mask"][0],
                contact=torch.zeros(60, 2), actor_valid=True,
                metadata=dict(critic_context=privileged, remaining_music_seconds=5.-index*.5,
                              data_split="train", event_reward=event if index == len(lengths)-1 else 0.))
            rows.append(row); tick += 12*count
        episode = AWRCompleteEpisode(rows, source, group, "audio-"+group, {"reward": "test"}, {"task_id": episode_id})
        mc_returns(episode)
        return episode
    return actor, batch, make
