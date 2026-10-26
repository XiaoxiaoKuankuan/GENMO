"""零Adapter真实回放的固定音乐标尺重算与一次性复用。

基线采集时s0尚未知，因此先记录真实状态/动作/原SM奖励和完整episode；只用训练歌曲
真实qd的二阶矩确定固定s0，再沿已发生的执行序列重建因果音乐记忆与奖励。控制动作、
物理状态、名义参考和SM奖励不变，绝不使用预测状态补标签。验证歌曲只用于MC/WM验证，
不进入s0。产物可供正式预热复用，避免为相同零策略重复跑全部物理歌曲。
"""

from dataclasses import replace

import torch

from .contracts import DT
from .music_reward import MusicState


@torch.no_grad()
def reprice_zero_replay(replay, music, music_weight):
    for key, rows in replay.episodes.items():
        meta = replay.episode_metadata[key]
        if not meta["true_start"] or meta["head_truncated"]:
            raise ValueError("零基线重算要求真实reset起点，不能恢复缺失音乐历史")
        if any(r.policy_version != 0 for r in rows):
            raise ValueError("只有零Adapter动作与音乐标尺独立，不能重算非零策略回放冒充同分布")
        z = rows[0].start
        nominal_activity = z.music_state.nominal_activity[:, -1, None].expand(-1, 21)
        memory = music.push(MusicState.empty(z.executed_state), z.executed_state[:, 34:], nominal_activity, z.tick)
        memory = replace(memory, nominal_activity=z.music_state.nominal_activity)
        priced = []
        for row in rows:
            start = replace(row.start, music_state=memory)
            nominal_activity = row.end.music_state.nominal_activity[:, -1, None].expand(-1, 21)
            memory = music.push(memory, row.end.executed_state[:, 34:], nominal_activity, row.end.tick)
            memory = replace(memory, nominal_activity=row.end.music_state.nominal_activity)
            memory, reward, activity, info = music.settle(memory, row.end.audio)
            weighted = row.weighted | {"music": DT*music_weight*reward}
            raw = row.raw | {"music": reward, "activity": activity, **info}
            priced.append(replace(row, start=start, end=replace(row.end, music_state=memory),
                                  raw=raw, weighted=weighted, reward=sum(weighted.values())))
        replay.episodes[key] = priced
    replay._windows.clear()


def combined_s0(baselines):
    """按真实训练样本数合并未截断二阶矩，不平均已clamp的s0。"""
    if not baselines or any(b.get("source") != "zero_adapter_real_training_episodes" for b in baselines):
        raise ValueError("音乐统计来源必须是真实零Adapter训练集")
    names = baselines[0]["joint_names"]
    if any(tuple(b["joint_names"]) != tuple(names) for b in baselines):
        raise ValueError("音乐标尺关节顺序不一致")
    count = sum(b["samples"] for b in baselines)
    if count <= 0:
        raise ValueError("缺少训练集速度样本")
    square = sum(b["sum_square_qd"] for b in baselines)
    return (square/count).sqrt().clamp_min(.05)
