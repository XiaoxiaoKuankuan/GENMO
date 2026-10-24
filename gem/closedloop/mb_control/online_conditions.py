"""保留原Stage1十字段语义的纯Tensor在线条件构造，不经过旧CPU/detach路径。

历史严格来自ClosedLoopState的已发生记录，预测上下文另传；旧缓存前缀用原codec
编码并保持末端水平位移两坐标未知。音乐按30Hz读取，标签不进入条件。世界anchor
来自旧参考，后续多决策中anchor/前缀的输入导数也保留。
"""

import torch

from gem.closedloop.contracts import validate_stage1_condition_batch


def build_conditions(state, ticket, codec):
    state.validate(decision=True)
    if state.tick != ticket.request_tick:
        raise ValueError("必须使用请求时冻结快照构造条件")
    dev, batch = state.state.device, len(state.state)
    ticks = state.tick + torch.arange(60, device=dev, dtype=torch.long) * 20
    prefix = state.reference.source_at(ticks[:ticket.prefix_frames])
    encoded, anchor = codec.encode_condition_features(prefix)
    tail = encoded.new_zeros((batch, 60 - ticket.prefix_frames, 30))
    known = torch.cat((encoded, tail), 1)
    frame = torch.arange(60, device=dev)[None, :, None]
    coordinate = torch.arange(30, device=dev)[None, None, :]
    mask = ((frame < ticket.prefix_frames) &
            ~((frame == ticket.prefix_frames - 1) & (coordinate < 2))).expand_as(known)
    index = (ticks - state.music_start_tick) // 20
    music_valid = (index >= 0) & (index < state.music.shape[1])
    music = state.music[:, index.clamp(0, state.music.shape[1] - 1)]
    music = torch.where(music_valid[None, :, None], music, 0.)
    history_ticks = state.tick - torch.arange(49, -1, -1, device=dev) * 12
    conditions = {
        "music_features": music, "music_valid": music_valid[None].expand(batch, -1),
        "proprio_history": state.genmo_history, "proprio_history_valid": state.genmo_valid,
        "proprio_history_times": (history_ticks.double() / 600)[None].expand(batch, -1),
        "known_qpos30": torch.where(mask, known, 0.), "known_qpos30_mask": mask,
        "future_valid": torch.ones(batch, 60, device=dev, dtype=torch.bool),
        "future_times": (ticks.double() / 600)[None].expand(batch, -1),
        "decision_time": torch.full((batch,), state.tick / 600, dtype=torch.float64, device=dev),
    }
    validate_stage1_condition_batch(conditions)
    world_anchor = torch.cat((anchor.position_w[:, 0], anchor.yaw.reshape(batch, 1)), -1)
    return conditions, world_anchor
