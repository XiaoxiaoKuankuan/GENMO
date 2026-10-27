"""episode累计音乐目标的固定坐标合同与执行积分尺度。

episode初始真实状态只计算一次SE(2)变换，独立存放在闭环快照中；后续真实接管点
只复用该变换，不能随着学生累计偏移重新对齐。局部H16目标仍保留自身anchor对齐，
两种目标分别输出。教师、初态和变换全部detach，不成为GENMO未来真实状态输入。
递归传播仅针对已经存在的请求快照，不引入未来观测；缺少初态合同显式报错。
"""
from dataclasses import replace
from .dance_losses import AnchorAlignment


def with_episode_alignment(state, alignment):
    pending=state.pending_request
    if pending is not None:
        pending=replace(pending,snapshot=with_episode_alignment(pending.snapshot,alignment))
    return replace(state,episode_teacher_rotation=alignment.rotation.detach().clone(),
        episode_teacher_translation=alignment.translation.detach().clone(),pending_request=pending)


def episode_alignment(state, required=False):
    if state.episode_teacher_rotation is None or state.episode_teacher_translation is None:
        if required:raise ValueError("episode累计目标缺少初态固定SE2，禁止按当前anchor补造")
        return None
    return AnchorAlignment(state.episode_teacher_rotation.detach(),state.episode_teacher_translation.detach())


def execution_mass(engine, config):
    return sum(.02*engine.gamma**i for i in range(engine.horizon_steps)) if config.get("normalize_execution_horizon",False) else 1.
