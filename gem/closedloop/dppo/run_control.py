"""显式策略构造与按累计 PPO 迭代停止的运行合同。

训练不设总墙钟期限；新运行与恢复都使用已接受的累计 PPO 外层迭代数。
调用方仍独立维护信号停止、checkpoint、资源预算及真实物理执行的时间约束。
策略构造从实际训练配置显式传入扩散步数、eta、CFG 与精度参数，消除默认值漂移。
本模块不启动仿真、不分配数据预算，也不改变实际控制时钟。
"""
from __future__ import annotations

def validate_stopping_iteration(requested, stage):
    """显式停止轮数是恢复后的累计上限，不能重新获得一份配置预算。"""
    maximum = stage['run_control']['max_training_iterations']
    if (type(requested) is not int or not 1 <= requested <= maximum
            or requested > stage['limits']['accepted_iterations']):
        raise ValueError('Stopping iteration must fit the cumulative PPO iteration budget')
    return requested


def iteration_checkpoint_reason(iteration, storage, *, is_warmup=False, rejected=False):
    """仅在新接受的整百等指定轮次保存里程碑；预热及拒绝更新不重复保存旧轮次。"""
    interval=storage.get('checkpoint_interval_iterations')
    if interval and iteration>0 and not is_warmup and not rejected and iteration%interval==0:
        return f'iteration_{iteration:06d}'
    return None


def build_policy(actor, settings, precision, *, frozen_actor=None):
    from .frozen_prefix import FrozenPrefixPolicy
    cls=FrozenPrefixPolicy
    extra=dict(frozen_prefix_steps=15,frozen_actor=frozen_actor)
    policy=cls(actor, **extra, std_schedule=settings['std_schedule'], steps=settings['denoising_steps'],
        eta=settings['eta'], guidance_scale=settings['guidance_scale'], precision=precision['mode'],
        attention=precision['attention'], linear_execution=precision['linear_execution'],
        compile_denoiser=precision['compile'], precise_output=precision['precise_output'],
        cuda_graph=precision.get('cuda_graph',False),
        weight_gradient=precision.get('weight_gradient', 'direct_joint_gemm'),generation_frames=settings.get('generation_frames',120))
    from .actor_scope import configure_actor_scope
    policy.actor_scope=configure_actor_scope(actor,settings.get('actor_trainable_scope','all'))
    return policy
