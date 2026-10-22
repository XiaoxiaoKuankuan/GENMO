"""保存连续滚动参考接管的紧凑故障证据，不复制网络、采样链或物理大数组。

每个环境只在内存保留最近32次生成、提交和执行结束事件，包含真实音乐位置、
计划身份、旧参考有效范围、预计与实际到达、拒绝原因和提交后的权威范围。
拒绝、参考耗尽和异常时原子写出当前episode的最近事件；普通成功推进不写盘。
因此即使完整MC episode尚未结束、训练随机链使用临时GPU缓存，也能追溯为何
仍在执行旧Motion。本模块只记录，不能延长参考、修改奖励、重试采样或接受候选。
"""
from .budget import atomic_json


def record_reference_event(env, event, **values):
    if not env.rolling:
        return
    snapshot=env.snapshot or {}
    identity=snapshot.get('episode_id')
    if getattr(env,'_reference_event_episode',None)!=identity:
        env._reference_event_episode=identity
        env._reference_events=[]
    state={key:snapshot[key] for key in ('tick','plan_id','source_end_tick',
        'reference_valid_end_tick','read_watermark_tick','executed_end_tick') if key in snapshot}
    entry=dict(event=event,decision=env.decision,reference=state,**values)
    env._reference_events=(getattr(env,'_reference_events',[])+[entry])[-32:]


def flush_reference_events(env, reason):
    events=getattr(env,'_reference_events',[])
    if not events:
        return
    sample=getattr(env,'sample',{})
    payload=dict(schema='genmo.reference_lifecycle.v1',reason=reason,
        env_id=env.snapshot.get('env_id'),episode_id=env.snapshot.get('episode_id'),
        dataset=sample.get('dataset'),sample_id=sample.get('row',{}).get('sample_id'),
        seed=getattr(env,'seed',None),music_start_frame=getattr(env,'music_start_frame',None),
        scope='last_32_events_not_complete_execution_trace',events=events)
    atomic_json(env.output/f'reference_lifecycle_{env.episode_count:06d}.json',payload)
