"""独立评估控制侧车中的 anchor 姿态取证，不参与物理状态或学习计算。

输入是 GMT 已完成控制步的 execution_trace.motion_tracking：同一份跟踪诊断
已经保存参考/实际 anchor 的世界系位置与 wxyz 四元数。本模块只核对来源 schema、
时间戳、坐标约定、数组形状、有限性和单位四元数，再复制四个小数组进入评估证据。
不重新查询仿真、不重采样、不归一化姿态，也不改观测、动作、奖励、终止或执行时钟。

anchor 名称必须随证据保留，不能把任意 anchor 擅称机器人 root。新字段沿用既有
无损控制侧车编码；旧侧车不迁移、不重写。读取缺失字段时显式返回 unknown，不能
将未记录误读为零误差。损坏或时间错位的姿态只标记 invalid 并保留校验原因，
不新增控制/评估终止条件；后处理归因必须另行要求证据有效，不补造取证数据。
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

SCHEMA = 'genmo.stage2.evaluation_anchor_pose.v1'
SOURCE_SCHEMA = 'gmt.motion_tracking.eight_terms.v1'
FIELD = 'anchor_pose_evidence'


def _unknown(reason, **diagnostics):
    return dict(schema=SCHEMA, available=False, reason=reason, **diagnostics)


def _validated_copy(pose, tick):
    if not isinstance(pose, Mapping) or pose.get('schema') != SCHEMA:
        raise ValueError('Unsupported evaluation anchor pose schema')
    if pose.get('available') is False:
        if not isinstance(pose.get('reason'), str) or not pose['reason']:
            raise ValueError('Unavailable anchor pose requires an explicit reason')
        return _unknown(pose['reason'], **({key: pose[key] for key in ('validation_error',)
                                           if key in pose}))
    if pose.get('available') is not True or pose.get('source_schema') != SOURCE_SCHEMA:
        raise ValueError('Evaluation anchor pose source is invalid')
    if pose.get('coordinate_frame') != 'world' or pose.get('quaternion_order') != 'wxyz':
        raise ValueError('Evaluation anchor pose coordinate convention differs')
    if not isinstance(pose.get('anchor_body_name'), str) or not pose['anchor_body_name']:
        raise ValueError('Evaluation anchor pose requires the recorded body name')
    for name, value in (('control tick', tick), ('control_tick', pose.get('control_tick')),
                        ('reference_tick', pose.get('reference_tick'))):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) != tick:
            raise ValueError(f'Evaluation anchor pose {name} does not match the control')
    result = {key: pose[key] for key in ('schema', 'available', 'source_schema',
        'coordinate_frame', 'quaternion_order', 'anchor_body_name')}
    result.update(control_tick=int(tick), reference_tick=int(tick))
    for side in ('reference', 'actual'):
        source = pose.get(side)
        if not isinstance(source, Mapping):
            raise ValueError(f'Evaluation anchor pose lacks {side}')
        result[side] = {}
        for name, shape in (('anchor_pos_w', (3,)), ('anchor_quat_w', (4,))):
            value = np.asarray(source.get(name))
            if value.shape != shape or value.dtype.kind not in 'fiu' or not np.isfinite(value).all():
                raise ValueError(f'Evaluation anchor pose {side}.{name} is not a finite {shape} array; '
                                 f'observed shape={value.shape}, dtype={value.dtype}')
            if name == 'anchor_quat_w':
                norm = float(np.linalg.norm(value))
                if not np.isclose(norm, 1., atol=1e-4, rtol=0.):
                    raise ValueError(f'Evaluation anchor pose {side} quaternion is not unit length: norm={norm}')
            result[side][name] = value.copy()
    return result


def _extract(trace, tick):
    if trace is None:
        return _unknown('execution_trace_not_recorded_unknown')
    if not isinstance(trace, Mapping):
        raise ValueError('Evaluation anchor pose requires a control trace mapping')
    if trace.get('tick') != tick:
        raise ValueError('Evaluation anchor pose trace tick does not match the reward')
    tracking = trace.get('motion_tracking')
    if tracking is None:
        return _unknown('motion_tracking_not_recorded_unknown')
    if not isinstance(tracking, Mapping):
        raise ValueError('Evaluation anchor pose tracking record is invalid')
    if 'reference' not in tracking and 'actual' not in tracking:
        return _unknown('anchor_pose_not_recorded_unknown')
    pose = dict(schema=SCHEMA, available=True, source_schema=tracking.get('schema'))
    pose.update({key: tracking.get(key) for key in ('coordinate_frame', 'quaternion_order',
        'anchor_body_name', 'control_tick', 'reference_tick', 'reference', 'actual')})
    return _validated_copy(pose, tick)


def extract_anchor_pose_evidence(trace, tick, *, trace_error=None):
    """复制原始 anchor；校验失败仅记录原因，绝不新增物理或评估终止条件。"""
    if trace_error is not None:
        return _unknown('invalid_execution_trace', validation_error=trace_error)
    try:
        return _extract(trace, tick)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        return _unknown('invalid_anchor_pose', validation_error=str(error))


def read_anchor_pose_evidence(control):
    """旧控制侧车缺字段时显式 unknown；原控制字典和历史文件保持不变。"""
    if FIELD not in control:
        return _unknown('not_recorded_unknown')
    try:
        return _validated_copy(control[FIELD], control['tick'])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        return _unknown('invalid_anchor_pose', validation_error=str(error))
