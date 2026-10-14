"""用可复算的状态反例检查当前 Stage10 条件能否表达根朝向跟踪偏差。

本工具不运行训练，不修改参考锚点，也不伪造真实仿真数据。它构造三组直立静止
机器人状态：相对同一旧参考的 yaw 偏差分别为 0、1.4、1.6 rad。沿世界竖直轴
旋转不改变本体重力投影；角速度、关节位置和关节速度保持相同，所以当前后端
的 48 维 proprio 历史相同。随后调用仓库真实 OnlineConditionBuilder，逐字段
比较 Actor/Critic 使用的十个条件。最后调用配套 GMT 的真实姿态误差公式，
证明相同网络输入可以对应不同的当前朝向误差和终止判定。

这只是观测信息是否充分的反例，不是三条真实执行轨迹，不证明本次所有失败
均由该缺项造成。0 与 1.4 rad 都在现行 1.5 rad 阈值以内，已足以说明失败前
存在状态混叠；1.6 rad 仅用于核对终止语义。Critic 在相同剩余时长下必然给出
相同确定性预测，但真实回报差异仍应以独立完整歌曲轨迹衡量。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import torch

from gem.closedloop.online_conditions import OnlineConditionBuilder
from gem.robots.bumi.feature_codec import BumiMotionFeatureCodec
from gem.robots.bumi.kinematics import BumiKinematics
from gem.utils.rotation_conversions import axis_angle_to_quaternion, quaternion_apply, quaternion_invert


def probe(gmt_root: Path):
    orientation_path = gmt_root / ('source/NoetixRobot/NoetixRobot/tasks/mimic/'
        'mimic_noetix_bumi4340_mha_sonic/closedloop/orientation_errors.py')
    spec = importlib.util.spec_from_file_location('stage10_orientation_probe', orientation_path)
    orientation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orientation)
    codec = BumiMotionFeatureCodec(BumiKinematics(
        REPO / 'configs/bumi/bumi_kinematics_robot_retargeter_fe934_v1.json'))
    builder = OnlineConditionBuilder(codec, motion_frames=60)
    reference = codec.kinematics.default_qpos.repeat(17, 1)
    reference[:, 3:7] = torch.tensor([1., 0., 0., 0.])
    reservation = dict(source_qpos=reference.numpy(), source_ticks=600+np.arange(17)*20,
        reference_anchor_qpos=reference[0].numpy(), prefix_frames=17, decision_tick=600,
        generation_horizon_frames=60, env_id=0, episode_id='representation_probe')
    music = torch.arange(120*35, dtype=torch.float32).reshape(120, 35)/4200
    conditions, results = [], []
    for yaw in (0., 1.4, 1.6):
        actual = reference[0].clone()
        actual[3:7] = axis_angle_to_quaternion(torch.tensor([0., 0., yaw]))
        gravity = quaternion_apply(quaternion_invert(actual[3:7]), torch.tensor([0., 0., -1.]))
        proprio = torch.cat((gravity, torch.zeros(3+21+21)))
        snapshot = dict(tick=600, env_id=0, episode_id='representation_probe', seed=20261014,
            history_values=proprio.repeat(50, 1).numpy(), history_valid=np.ones(50, dtype=bool),
            history_ticks=600-np.arange(49, -1, -1)*12, actual_qpos=actual.numpy())
        condition, _ = builder.build(snapshot, reservation, music)
        non_yaw, error = orientation.separated_orientation_errors(reference[0, 3:7], actual[3:7])
        conditions.append(condition)
        results.append(dict(actual_yaw_rad=yaw, projected_gravity_b=gravity.tolist(),
            yaw_tracking_error_rad=float(error), non_yaw_tracking_error_rad=float(non_yaw),
            threshold_rad=1.5, would_yaw_terminate=bool(error>1.5)))
    equal = {key: all(torch.equal(conditions[0][key], value[key]) for value in conditions[1:])
        for key in conditions[0]}
    if not all(equal.values()) or [r['would_yaw_terminate'] for r in results] != [False, False, True]:
        raise AssertionError('The current representation or termination semantics changed')
    files = [REPO/'gem/closedloop/online_conditions.py', REPO/'gem/robots/bumi/feature_codec.py',
        REPO/'gem/closedloop/dppo/pretrained_critic.py', orientation_path,
        orientation_path.with_name('vector_env.py')]
    return dict(schema='genmo.stage10.condition_observability_probe.v1', status='passed',
        scope='constructed_representation_counterexample_not_physical_rollout', states=results,
        all_ten_condition_fields_identical=all(equal.values()), field_equality=equal,
        same_remaining_music_seconds=True,
        conclusion='current_inputs_do_not_identify_absolute_reference_relative_yaw_drift',
        limitation='does_not_quantify_causal_share_of_observed_training_failures',
        sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in files})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gmt-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.gmt_root.resolve())
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
    print(json.dumps(dict(status=result['status'], all_inputs_identical=result['all_ten_condition_fields_identical'],
        yaw_errors=[row['yaw_tracking_error_rad'] for row in result['states']]), ensure_ascii=False))


if __name__ == '__main__':
    main()
