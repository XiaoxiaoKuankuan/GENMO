#!/usr/bin/env python3
"""逐帧核验GENMO原始输出、GMT参考和蓝色影子姿态的数据一致性。

读取已有完整sim2sim运行的run_report.json、generated_plans.npz和control_trace.npz，
按原决定/提交tick重放权威ReferenceTimeline，核对每个原始计划SHA、前缀恢复及
最终参考。再独立按30Hz源帧做线性位置/关节插值和SciPy最短弧SLERP，按关节名
重排到MuJoCo；最后调用生产viewer的影子场景构造，比较qpos、各可见网格位置
和旋转矩阵，避免只用“赋值的是同一数组”代替整条链验证。

输出JSON结论和逐帧源/参考/影子qpos的NPZ。比较的是同一时刻的三维姿态和
网格变换，不把屏幕二维像素误称为关节数据；支持已记录到达时刻的迟到拒绝，
当前要求已有运行无参考保持。仅做CPU重放/FK，不加载模型、不执行动力学或训练，
不修改原始运行；新输出必须写入独立目录。容差只用于float32参考与float64
独立插值之间的舍入，任何关节顺序或轨迹内容不一致均应明确失败。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation,Slerp

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from gem.closedloop.sim2sim.viewer import Sim2SimViewer


def verify(run,output):
    run=Path(run);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    report=json.loads((run/'run_report.json').read_text());config=report['config']
    episode=report['episodes'][0]['report'];directory=Path(report['episodes'][0]['directory'])
    if episode.get('reference_fallback_steps',0):raise ValueError('当前重放工具不包含保持回退事件')
    with np.load(directory/'generated_plans.npz',allow_pickle=False) as data:
        plans={int(t):q.copy() for t,q,ok in zip(data['decision_tick'],data['qpos_world'],data['committed']) if ok}
        proposed={int(t):int(p) for t,p in zip(data['decision_tick'],data['prefix_frames'])}
    with np.load(directory/'control_trace.npz',allow_pickle=False) as data:
        ticks=data['tick'].copy();observed=data['reference_qpos'].copy()
    sys.path.insert(0,str(Path(config['paths']['gmt_repo'])/'scripts/motion'))
    from bumi4340_reference_kinematics import UrdfKinematics
    from closedloop_reference import ReferenceTimeline
    contract=json.loads(Path(config['paths']['isaac_contract']).read_text())
    kinematics=UrdfKinematics(config['paths']['urdf'],contract)
    source_names=json.loads(Path(config['paths']['kinematics']).read_text())['joint_order']
    model=mujoco.MjModel.from_xml_path(config['paths']['mjcf'])
    joint_ids=[j for j in range(model.njnt) if model.jnt_type[j]!=mujoco.mjtJoint.mjJNT_FREE]
    joint_names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j) for j in joint_ids]
    assert model.nq==28 and np.array_equal(model.jnt_qposadr[joint_ids],np.arange(7,28))
    source_to_mujoco=[source_names.index(name) for name in joint_names]
    initial=np.asarray(episode['events'][0]['initial_qpos'])
    initial_source=np.r_[initial[:7],initial[7:][[joint_names.index(n) for n in source_names]]]
    timeline=ReferenceTimeline(kinematics,initial_source,episode_id=episode['events'][0]['episode_id'],
                               source_joint_names=source_names,plan_frames=config['generation_frames'])
    view=Sim2SimViewer.__new__(Sim2SimViewer);view.model=model;view.data=mujoco.MjData(model)
    view.ref_data=mujoco.MjData(model);view.scene=mujoco.MjvScene(model,maxgeom=3000)
    view.ghost_scene=mujoco.MjvScene(model,maxgeom=1500);view.camera=mujoco.MjvCamera()
    view.options=mujoco.MjvOption();view.perturb=mujoco.MjvPerturb();view.show_reference=True
    expected_data=mujoco.MjData(model)
    by_decision={r['decision_tick']:r for r in episode['requests']}
    by_commit={r['commit_tick']:r for r in episode['requests']}
    rejected_arrivals={int(e['arrival_logical_seconds']*50)*12:e for e in episode['events']
        if e.get('event')=='reject_completed_late_generation'}
    rejected_by_decision={e['decision_tick']:e for e in rejected_arrivals.values()}
    reservations={};active=False;reconstructed=[];rendered=[];prefix_error=0.;hashes=0
    replay_error=0.;interpolation_error=0.;geom_position_error=0.;geom_rotation_error=0.
    isaac_to_mujoco=[contract['joint_names'].index(n) for n in joint_names]
    base=contract['body_names'].index('base_link');music_start=config['runtime']['warmup_control_steps']*12
    for tick,reference in zip(ticks,observed):
        current=int(tick)-12
        if current in by_commit:
            request=by_commit[current];raw=plans[request['decision_tick']]
            assert hashlib.sha256(raw.tobytes()).hexdigest()==request['generated_qpos_sha256'];hashes+=1
            meta=reservations[request['decision_tick']];p=meta['prefix_frames']
            prefix=raw[:p].astype(np.float64).copy()
            prefix[:,3:7]*=np.where(np.sum(prefix[:,3:7]*meta['source_qpos'][:,3:7],axis=-1)<0,-1.,1.)[:,None]
            prefix_error=max(prefix_error,float(np.max(np.abs(prefix-meta['source_qpos']))))
            ticket=timeline.prepare_plan(dict(meta,qpos_world=raw,plan_id=request['ack']['plan_id']))
            timeline.commit_plan(ticket['ticket_id'],current);active=False
        if current in rejected_arrivals:active=False
        if current>=music_start and (current-music_start)%300==0 and not active:
            request=by_decision.get(current,rejected_by_decision.get(current))
            deadline=request['deadline_tick'] if request else current+episode['calibration']['deadline_ticks']
            meta=timeline.reserve_prefix(request['request_id'] if request else 'unfinished_final_request',current,deadline,12)
            if current in proposed:assert meta['prefix_frames']==proposed[current]
            reservations[current]=meta;active=True
        timeline.commit_window(current)
        ref=timeline.read(int(tick))
        replay=np.r_[ref['body_pos_w'][0,base],ref['body_quat_w'][0,base],ref['joint_pos'][0,isaac_to_mujoco]]
        replay_error=max(replay_error,float(np.max(np.abs(replay-reference))))
        numerator=int(tick)-int(timeline.source_ticks[0]);index,rem=divmod(numerator,20);alpha=rem/20
        left,right=timeline.source_qpos[index:index+2]
        source=(1-alpha)*left+alpha*right
        rotations=Rotation.from_quat(np.stack([left[3:7],right[3:7]])[:,[1,2,3,0]])
        quat=Slerp([0.,1.],rotations)([alpha]).as_quat()[0][[3,0,1,2]]
        if np.dot(quat,reference[3:7])<0:quat=-quat
        independent=np.r_[source[:3],quat,source[7:][source_to_mujoco]]
        interpolation_error=max(interpolation_error,float(np.max(np.abs(independent-reference))))
        view.scene.ngeom=0;view._ghost({'reference_qpos':reference})
        rendered.append(view.ref_data.qpos.copy());reconstructed.append(independent)
        expected_data.qpos[:]=independent;mujoco.mj_forward(model,expected_data)
        for geom in view.scene.geoms[:view.scene.ngeom]:
            geom_position_error=max(geom_position_error,float(np.max(np.abs(geom.pos-expected_data.geom_xpos[geom.objid]))))
            geom_rotation_error=max(geom_rotation_error,float(np.max(np.abs(geom.mat.reshape(9)-expected_data.geom_xmat[geom.objid]))))
        if not active:timeline.trim(int(tick))
    rendered=np.array(rendered);reconstructed=np.array(reconstructed)
    result=dict(frames=len(ticks),plans_hashed=hashes,rejected_plans=len(rejected_by_decision),source_joint_names=source_names,mujoco_joint_names=joint_names,
        raw_prefix_restore_max_error=prefix_error,raw_plan_replay_vs_recorded_reference_max_error=replay_error,
        independent_interpolation_vs_reference_max_error=interpolation_error,
        viewer_qpos_vs_reference_max_error=float(np.max(np.abs(rendered-observed))),
        viewer_geom_position_vs_independent_fk_max_m=geom_position_error,
        viewer_geom_rotation_matrix_vs_independent_fk_max_error=geom_rotation_error,
        source=str(directory),scope='same-time pose/geom consistency; rendering pixels are not joint angles')
    result['passed']=bool(replay_error<=1e-6 and interpolation_error<=1e-6 and
        result['viewer_qpos_vs_reference_max_error']==0 and geom_position_error<=2e-6 and geom_rotation_error<=2e-6)
    np.savez_compressed(output/'pose_chain.npz',tick=ticks,source_interpolated_qpos=reconstructed,
                        reference_qpos=observed,viewer_qpos=rendered)
    (output/'consistency.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=verify(args.run,args.output)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(0 if result['passed'] else 1)
