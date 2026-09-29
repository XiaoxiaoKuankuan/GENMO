"""离线审计器测试：临时制造简短证据并逐项注入错误，不能替代真实Isaac结果。

覆盖暂停/延迟模式、25/100宏步聚合、实际读取窗口的逐帧plan归属、初末历史计数、
源支持halo和terminal后继续执行拒绝。所有JSONL只写pytest临时目录，不启动服务。
"""
import copy
import hashlib
import json

import numpy as np
import pytest

from tools.eval.audit_closedloop_baseline import audit_run


def evidence(tmp_path, mode="paused"):
    ep = tmp_path / "episodes" / "0"
    ep.mkdir(parents=True)
    initial = {"episode_id": "ep1", "env_id": 0, "tick": 0, "plan_id": "bootstrap",
               "gmt_history_update_count": 1, "history_append_count": 1}
    (ep / "initial.json").write_text(json.dumps({"snapshot": initial}))
    def reference():
        return {"joint_pos": [0.]*21, "joint_vel": [0.]*21, "body_pos_w": [[0.]*3]*22,
                "body_quat_w": [[1., 0., 0., 0.]]*22, "body_lin_vel_w": [[0.]*3]*22,
                "body_ang_vel_w": [[0.]*3]*22}
    rows = []
    for tick in range(12, 301, 12):
        ticks = list(range(tick-132, tick+109, 12))
        rows.append({"episode_id": "ep1", "env_id": 0, "tick": tick, "control_tick_begin": tick-12,
                     "reference_tick": tick, "reference_plan_id": "new" if tick > 132 else "bootstrap",
                     "consumed_reference_ticks": ticks,
                     "consumed_plan_ids": ["new" if t > 132 else "bootstrap" for t in ticks],
                     "reference": reference(), "actual_qpos": [0.]*28, "phase": "music", "decision_id": 0})
    def advance(begin,end):
        n=(end-begin)//12
        return {"event": "advance", "episode_id": "ep1", "begin_tick": begin, "end_tick": end,
                "executed_control_steps": n, "requested_control_steps": n, "executed_physics_steps": n*4,
                "gmt_history_update_count": end//12+1, "proprio_history_update_count": None,
                "phase": "music", "decision_id": 0, "done": False}
    prepared = {"event": "plan_prepared", "episode_id": "ep1", "request_tick": 0, "deadline_tick": 12,
                "prefix_frames": 12, "protected_end_tick": 132, "parent_plan_id": "bootstrap", "plan_id": "new"}
    committed = {"event": "plan_committed", "episode_id": "ep1", "request_tick": 0,
                 "deadline_tick": 12, "arrival_tick": 0 if mode=="paused" else 12,
                 "effective_tick": 0 if mode=="paused" else 12, "end_to_end_seconds": .01, "plan_id": "new",
                 "acknowledgement": {"episode_id": "ep1", "plan_id": "new", "parent_plan_id": "bootstrap",
                    "protected_modification_count": 0, "protected_end_tick": 132,
                    "max_position_error": 0, "max_velocity_error": 0}}
    events = [prepared,committed,advance(0,300)] if mode=="paused" else [prepared,advance(0,12),committed,advance(12,300)]
    terminal={**initial,"tick":300,"gmt_history_update_count":26,"history_append_count":26,"done":False}
    summary={"episode_id":"ep1","mode":mode,"dataset":"Mine","artifacts":{"trace":"episodes/0/trace.jsonl"},
             "terminal_snapshot":terminal,"executed_control_steps":25,"executed_physics_steps":100}
    def save():
        (ep/'trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (tmp_path/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
        (tmp_path/'report.json').write_text(json.dumps({'episodes':[summary]}))
    save()
    return rows, events, summary, save


@pytest.mark.parametrize("mode", ["paused", "latency"])
def test_valid_logs_macro_counts_and_missing_event_boundary(tmp_path,mode):
    evidence(tmp_path,mode)
    result=audit_run(tmp_path)
    assert result['status']=='passed'
    ep=result['episodes'][0]
    assert ep['counts']['complete_upper_intervals_25_100']==1
    assert ep['counts']['control_steps']==25 and ep['counts']['physics_steps']==100
    assert ep['warnings'] and 'event field is null' in ep['warnings'][0]


@pytest.mark.parametrize("damage", ["owner", "tick", "history", "halo", "protected", "latency", "parent", "extra_trace"])
def test_corrupted_evidence_fails(tmp_path,damage):
    rows, events, summary, save=evidence(tmp_path)
    if damage=='owner': rows[0]['consumed_plan_ids'][0]='new'
    elif damage=='tick': rows[2]['control_tick_begin']=0
    elif damage=='history':summary['terminal_snapshot']['history_append_count']=27
    elif damage=='halo':events[0]['prefix_frames']=6
    elif damage=='protected':events[1]['acknowledgement']['protected_modification_count']=1
    elif damage=='latency':events[1]['arrival_tick']=12
    elif damage=='parent':events[0]['parent_plan_id']='older_plan'
    elif damage=='extra_trace':rows.append(copy.deepcopy(rows[-1]))
    save()
    assert audit_run(tmp_path)['status']=='failed'


def test_terminal_transition_cannot_continue(tmp_path):
    rows,events,summary,save=evidence(tmp_path,mode='latency')
    events[1]['done']=True
    summary['terminal_snapshot']['done']=True
    save()
    assert audit_run(tmp_path)['status']=='failed'


def add_visual_physics(rows):
    """造出同 tick 的 22-link 诊断；它仅校验审计器，不是渲染实验。"""
    names=['base_link','l_ankle_roll_link','r_ankle_roll_link']+[f'body_{i}' for i in range(19)]
    joint_names=[f'joint_{i}' for i in range(21)]
    feet=names[1:3]
    camera={'status':'available','world_transform':np.eye(4).tolist(),
            'focal_length':24.,'horizontal_aperture':20.,'vertical_aperture':15.}
    for i,row in enumerate(rows):
        pos=np.zeros((22,3)); pos[:,2]=.1; pos[0,0]=i*.002
        quat=np.zeros((22,4)); quat[:,0]=1.
        all_force=np.zeros((22,3)); all_force[1:3,2]=100.+i
        row['actual_joint_pos_gmt']=[i*.01]*21
        row['actual_qpos']=[i*.002,0.,.1,1.,0.,0.,0.]+[i*.01]*21
        row['physical_diagnostics']={
            'control_tick':row['tick'],'env_index':0,'body_names':names.copy(),'joint_names':joint_names.copy(),
            'body_link_pos_w':pos.tolist(),'body_link_quat_w':quat.tolist(),
            'foot_body_names':feet.copy(),'contact_body_names':names.copy(),
            'net_contact_forces_w_n':all_force.tolist(),'foot_net_contact_forces_w_n':all_force[1:3].tolist(),
            'foot_min_support_clearance_m':[.001,.002],
            'foot_support_sphere_clearance_m':{feet[0]:[.001]*8,feet[1]:[.002]*8},
            'contact_is_ground_only':False,'applied_joint_torque_nm':[.5]*21,'computed_joint_torque_nm':[1.]*21}
        poses=np.c_[pos,quat].tolist()
        row['render_pose_sync']={
            'status':'passed','control_tick':row['tick'],'physics_steps_added':0,'history_updates_added':0,
            'body_names':names.copy(),'physics_body_pose_wxyz':copy.deepcopy(poses),'usd_body_pose_wxyz':copy.deepcopy(poses),
            'max_position_error_m':0.,'max_orientation_error_rad':0.,
            'render_sync_state':{'route':'usd','fabric_enabled':False,'update_to_usd':True},
            'camera':copy.deepcopy(camera)}
        row['video_frame_index']=i
        row['video_frame_sha256']=hashlib.sha256(str(i).encode()).hexdigest()


def test_visual_physics_passed_and_implicit_torque_estimate_semantics(tmp_path):
    rows,_,_,save=evidence(tmp_path)
    add_visual_physics(rows); save()
    result=audit_run(tmp_path)
    assert result['status']=='passed'
    assert result['visual_evidence']['status']=='passed'
    ep=result['episodes'][0]
    assert ep['visual_evidence']['verified_frames']==25
    assert ep['visual_evidence']['unique_rgb_frame_hashes']==25
    physics=ep['physical_diagnostics']
    assert physics['status']=='passed'
    assert 'Implicit PD' in physics['torque_semantics']
    assert physics['phases']['warmup']['frames']==0
    music=physics['phases']['music']
    assert music['frames']==25
    np.testing.assert_allclose(music['actual_joint_motion']['peak_to_peak_rad'],.24)
    assert music['foot_net_contact_force_z_n'][0]['max']==124.
    assert music['foot_min_support_clearance_m'][1]['min']==.002
    assert music['applied_joint_torque_estimate_abs_nm']['max']==.5


def test_old_numeric_evidence_never_claims_visual_validation(tmp_path):
    rows,_,_,save=evidence(tmp_path)
    result=audit_run(tmp_path)
    assert result['status']=='passed'
    assert result['visual_evidence']['status']=='not_recorded'
    for i,row in enumerate(rows):row['video_frame_index']=i
    save()
    result=audit_run(tmp_path)
    assert result['status']=='passed'
    assert result['visual_evidence']['status']=='failed'
    assert result['episodes'][0]['visual_evidence']['status']=='incomplete'


@pytest.mark.parametrize('damage',['camera','hash','frame_gap','render_tick','physical_tick','body_count','stale_usd','false_maximum','torque_nan','capture_mismatch','sync_route'])
def test_visual_diagnostic_errors_do_not_relabel_protocol_evidence(tmp_path,damage):
    rows,_,_,save=evidence(tmp_path)
    add_visual_physics(rows)
    row=rows[3]
    if damage=='camera':row['render_pose_sync']['camera']['world_transform'][3][0]=.1
    elif damage=='hash':row['video_frame_sha256']='not-a-hash'
    elif damage=='frame_gap':row['video_frame_index']=8
    elif damage=='render_tick':row['render_pose_sync']['control_tick']+=12
    elif damage=='physical_tick':row['physical_diagnostics']['control_tick']+=12
    elif damage=='body_count':row['render_pose_sync']['body_names'].pop()
    elif damage=='stale_usd':row['render_pose_sync']['usd_body_pose_wxyz'][2][2]+=.18
    elif damage=='false_maximum':row['render_pose_sync']['max_position_error_m']=.00001
    elif damage=='torque_nan':row['physical_diagnostics']['computed_joint_torque_nm'][0]=float('nan')
    elif damage=='capture_mismatch':row['physical_diagnostics']['body_link_pos_w'][0][2]+=.01
    elif damage=='sync_route':row['render_pose_sync']['render_sync_state']['update_to_usd']=False
    save()
    result=audit_run(tmp_path)
    assert result['status']=='passed'
    assert result['visual_evidence']['status']=='failed'
    assert result['episodes'][0]['visual_evidence']['issues']
    if damage=='torque_nan':
        physical=result['episodes'][0]['physical_diagnostics']
        assert physical['status']=='failed'
        assert physical['phases']['music']['frames']==24


def test_physics_phase_summaries_exclude_warmup_from_music(tmp_path):
    from tools.eval.audit_closedloop_baseline import _audit_render_physics
    rows,_,summary,save=evidence(tmp_path)
    add_visual_physics(rows)
    for row in rows[:5]:row['phase']='warmup'
    save()
    result=_audit_render_physics(tmp_path,summary)
    phase=result['physical_diagnostics']['phases']
    assert phase['warmup']['frames']==5 and phase['music']['frames']==20
    assert phase['warmup']['foot_net_contact_force_z_n'][0]['max']==104.
    assert phase['music']['foot_net_contact_force_z_n'][0]['min']==105.
