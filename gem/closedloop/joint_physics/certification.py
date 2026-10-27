"""跨来源、分运动/真实接触类型的Newton局部方向认证。

H16/H32/H64使用同一最大H合法起点、同一学生预测和扰动；结构性缺帧在预测前报告，
不延长末端参考。各方向比较autograd、有限差分、重放同起点的Newton正负扰动、状态误差
和真实接触变化。GT接口认证显式标注，不算学生训练；学生认证保留checkpoint有效更新数。
类型证书只从实际成功的同H物理轨迹建立数值范围，未通过类型不能用于训练。来源ID只
作追溯；来源允许集合由原train目录另行控制，绝不把val/test混入训练。
"""

from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path

import torch

from gem.closedloop.stage1_dataset import collate_stage1_training_samples
from gem.closedloop.training import batch_to_device
from gem.closedloop.mb_control.validation import state_envelope
from gem.closedloop.mb_control.model_audit import state_errors
from .checkpoint import read_checkpoint
from .contracts import PhysicsRejected, execution_contract
from .data import SourceCatalog, SidecarStore
from .evaluation import load_actor, replay_to_start, sample_seed
from .execution import execute_window
from .prepare_rollouts import asset_contract
from .reference import attach_sidecars
from .requests import PhysicsRequestStream, CLASSIFICATION, motion_tags


def perturbation(prediction, name, window, actor):
    direction = torch.zeros_like(prediction)
    p = window.ticket.prefix_frames
    phase = torch.sin(torch.linspace(0,torch.pi,60-p,device=prediction.device)).square()
    names = actor.stage1.endecoder.kinematics.joint_order
    if name in ('root_x','root_y'):
        direction[:,p:,0 if name=='root_x' else 1] = phase
    elif name == 'root_yaw':
        from gem.utils.rotation_conversions import rotation_6d_to_matrix, matrix_to_rotation_6d
        physical = actor.stage1.endecoder.denormalize(prediction.detach())
        r = rotation_6d_to_matrix(physical[...,3:9])
        generator = r.new_tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,0.]])
        delta = physical.new_zeros(physical.shape)
        delta[...,3:9] = matrix_to_rotation_6d(generator@r)
        # normalize的线性缩放决定归一化空间中的真正yaw切线，不复制stats规则。
        tangent = actor.stage1.endecoder.normalize(physical+delta)-actor.stage1.endecoder.normalize(physical)
        direction[:,p:] = tangent[:,p:]*phase[None,:,None]
    else:
        if name=='fastest_joint':
            physical = actor.stage1.endecoder.denormalize(prediction.detach())
            j = int((physical[:,p+1:,9:]-physical[:,p:-1,9:]).abs().mean((0,1)).argmax())
            indices = [9+j]
        else:
            indices = [9+i for i,n in enumerate(names) if
                       (name=='upper_body' and ('arm' in n or 'elbow' in n)) or
                       (name=='left_leg' and n.startswith('l_') and any(s in n for s in ('leg','knee','ankle'))) or
                       (name=='right_leg' and n.startswith('r_') and any(s in n for s in ('leg','knee','ankle')))]
        if not indices:
            raise ValueError(f'方向{name}没有权威关节映射')
        direction[:,p:,indices] = phase[:,None]/len(indices)**.5
    return direction


def audit(cfg, checkpoint, output, adapter, env, interface_probe=False):
    if env is None:
        raise ValueError('分层认证必须实际运行Newton')
    if checkpoint is None and not interface_probe:
        raise ValueError('学生认证需要本流程checkpoint')
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    actor,_ = load_actor(cfg,checkpoint,adapter.runtime.kin.device)
    state = read_checkpoint(checkpoint) if checkpoint else None
    frozen = adapter.bundle.frozen_snapshot()
    assets = asset_contract(cfg,adapter)
    catalog = SourceCatalog(cfg,'train')
    store = SidecarStore(cfg['data']['sidecar'],'train',assets,sample_contract=cfg['data']['sample_contract'])
    stream = PhysicsRequestStream(catalog,store,adapter.bundle.spec.joint_names,max(cfg['audit']['horizons']),cfg['training']['seed'])
    (output/'input_manifest.json').write_text(json.dumps({'config':cfg,'inventory':stream.inventory},ensure_ascii=False,indent=2))
    rows, certificates, supported = [], {}, []
    collected = defaultdict(lambda: {'states':[],'actions':[],'rows':[]})
    profiles = defaultdict(lambda: {'states':[],'actions':[],'rows':[]})
    all_states,all_actions,source_ids = [],[],set()
    for key in stream.keys[:cfg['evaluation']['max_sources']]:
        dataset,row_index = stream.sources[key]
        items = stream.requests[key]
        requested_profiles = cfg['audit'].get('profiles')
        if requested_profiles is not None:
            items = [item for item in items if '|'.join(item['tags']) in requested_profiles]
        if not items:
            continue
        # 仅按真实离线类别与音乐时间选择，绝不读取学生预测或方向结果进行挑选。
        selected, covered, covered_profiles = [],set(),set()
        for item in sorted(items,key=lambda item:item['frame']):
            profile = '|'.join(item['tags'])
            if profile not in covered_profiles:
                selected.append(item);covered.update(item['tags']);covered_profiles.add(profile)
            if len(selected)>=cfg['audit']['anchors_per_source']:
                break
        if len(selected)<cfg['audit']['anchors_per_source']:
            for fraction in (.15,.5,.85):
                item = items[min(len(items)-1,int(len(items)*fraction))]
                if item not in selected:
                    selected.append(item)
                if len(selected)>=cfg['audit']['anchors_per_source']:
                    break
        episode = store.episode(dataset.get_window(row_index,0)['meta'])
        source_ids.add(key)
        for item in selected:
            batch = batch_to_device(collate_stage1_training_samples([dataset.get_window(row_index,item['frame'])]),env.device)
            batch,contexts,windows,reasons = attach_sidecars(batch,store,adapter,actor.stage1.endecoder.codec,max(cfg['audit']['horizons']))
            window = windows[0]
            if window is None:
                rows.append({'source':key,'frame':item['frame'],'rejection':reasons[0]});continue
            generator = torch.Generator().manual_seed(sample_seed(batch['meta'][0],cfg['evaluation']['seed']))
            noise = torch.randn(1,60,30,generator=generator).to(env.device)
            for k in cfg['audit']['timesteps']:
                out = actor.denoise_supervised(batch,contexts,torch.tensor([k],device=env.device),noise,
                                              torch.zeros(1,dtype=torch.bool,device=env.device))
                prediction = (actor.stage1.endecoder.normalize(batch['target_qpos30']).detach().requires_grad_()
                              if interface_probe else out['pred_x_start'])
                for horizon in cfg['audit']['horizons']:
                    selected_window = replace(window,horizon=horizon)
                    start = window.start.tick//12
                    tags = motion_tags(episode['state'][start:start+horizon+1],adapter.bundle.spec.joint_names,
                                       episode['contacts'][start:start+horizon])
                    base = {'source':key,'dataset':item['dataset'],'frame':item['frame'],'horizon':horizon,
                            'timestep':k,'tags':tags,'prediction_origin':'gt_interface_only' if interface_probe else 'student_denoised_x0'}
                    try:
                        result = execute_window(selected_window,prediction,out['static_conf_logits'],actor,adapter,cfg['execution_loss'])
                        grad, = torch.autograd.grad(result['loss'],prediction,retain_graph=True)
                        if not torch.isfinite(grad).all() or not bool(grad.norm()>0):
                            raise PhysicsRejected('zero_or_nonfinite_execution_gradient')
                        if not interface_probe:
                            from .trainer import grad_summary
                            base['execution_parameter_gradients'] = grad_summary(result['loss'],actor)
                    except PhysicsRejected as error:
                        rows.append(dict(base,rejection=error.reason,details=error.details));continue
                    for direction_name in cfg['audit']['directions']:
                        record = dict(base,direction=direction_name,coverage=result['coverage'],x0_gradient_norm=float(grad.norm()))
                        try:
                            direction = perturbation(prediction,direction_name,window,actor)
                            automatic = float((grad*direction).sum())
                            values,physical,repeats,contacts = [],[],[],[]
                            eps = cfg['audit']['epsilon']
                            with torch.no_grad():
                                for sign in (1,-1):
                                    candidate = prediction+sign*eps*direction
                                    model = execute_window(selected_window,candidate,out['static_conf_logits'],actor,adapter,cfg['execution_loss'])
                                    values.append(float(model['loss']))
                                    repeats.append(replay_to_start(env,episode,selected_window))
                                    trace = []
                                    def transition(action):
                                        value = env.step_physics(action)
                                        trace.append(env.contact_events())
                                        return value
                                    physical.append(execute_window(selected_window,candidate,out['static_conf_logits'],actor,adapter,
                                                                   cfg['execution_loss'],transition=transition))
                                    contacts.append(trace)
                            fd = (values[0]-values[1])/(2*eps)
                            pd = (float(physical[0]['loss'])-float(physical[1]['loss']))/(2*eps)
                            errors = state_errors(result['states'].flatten(0,1),((physical[0]['states']+physical[1]['states'])/2).flatten(0,1))
                            record.update(autograd=automatic,finite_difference=fd,newton_derivative=pd,
                                fd_pass=abs(fd-automatic)<=cfg['audit']['fd_atol']+cfg['audit']['fd_rtol']*abs(fd),
                                newton_direction_pass=abs(pd)>cfg['audit']['direction_min_abs'] and automatic*pd>0,
                                model_accuracy_pass=all(float(errors[n])<=bound for n,bound in cfg['audit']['state_rmse_limits'].items()),
                                model_errors={n:float(v) for n,v in errors.items()},same_start_max_abs=max(repeats),
                                real_contacts=contacts,
                                contact_changed_steps=sum(a['pair_signature']!=b['pair_signature'] for a,b in zip(*contacts)))
                            good = record['fd_pass'] and record['newton_direction_pass'] and record['model_accuracy_pass'] and max(repeats)<=cfg['audit']['same_start_atol']
                            if good:
                                targets = [collected[(horizon,tag)] for tag in tags]
                                targets.append(profiles[(horizon,'|'.join(tags))])
                                for target in targets:
                                    target['states'].append(selected_window.start.state.detach().cpu())
                                    target['states'].extend(p['states'].detach().cpu().flatten(0,1) for p in physical)
                                    target['actions'].extend(p['actions'].detach().cpu().flatten(0,1) for p in physical)
                                    target['rows'].append(record)
                                all_states.extend(p['states'].detach().cpu().flatten(0,1) for p in physical)
                                all_states.append(selected_window.start.state.detach().cpu())
                                all_actions.extend(p['actions'].detach().cpu().flatten(0,1) for p in physical)
                        except PhysicsRejected as error:
                            record.update(rejection=error.reason,details=error.details)
                        rows.append(record)
                        with (output/'probes.jsonl').open('a') as file:
                            file.write(json.dumps(record,ensure_ascii=False)+'\n')
                        print(json.dumps({k:v for k,v in record.items() if k not in ('real_contacts','coverage')},ensure_ascii=False),flush=True)
    profile_certificates = {}
    for horizon in cfg['audit']['horizons']:
        certificates[str(horizon)] = {}
        for (h,tag),values in collected.items():
            if h!=horizon:continue
            selected = [r for r in rows if r.get('horizon')==h and tag in r.get('tags',[])]
            successes = values['rows']
            sources = sorted({r['source'] for r in successes})
            anchors = {(r['source'],r['frame']) for r in successes}
            rate = len(successes)/max(1,len(selected))
            passed = (len(sources)>=cfg['audit']['minimum_class_sources'] and len(anchors)>=cfg['audit']['minimum_class_anchors'] and
                      rate>=cfg['audit']['minimum_direction_fraction'])
            actions = torch.cat(values['actions'])
            certificates[str(h)][tag] = {'passed':passed,'sources':sources,'anchors':len(anchors),'probes':len(selected),
                'passed_probes':len(successes),'fraction':rate,'state_envelope':state_envelope(torch.cat(values['states'])),
                'audited_timesteps':sorted({r['timestep'] for r in successes}),
                'action_min':actions.amin(0).tolist(),'action_max':actions.amax(0).tolist()}
        profile_certificates[str(horizon)] = {}
        for (h,profile),values in profiles.items():
            if h!=horizon:continue
            selected = [r for r in rows if r.get('horizon')==h and '|'.join(r.get('tags',[]))==profile]
            successes = values['rows']
            sources = sorted({r['source'] for r in successes})
            anchors = {(r['source'],r['frame']) for r in successes}
            rate = len(successes)/max(1,len(selected))
            passed = (len(sources)>=cfg['audit']['minimum_class_sources'] and len(anchors)>=cfg['audit']['minimum_class_anchors'] and
                      rate>=cfg['audit']['minimum_direction_fraction'])
            actions = torch.cat(values['actions'])
            profile_certificates[str(h)][profile] = {'passed':passed,'sources':sources,'anchors':len(anchors),
                'probes':len(selected),'passed_probes':len(successes),'fraction':rate,
                'state_envelope':state_envelope(torch.cat(values['states'])),
                'audited_timesteps':sorted({r['timestep'] for r in successes}),
                'action_min':actions.amin(0).tolist(),'action_max':actions.amax(0).tolist()}
        if any(c['passed'] for c in profile_certificates[str(horizon)].values()):supported.append(horizon)
    adapter.bundle.assert_unchanged(frozen)
    report = {'schema':'genmo.joint_physics.validation.v2','assets':assets,'objective':execution_contract(cfg),
        'source_ids':sorted(source_ids),'supported_horizons':supported,'certificates':certificates,
        'profile_certificates':profile_certificates,'certification_scope':'exact_motion_and_contact_profile',
        'audited_timesteps':cfg['audit']['timesteps'],'classification':CLASSIFICATION,'criteria':cfg['audit'],
        'prediction_origin':'gt_interface_not_training_success' if interface_probe else 'student_denoised_x0',
        'student_physics_updates':state['counters']['physics_updates'] if state else 0,
        'checkpoint':str(checkpoint) if checkpoint else None,'rows':rows,'frozen_unchanged':True,
        'forecast_passed':False,'evidence':'paired_newton_replayed_prefix_real_contacts'}
    if all_states:
        report.update(state_envelope=state_envelope(torch.cat(all_states)),action_min=torch.cat(all_actions).amin(0).tolist(),
                      action_max=torch.cat(all_actions).amax(0).tolist())
    # 条件前瞻验证使用原数据固定起点，单独补充，不把方向认证当作前瞻正确。
    from .evaluation import validate_forecasts
    forecasts = validate_forecasts(cfg,adapter)
    report.update(forecast_rows=forecasts,forecast_passed=bool(forecasts) and all(r['passed'] for r in forecasts))
    (output/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    return report
