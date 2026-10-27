"""用真实Newton补充冻结WM实际动作输入的数值支持证据。

原方向认证比较真实反馈SM闭环与WM反馈SM闭环；两条轨迹的动作通常不完全相同。
只用前者动作极值建门禁，可能把已经验证过的后者本身拒绝。本模块不增加经验容差，
而在原认证的固定anchor上，将WM闭环产生的逐步动作序列实际交给Newton，检查同动作
模型误差、硬关节限位、原SM终止和同起点重放。只有真正执行且通过的输入才进入新证书。
这是GT参考的动力学接口诊断，没有GENMO/AdamW更新，也不作为学生训练或实时闭环证据。
原报告及失败结果不改写；新增报告绑定原方向证据、资产、H、目标和真实动作轨迹。
"""

from collections import defaultdict
import copy
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from gem.closedloop.stage1_dataset import collate_stage1_training_samples
from gem.closedloop.training import batch_to_device
from gem.closedloop.mb_control.validation import state_envelope, combine_envelopes
from gem.closedloop.mb_control.model_audit import state_errors
from .checkpoint import atomic_save
from .config import ROOT
from .contracts import PhysicsRejected, digest, execution_contract
from .data import SourceCatalog, SidecarStore
from .evaluation import replay_to_start
from .execution import execute_window
from .prepare_rollouts import asset_contract
from .reference import attach_sidecars
from .requests import motion_tags


@torch.no_grad()
def validate(cfg, report_path, output, adapter, env):
    if env is None:
        raise ValueError('动作输入支持必须实际执行Newton')
    report = json.loads(Path(report_path).read_text())
    assets = asset_contract(cfg,adapter)
    if report.get('assets')!=assets or report.get('objective')!=execution_contract(cfg) or not report.get('forecast_passed'):
        raise ValueError('基础方向证据的资产/目标/前瞻合同不一致')
    if report.get('certification_scope')!='exact_motion_and_contact_profile':
        raise ValueError('需要完整运动接触组合证书')
    output = Path(output);output.mkdir(parents=True,exist_ok=False)
    (output/'input_manifest.json').write_text(json.dumps({'config':cfg,'base_report':str(report_path),
        'base_sha256':digest(report_path),'selection':'all_anchors_in_previously_passed_profiles'},ensure_ascii=False,indent=2))
    from gem.robots.bumi.endecoder import BumiEndecoder
    options = dict(cfg['stage1']['endecoder'])
    options['kinematics_path'] = str(ROOT/options['kinematics_path'])
    codec = BumiEndecoder(**options).to(env.device)
    actor = SimpleNamespace(stage1=SimpleNamespace(endecoder=codec))
    catalog = SourceCatalog(cfg,'train')
    sources = {json.dumps([ds.dataset_name,ds.split,ds.rows[i]['sample_id']],ensure_ascii=False,separators=(',',':')):(ds,i)
               for ds,i in catalog.sources()}
    store = SidecarStore(cfg['data']['sidecar'],'train',assets,sample_contract=cfg['data']['sample_contract'])
    frozen = adapter.bundle.frozen_snapshot()
    anchors = {}
    for row in report['rows']:
        h = row.get('horizon')
        profile = '|'.join(row.get('tags',[]))
        if report['profile_certificates'].get(str(h),{}).get(profile,{}).get('passed'):
            anchors[(row['source'],row['frame'],h)] = profile
    rows,traces = [],[]
    groups = defaultdict(list)
    began = time.perf_counter()
    for (source,frame,horizon),profile in sorted(anchors.items()):
        row = {'source':source,'frame':frame,'horizon':horizon,'profile':profile,
               'reference_origin':'gt_interface_only','action_origin':'frozen_sm_on_wm_predicted_history'}
        try:
            ds,index = sources[source]
            batch = batch_to_device(collate_stage1_training_samples([ds.get_window(index,frame)]),env.device)
            batch,_,windows,reasons = attach_sidecars(batch,store,adapter,codec.codec,horizon)
            window = windows[0]
            if window is None:raise PhysicsRejected(reasons[0])
            prediction = codec.normalize(batch['target_qpos30'])
            model = execute_window(window,prediction,prediction.new_zeros(1,60,2),actor,adapter,cfg['execution_loss'])
            episode = store.episode(batch['meta'][0])
            same_start = replay_to_start(env,episode,window)
            states,contacts = [],[]
            for step,action in enumerate(model['actions']):
                measured = env.step_physics(action)
                target,_,_ = window.target.read_current_future(window.start.tick+(step+1)*12,0)
                if not torch.isfinite(measured).all():raise PhysicsRejected('nonfinite_newton_state')
                q = measured[:,13:34]
                if bool(((q<adapter.runtime.kin.q_lo-1e-4)|(q>adapter.runtime.kin.q_hi+1e-4)).any()):
                    raise PhysicsRejected('measured_hard_joint_limit')
                if bool(adapter.runtime.termination(measured,target).any()):
                    raise PhysicsRejected('sm_physical_termination')
                states.append(measured);contacts.append(env.contact_events())
            physical = torch.cat(states)
            errors = state_errors(model['states'].flatten(0,1),physical)
            observed_profile = '|'.join(motion_tags(torch.cat((window.start.state,physical)),
                adapter.bundle.spec.joint_names,contacts))
            passed = same_start<=report['criteria']['same_start_atol'] and all(
                float(errors[name])<=bound for name,bound in report['criteria']['state_rmse_limits'].items()) and observed_profile==profile
            row.update(passed=passed,same_start_max_abs=same_start,model_errors={k:float(v) for k,v in errors.items()},
                       control_steps=horizon,real_contacts=contacts,observed_profile=observed_profile)
            trace = {'source':source,'frame':frame,'horizon':horizon,'profile':profile,
                     'model_states':model['states'].cpu(),'executed_actions':model['actions'].cpu(),
                     'newton_states':physical.cpu(),'real_start':window.start.state.cpu(),'passed':passed}
            traces.append(trace)
            if passed:groups[(horizon,profile)].append(trace)
        except PhysicsRejected as error:
            row.update(passed=False,rejection=error.reason,details=error.details)
        rows.append(row)
        with (output/'action_inputs.jsonl').open('a') as file:file.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='real_contacts'},ensure_ascii=False),flush=True)
    result = copy.deepcopy(report)
    for h,certificates in result['profile_certificates'].items():
        for profile,certificate in certificates.items():
            if not certificate['passed']:continue
            accepted = groups[(int(h),profile)]
            attempted = [r for r in rows if r['horizon']==int(h) and r['profile']==profile]
            passed = (len(accepted)>=report['criteria']['minimum_class_anchors'] and
                      len({r['source'] for r in accepted})>=report['criteria']['minimum_class_sources'] and
                      len(accepted)/max(1,len(attempted))>=report['criteria']['minimum_direction_fraction'])
            certificate['action_input_validation'] = {'passed':passed,'anchors':len(accepted),'attempted':len(attempted),
                'source_count':len({r['source'] for r in accepted}),'evidence':'model_actions_actually_executed_in_newton'}
            certificate['passed'] = passed
            if passed:
                real = torch.cat([torch.cat((r['real_start'],r['newton_states'])) for r in accepted])
                actions = torch.cat([r['executed_actions'].flatten(0,1) for r in accepted])
                certificate['state_envelope'] = combine_envelopes([certificate['state_envelope'],state_envelope(real)])
                certificate['action_min'] = torch.minimum(torch.tensor(certificate['action_min']),actions.amin(0)).tolist()
                certificate['action_max'] = torch.maximum(torch.tensor(certificate['action_max']),actions.amax(0)).tolist()
    adapter.bundle.assert_unchanged(frozen)
    result['supported_horizons'] = [int(h) for h,c in result['profile_certificates'].items() if any(v['passed'] for v in c.values())]
    result.update(action_input_rows=rows,action_input_wall_seconds=time.perf_counter()-began,
        base_direction_report=str(report_path),base_direction_report_sha256=digest(report_path),
        action_input_evidence='real_newton_same_actions_not_student_training',frozen_unchanged=True)
    atomic_save(traces,output/'executed_action_traces.pt')
    (output/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    return result
