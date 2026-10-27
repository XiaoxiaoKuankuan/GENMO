"""固定原请求条件与噪声的生成器漂移分解，区分参数变化和在线历史反馈。

使用原Stage1真实闭环的请求快照、预测状态、已知前缀及DDIM噪声，分别加载原Actor、
完整新Actor、仅新FutureEncoder及仅新Stage1参数；不创建优化器、不运行假物理。
与同音乐时间上实际在线新版本的参考比较，记录权重改变的直接输出差异及滚动条件
改变带来的额外差异。非线性交互单列，禁止把四种差异简单解释为可加的贡献百分比。
该模块仅是实际权重生成输出审计，Newton执行收益必须看独立完整真实评价。
"""
import gc
import json
from pathlib import Path
import torch
from .checkpoint import unpack_value
from .output_trust_region import motion_distance,activity_metrics
from .drift_diagnostics import write_json,stats


@torch.no_grad()
def run_conditionals(actor,engine,cfg,tasks,output,device):
    settings=cfg['diagnostic'];root=Path(settings['source_run'])/'evaluations/fixed_delay_replay'
    sources=settings['conditional_checkpoints']
    base=torch.load(sources['0']['checkpoint'],map_location='cpu',weights_only=False,mmap=True)['actor']
    models={label:torch.load(v['checkpoint'],map_location='cpu',weights_only=False,mmap=True)['actor'] for label,v in sources.items()}
    reports=[]
    for task in tasks:
        src=next((root/'initial_genmo_baseline').glob('shard_*/'+task['id']+'_seed42_real_controls.pt'))
        payload=torch.load(src,map_location='cpu',weights_only=False,mmap=True)
        # 教师不是GENMO输入；删除仅用于奖励的旁路元数据，避免重复构造整曲教师FK对象。
        for g in payload['generations']['data']:
            g['data']['request']['data']['snapshot']['data']['teacher_reference']=None
        records=unpack_value(payload['generations'],engine.adapter,device,{})
        records=sorted([r for r in records if r.accepted],key=lambda r:r.request.ticket.request_tick)
        indices=sorted({0,min(5,len(records)-1),min(15,len(records)-1),len(records)//4,len(records)//2,3*len(records)//4,len(records)-2})
        selected=[records[i] for i in indices if i>=0]
        actual={}
        for label,v in sources.items():
            folder=Path(v['evaluation'])
            source=next(folder.glob('shard_*/'+task['id']+'_seed42_real_controls.pt'),None)
            if source is None:continue
            data=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
            actual[label]={g['data']['request']['data']['ticket']['data']['request_tick']:g['data']['plan']['data']['world_qpos']
                           for g in data['generations']['data'] if g['data']['accepted']}
        for record in selected:
            request=record.request;prefix=request.ticket.prefix_frames
            actor.load_state_dict(base,strict=True)
            before=actor.generate(request.snapshot,request.ticket,record.forecast,request.noise).world_qpos
            free=torch.arange(60,device=device)>=prefix
            baseline_parity=float((before-record.plan.world_qpos).abs().max())
            for label,model in models.items():
                if label=='0':continue
                outputs={};parts={}
                for mode in ['full','future_only','stage1_only']:
                    state={k:(model[k] if mode=='full' or (mode=='future_only' and k.startswith('future_encoder.')) or
                        (mode=='stage1_only' and k.startswith('stage1.')) else v) for k,v in base.items()}
                    actor.load_state_dict(state,strict=True)
                    after=actor.generate(request.snapshot,request.ticket,record.forecast,request.noise).world_qpos
                    outputs[mode]=after
                    parts[mode]={'distance_from_stage1':motion_distance(after,before,free),
                        'activity_vs_stage1':activity_metrics(after,before,free),
                        'known_prefix_equal':torch.equal(after[:,:prefix],before[:,:prefix])}
                row={'task':task['id'],'checkpoint':label,'request_tick':request.ticket.request_tick,
                    'original_request_regeneration_max_abs':baseline_parity,'same_request_noise_forecast_history':True,'parts':parts,
                    'joint_nonlinear_interaction_rms_rad':float((outputs['full'][:,free,7:]-outputs['future_only'][:,free,7:]-outputs['stage1_only'][:,free,7:]+before[:,free,7:]).square().mean().sqrt())}
                online=actual.get(label,{}).get(request.ticket.request_tick)
                if online is not None:
                    online=online.to(before)
                    row['online_full_vs_original_reference']=motion_distance(online,record.plan.world_qpos,free)
                    row['online_conditions_feedback_vs_fixed_conditions']=motion_distance(online,outputs['full'],free)
                reports.append(row)
                with (output/'conditional_drift.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        del payload,records,actual;gc.collect()
        write_json(output/'summary.json',{label:{'requests':len(rows),
            'fixed_conditions':{mode:{key:stats([r['parts'][mode]['distance_from_stage1'][key] for r in rows])
                for key in ['joint_rms_rad','root_rms_m','orientation_rms_rad']} for mode in ['full','future_only','stage1_only']},
            'online_feedback_joint_rms':stats([r.get('online_conditions_feedback_vs_fixed_conditions',{}).get('joint_rms_rad') for r in rows]),
            'baseline_regeneration_max_abs':max(r['original_request_regeneration_max_abs'] for r in rows),
            'no_physics_quality_claim':True} for label in models if label!='0' and (rows:=[r for r in reports if r['checkpoint']==label])})
        print(json.dumps({'event':'fixed_condition_drift_task_complete','task':task['id'],'cases':len(reports)}),flush=True)
    actor.load_state_dict(base,strict=True)
