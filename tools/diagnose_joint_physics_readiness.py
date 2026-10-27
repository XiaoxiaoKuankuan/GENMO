"""区分学生去噪不足与真实起点/WM支持不足的有界诊断。

按独立真实请求游标取样，分别执行当前学生的x0和明确标记的GT接口对照；GT结果只用于
定位失败原因，绝不回填学生预测、不参与AdamW、不计物理有效更新。保存逐坐标首个越界、
噪声等级、未知坐标重建误差和同一窗口的两条拒绝路径，避免把所有拒绝归咎于学生。
该命令需要真实分层认证和v2缓存，全部结果写入新目录。
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--samples',type=int,default=32)
    parser.add_argument('--set',action='append',default=[])
    args=parser.parse_args()
    import torch
    from gem.closedloop.joint_physics.config import read_config,build_adapter
    cfg=read_config(args.config,args.set)
    torch.set_num_threads(cfg['runtime']['cpu_threads'])
    adapter=build_adapter(cfg,cfg['runtime']['device'])
    from gem.closedloop.joint_physics.evaluation import load_actor
    from gem.closedloop.joint_physics.data import SourceCatalog,SidecarStore
    from gem.closedloop.joint_physics.prepare_rollouts import asset_contract
    from gem.closedloop.joint_physics.requests import PhysicsRequestStream
    from gem.closedloop.joint_physics.execution import ValidatedSupport,execute_window
    from gem.closedloop.joint_physics.contracts import PhysicsRejected,execution_contract
    from gem.closedloop.joint_physics.reference import attach_sidecars
    from gem.closedloop.joint_physics.curriculum import physics_noise_levels
    actor,_=load_actor(cfg,args.checkpoint,cfg['runtime']['device'])
    assets=asset_contract(cfg,adapter)
    support=ValidatedSupport(cfg['physics']['validation_report'],assets,cfg['physics']['horizon'],cfg['runtime']['device'],execution_contract(cfg))
    store=SidecarStore(cfg['data']['sidecar'],'train',assets,sample_contract=cfg['data']['sample_contract'])
    stream=PhysicsRequestStream(SourceCatalog(cfg,'train'),store,adapter.bundle.spec.joint_names,cfg['physics']['horizon'],12345,certification=support)
    levels=physics_noise_levels(actor.stage1.train_diffusion,cfg['physics'],support.report)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    rows=[]
    generator=torch.Generator().manual_seed(12345)
    frozen=adapter.bundle.frozen_snapshot()
    for _ in range(args.samples if stream.keys else 0):
        batch,metadata=stream.next(1,cfg['runtime']['device'])
        batch,contexts,windows,missing=attach_sidecars(batch,store,adapter,actor.stage1.endecoder.codec,cfg['physics']['horizon'])
        k=int(levels[int(torch.randint(len(levels),(1,),generator=generator))])
        noise=torch.randn(1,60,30,generator=generator).to(cfg['runtime']['device'])
        with torch.no_grad():
            out=actor.denoise_supervised(batch,contexts,torch.tensor([k],device=noise.device),noise,torch.zeros(1,dtype=torch.bool,device=noise.device))
            mask=out['unknown_mask']
            record=dict(metadata[0],timestep=k,unknown_reconstruction_rmse=float((out['pred_x_start']-out['target_x_start'])[mask].square().mean().sqrt()))
            for name,prediction in (('student',out['pred_x_start']),('gt_interface_only',out['target_x_start'])):
                try:
                    if windows[0] is None:raise PhysicsRejected(missing[0])
                    value=execute_window(windows[0],prediction,out['static_conf_logits'],actor,adapter,cfg['execution_loss'],support=support.for_tags(metadata[0]['tags']))
                    record[name]={'valid':True,'loss':float(value['loss'])}
                except PhysicsRejected as error:
                    record[name]={'valid':False,'reason':error.reason,'details':error.details}
        record['cause_hint']=('student_deviation' if not record['student']['valid'] and record['gt_interface_only']['valid'] else
                              'support_or_world_model_or_reference_not_student_only' if not record['gt_interface_only']['valid'] else 'both_valid')
        rows.append(record)
    adapter.bundle.assert_unchanged(frozen)
    (output/'diagnosis.json').write_text(json.dumps({'checkpoint':args.checkpoint,'rows':rows,'inventory':stream.inventory,
        'optimizer_updates':0,'gt_is_not_training_success':True,'frozen_unchanged':True},ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
