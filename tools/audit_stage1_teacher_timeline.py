"""审计旧独立教师A、原Stage1真实消费参考B和真实机器人C的音乐时间线。

只读保存的完整Newton在线轨迹，按同一控制时间戳导出根部、yaw、21关节、速度、
幅度、相位及参考覆盖边界。A与B若噪声身份不同，明确报告不可将不同舞蹈逐帧差异
归咎于延迟或相位错误。B/C来自同一次真实闭环，失败曲目保留实际时长及原因。
输出均为独立诊断产物，不改写被冻结的教师缓存，不训练Stage1或SuperMimic。
"""
import argparse,json,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True);p.add_argument("--output",required=True)
    p.add_argument("--device",default="cpu")
    a=p.parse_args();sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from omegaconf import OmegaConf
    cfg=OmegaConf.to_container(OmegaConf.load(a.config),resolve=True);sys.path.insert(0,cfg["supermimic"]["repo"])
    import numpy as np
    import torch
    from gem.closedloop.mb_control.application import build_components
    from gem.closedloop.mb_control.checkpoint import unpack_value
    from gem.closedloop.mb_control.dance_evaluation import trajectory_content
    from gem.closedloop.mb_control.evaluation import motion_metrics
    from supermimic.rotations import quat_to_mat,yaw_mat,so3_log
    _,adapter,_,_,_,_,_=build_components(cfg,a.device)
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(Path(cfg["teacher"]["online_manifest"]).read_text());rows=[]
    for item in manifest["timelines"].values():
        payload=torch.load(item["path"],map_location="cpu",weights_only=False)
        b=payload["reference_state"][0];c=payload["actual_state"][0];research=payload["research_independent_world_qpos"][0]
        inverse=torch.argsort(torch.tensor(adapter.to_sm))
        online=torch.cat((b[:,:7],b[:,13:34][:,inverse]),-1)
        memo={};controls=unpack_value(payload["controls"],adapter,"cpu",memo)
        consumed=[]
        for step in controls:
            cache=step.start.reference;support=cache.dependency_support(step.end.tick,future_steps=0)
            consumed.append({"tick":step.end.tick,"cache_version":cache.version,
                "source_support_ticks":support.tolist(),"source_versions":cache.source_versions[(support-cache.lower)//20].tolist()})
        ar=quat_to_mat(research[:,3:7]);br=quat_to_mat(b[:,3:7])
        aq=research[:,7:][:,adapter.to_sm]
        av=torch.cat((torch.zeros(1,3),torch.diff(research[:,:3],dim=0)*50))
        aw=torch.cat((torch.zeros(1,3),so3_log(ar[1:]@ar[:-1].transpose(-1,-2))*50))
        aqd=torch.cat((torch.zeros(1,21),torch.diff(aq,dim=0)*50))
        astate=torch.cat((research[:,:7],av,aw,aq,aqd),-1)
        safe=str(payload["task"]["id"]).replace("/","_")
        np.savez_compressed(output/(safe+"_abc.npz"),ticks=payload["ticks"].numpy(),
            independent_A_qpos28=research.numpy(),consumed_B_state55=b.numpy(),physical_C_state55=c.numpy(),
            A_minus_B_joint_rad=(research[:,7:][:,adapter.to_sm]-b[:,13:34]).numpy(),
            A_minus_B_yaw_rad=so3_log(yaw_mat(ar)@yaw_mat(br).transpose(-1,-2))[:,2].numpy())
        row={"task":payload["task"]["id"],"complete":payload["complete"],"reason":payload["reason"],
            "actual_seconds":int(payload["ticks"][-1])/600,"A_noise_seed":payload["research_seed"],
            "B_noise_seed":payload["online_noise_seed"],"same_dance_identity":payload["research_noise_identity_matches_online"],
            "A_B_framewise_differences_are_not_a_phase_failure_test":not payload["research_noise_identity_matches_online"],
            "B_against_A":trajectory_content(b,b[:,13:34],research,adapter),
            "C_against_B":trajectory_content(c,b[:,13:34],online,adapter),
            "A_beats":motion_metrics([(int(t),s[None]) for t,s in zip(payload["ticks"],astate)],controls[0].start.music),
            "A_velocity_scope":"finite_difference_of_saved_50hz_independent_reference",
            "B_beats":motion_metrics([(int(t),s[None]) for t,s in zip(payload["ticks"],b)],controls[0].start.music),
            "C_beats":motion_metrics([(int(t),s[None]) for t,s in zip(payload["ticks"],c)],controls[0].start.music),
            "request_events":payload["events"],"consumed_support":consumed,
            "original_stage1_parity_max_abs":payload["original_stage1_parity_max_abs"]}
        q=c[:,13:34];kin=adapter.runtime.kin;mdp=adapter.runtime.mdp
        row["real_joint_limit_frequency"]={name:{
            "soft":float(((q[:,i]<mdp.soft_lo[i])|(q[:,i]>mdp.soft_hi[i])).float().mean()),
            "hard":float(((q[:,i]<kin.q_lo[i])|(q[:,i]>kin.q_hi[i])).float().mean()),
            "near_hard_within_005rad":float((torch.minimum(q[:,i]-kin.q_lo[i],kin.q_hi[i]-q[:,i])<.05).float().mean())}
            for i,name in enumerate(adapter.bundle.spec.joint_names)}
        rows.append(row)
        print(json.dumps({k:row[k] for k in ("task","complete","reason","actual_seconds","same_dance_identity")},ensure_ascii=False),flush=True)
    (output/"timeline_abc.json").write_text(json.dumps({"episodes":rows,"target_for_training":"B_original_online_consumed_reference"},ensure_ascii=False,indent=2,allow_nan=False))


if __name__=="__main__":main()
