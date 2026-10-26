"""合并独立完整真实MC分片，从随机初始化重新拟合音乐教师奖励Critic。

严格核对实际Actor参数、冻结资产、教师、奖励、采样和物理合同；源码版本只记录。
训练/验证音乐必须不相交，不能把旧奖励Critic或模型回报混入监督。该进程不启动
物理求解器，不优化Actor，输出独立真实回报误差和新schema的critic_ready.pt。
"""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--shards",nargs="+",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from omegaconf import OmegaConf
    cfg=OmegaConf.to_container(OmegaConf.load(args.config),resolve=True)
    sys.path.insert(0,cfg["supermimic"]["repo"])
    import torch
    from gem.closedloop.mb_control.application import build_components,build_stage1_losses,seed_all
    from gem.closedloop.mb_control.trainer import Stage2Trainer
    from gem.closedloop.mb_control.critic_warmup import fit_mc_dataset,save_warmup
    from gem.closedloop.mb_control.checkpoint import unpack_state
    seed_all(cfg["training"]["seed"])
    actor,adapter,_,_,critic,engine,original=build_components(cfg,args.device)
    trainer=Stage2Trainer(actor,critic,engine,build_stage1_losses(actor.stage1,original).to(args.device),cfg["training"])
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    pairs={"train":[],"validation":[]};episodes={"train":[],"validation":[]};steps=0
    actor_before={n:p.detach().clone() for n,p in actor.named_parameters()}
    episode_keys=set()
    for directory in map(Path,args.shards):
        recorded=json.loads((directory/"manifest.json").read_text())["config"]
        if recorded.get("teacher",{}).get("seed")!=cfg["teacher"]["seed"]:
            raise ValueError("真实MC固定音乐时间线的原始运行种子不同")
        complete=torch.load(directory/"mc_complete.pt",map_location="cpu",weights_only=False,mmap=True)
        if complete["contract"] != {k:cfg[k] for k in ("model","clock","rewards","critic")}:
            raise ValueError("真实MC奖励或执行合同不一致")
        if complete["teacher_identity"]!=engine.teacher.identity or complete["frozen_sha256"]!=adapter.bundle.metadata["sha256"]:
            raise ValueError("真实MC教师或冻结资产不一致")
        if complete["physics_determinism"]!=cfg["execution"].get("physics_determinism"):
            raise ValueError("真实MC物理设置不一致")
        if cfg.get("teacher",{}).get("target_mode")=="online_consumed_reference_v1":
            from gem.closedloop.mb_control.music_mc import critic_reward_contract
            if complete.get("critic_reward_contract")!=critic_reward_contract(cfg):
                raise ValueError("完整MC不是同一anchor对齐的新奖励合同")
        if any(not torch.equal(v.detach().cpu(),complete["actor"][n]) for n,v in actor.state_dict().items()):
            raise ValueError("真实MC不是当前原Stage1初始Actor产生")
        split=complete["split"];steps+=complete["physical_control_steps"]
        episodes[split].extend(complete["episodes"])
        for path in sorted((directory/"mc_episodes").glob(f"{split}_*.pt")):
            payload=torch.load(path,map_location="cpu",weights_only=False)
            key=(split,payload["episode"]["task"],payload["episode"]["seed"])
            if key in episode_keys:raise ValueError("重复MC episode，不能重复计数")
            episode_keys.add(key);memo={}
            if payload["evidence"]!="real_complete_episode_mc":raise ValueError("必须是真实完整MC")
            pairs[split].extend((unpack_state(z,adapter,args.device,memo),y.to(args.device)) for z,y in payload["pairs"])
    if {x["task"] for x in episodes["train"]}&{x["task"] for x in episodes["validation"]}:
        raise ValueError("真实MC训练/验证音乐重叠")
    (output/"collection_manifest.json").write_text(json.dumps({"shards":args.shards,"episodes":episodes,
        "physical_control_steps":steps,"reward":cfg["rewards"],"teacher":engine.teacher.identity,
        "teacher_seed":cfg["teacher"]["seed"]},ensure_ascii=False,indent=2))
    try:
        fit_mc_dataset(trainer,pairs["train"],pairs["validation"],cfg["formal"]["critic_warmup"],output,steps,actor_before)
    finally:
        from gem.closedloop.mb_control.trainer import frozen_parameters
        with frozen_parameters(critic):
            audits=[critic.audit_mutable_reference_gradient(z) for z,_ in pairs["validation"][::max(1,len(pairs["validation"])//12)]]
        (output/"critic_mutable_reference_gradient.json").write_text(json.dumps({"samples":audits,
            "terminal_value_weight":0.,"gradient_is_not_execution_improvement":True},indent=2,allow_nan=False))
        torch.save({"schema":"critic_fit_diagnostic_not_actor_authorization.v1","critic":critic.state_dict(),
            "quality":trainer.critic_quality,"qualified":trainer.critic_ready},output/"critic_diagnostic.pt")
    save_warmup(output/"critic_ready.pt",trainer,cfg)


if __name__=="__main__":main()
