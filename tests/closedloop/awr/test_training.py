"""AWR优化、Critic完整拟合、跨优化步恢复与八进程SUM归约测试。

Actor使用真实小型Stage1；BC替身只用于隔离优化/恢复算术，原Stage1完整损失由
既有回归测试独立覆盖。八进程CPU/Gloo检验空rank和独立AWR/BC分母，不冒充NCCL。
"""
import copy
from pathlib import Path
import tempfile
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from gem.closedloop.awr.critic import CriticFit, predict
from gem.closedloop.awr.fvi import fit_crossfit
from gem.closedloop.awr.folds import build_folds
from gem.closedloop.awr.dataset import ReplayDataset
from gem.closedloop.awr.distributed import Distributed
from gem.closedloop.awr.trainer import ActorTrainer
from gem.closedloop.awr.checkpoint import Checkpoints


class ArithmeticBC:
    """仅检验BC全局分母和恢复，不代表真实预训练数据。"""
    def __init__(self):
        self.bc_update_steps = 0
    def state_dict(self):
        return dict(bc_update_steps=self.bc_update_steps)
    def load_state_dict(self, value):
        self.bc_update_steps = value["bc_update_steps"]
    def loss(self, actor, count):
        parameter = next(p for p in actor.parameters() if p.requires_grad)
        return (parameter.square().mean()+.01).expand(count)


def test_actor_optimizer_resume_never_samples_chain(tmp_path, episode_factory, monkeypatch):
    actor, _, make = episode_factory
    data = ReplayDataset([make(episode_id="a"), make(episode_id="b")], unit_weights=True)
    initial = copy.deepcopy(actor.state_dict())
    config = dict(epochs=2, global_batch=3, microbatch=1, bc_global_batch=3, bc_microbatch=2,
                  lr=1e-4, warmup_steps=1)
    def forbidden(*args, **kwargs):
        raise AssertionError("Actor training must not call the DDIM sampling chain")
    monkeypatch.setattr(actor, "sample", forbidden)
    trainer = ActorTrainer(actor, data, ArithmeticBC(), config, Distributed(), bc_weight=.1)
    saved = []
    def on_step(current, _):
        if current.steps == 1:
            saved.append(current.state_dict())
    report = trainer.run(on_step=on_step)
    assert report["steps"] == 4 and report["complete"]
    final = copy.deepcopy(actor.state_dict())
    assert any(not torch.equal(initial[k], v) for k, v in final.items())
    checkpoint = Checkpoints(tmp_path/"checkpoints", {"run": "synthetic"})
    checkpoint.save("one", saved[0], progress={"phase": "actor", "step": 1})
    trainer2 = ActorTrainer(actor, data, ArithmeticBC(), config, Distributed(), bc_weight=.1)
    trainer2.load_state_dict(checkpoint.load()["payload"])
    assert trainer2.run()["steps"] == 4
    for key, value in final.items():
        torch.testing.assert_close(actor.state_dict()[key], value, atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["mc_only", "mc_fvi"])
def test_crossfit_full_fit_mc_fvi_is_finite(episode_factory, mode):
    actor, _, make = episode_factory
    samples = [dict(split="train", group_id=f"g{i}", audio_sha256=f"a{i}", source="s") for i in range(8)]
    folds = build_folds(samples)
    episodes = [make(episode_id=str(i), group=f"g{i}", rewards=((float(i)-3., float(i)), (float(i)/2.,)))
                for i in range(8)]
    config = dict(mode=mode, history_lr=1e-5, lr=1e-4, global_batch=4, microbatch=2,
                  mc_epochs=1, fvi_epochs=1, fvi_iterations=1, nstep=1, patience=1)
    labels, report, states = fit_crossfit(actor, episodes, folds, config, Distributed())
    assert torch.isfinite(labels.values).all() and len(labels.sample_ids) == 16
    assert torch.equal(labels.advantages, labels.returns-labels.values)
    for evidence in report["provenance"]:
        assert not set(evidence["trained_groups"]) & set(evidence["labeled_groups"])
    assert len(states["selected"]) == 2
    if mode == "mc_only":
        assert report["selected"] == "mc_only"
    else:
        assert all(r["fvi"][0]["frozen_target_identity"] for r in report["folds"])


def _distributed_gradient_worker(rank, world, path):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://"+path+"/rendezvous", rank=rank, world_size=world)
    try:
        d = Distributed()
        torch.manual_seed(22)
        model = torch.nn.Linear(3, 2).double()
        x = torch.arange(15, dtype=torch.float64).reshape(5, 3)/10
        y = torch.arange(10, dtype=torch.float64).reshape(5, 2)/7
        w = torch.tensor([.5, 3., .1, 1., 2.], dtype=torch.float64)
        ids = list(range(rank, 5, world))
        for index in ids:  # 非整除尾批、三个空rank、不同micro切分。
            ((model(x[index:index+1])-y[index:index+1]).square().mean()*w[index]/5).backward()
        # BC只有3条，使用自己的全局分母，与AWR不同。
        for index in range(rank, 3, world):
            (.1*model(x[index:index+1]).square().mean()/3).backward()
        d.gradients(model)
        torch.save([p.grad for p in model.parameters()], Path(path)/f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_eight_rank_cpu_global_denominators_and_empty_rank(tmp_path):
    mp.start_processes(_distributed_gradient_worker, args=(8, str(tmp_path)), nprocs=8,
                       join=True, start_method="fork")
    torch.manual_seed(22)
    model = torch.nn.Linear(3, 2).double()
    x = torch.arange(15, dtype=torch.float64).reshape(5, 3)/10
    y = torch.arange(10, dtype=torch.float64).reshape(5, 2)/7
    w = torch.tensor([.5, 3., .1, 1., 2.], dtype=torch.float64)
    (((model(x)-y).square().mean(-1)*w).mean()+.1*model(x[:3]).square().mean()).backward()
    for rank in range(8):
        values = torch.load(tmp_path/f"rank{rank}.pt", weights_only=True)
        for actual, expected in zip(values, model.parameters()):
            torch.testing.assert_close(actual, expected.grad, atol=1e-12, rtol=1e-12)
