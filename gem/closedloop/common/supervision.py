"""Stage1 Actor装配与真实监督保留支路。

从原始Stage1 checkpoint装配相同网络和统计量，只恢复权重，不继承PPO优化器。
BC读取原始配对数据及真实Contact标签，复用完整Stage1损失和独立可恢复随机状态；
不接受AWR生成Contact作为真值，不依赖PPO概率、GAE或训练记录。
"""
from __future__ import annotations
import math
import random
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from gem.closedloop.checkpoint import load_stage1_checkpoint
from gem.closedloop.training import build_stage1_actor, build_stage1_losses, batch_to_device
from .performance import measure

def load_actor(config):
    paths = config['paths']
    payload = torch.load(paths['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    train = OmegaConf.create(payload['config'])
    train.endecoder.stats_path = paths['stats']
    train.endecoder.kinematics_path = paths['kinematics']
    train.endecoder.allow_placeholder_stats = False
    data = OmegaConf.create(dict(qpos30_stats=dict(path=paths['stats']),
        dataset_defaults=dict(kinematics_path=paths['kinematics']),sample_contract=dict(history_steps=50),datasets={}))
    actor = build_stage1_actor(train,data)
    from gem.closedloop.dppo.position_repair import canonical_position_tables, audit_position_tables
    canonical = canonical_position_tables(actor)
    report = load_stage1_checkpoint(actor,payload)
    report['fixed_position_tables'] = audit_position_tables(actor, canonical=canonical)
    if report['fixed_position_tables']['modified_before_repair']:
        raise ValueError('Source fixed position tables differ from the architecture; explicit weights-only repair is required')
    for key, value in actor.state_dict().items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f'Nonfinite checkpoint parameter: {key}')
    report.update(parameter_count=sum(p.numel() for p in actor.parameters()),
                  source_global_step=payload.get('global_step'),optimizer_restored=False,
                  global_step_restored=False)
    # 保留构建时的逐参数冻结规则，尤其不能解冻固定正余弦位置表。
    report.update(trainable_parameter_names=[name for name, value in actor.named_parameters() if value.requires_grad],
                  frozen_parameter_names=[name for name, value in actor.named_parameters() if not value.requires_grad])
    return actor.float().eval().to(config['runtime']['genmo_device']), train, report


def trainable_actor_parameters(actor):
    """仅将架构明确允许训练的参数交给新优化器；不迁移旧位置表 Adam 状态。"""
    parameters = [parameter for parameter in actor.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError('Actor has no explicitly trainable parameters')
    return parameters


class SupervisedAnchor:
    """原Stage1 Dataset/loss监督支路；独立随机状态和bc_update_steps可完整恢复。"""
    def __init__(self, config, actor, train_config):
        from gem.closedloop.stage1_dataset import BumiClosedLoopStage1Dataset
        data=OmegaConf.load(Path(config['paths']['genmo_repo'])/'configs/closedloop/stage1_dataset_server1_fourset_90505_v1.yaml')
        data.server1_example.root=config['stage9']['bc_data_root']
        data.qpos30_stats.path=config['paths']['stats']
        data.dataset_defaults.kinematics_path=config['paths']['kinematics']
        frames=config.get('stage10',{}).get('generation_horizon',{}).get('frames',120)
        prefix=config['stage9'].get('bc_prefix',dict(min_frames=6,max_frames=18,zero_probability=.15))
        if (type(prefix['min_frames']) is not int or type(prefix['max_frames']) is not int
                or not 1 <= prefix['min_frames'] <= prefix['max_frames'] < frames
                or not 0 <= prefix['zero_probability'] < 1):
            raise ValueError('BC prefix range must preserve at least one free future frame')
        data.sample_contract.update(prefix_min_frames=prefix['min_frames'],prefix_max_frames=prefix['max_frames'],
                                    prefix_zero_probability=prefix['zero_probability'])
        self.prefix_contract=dict(prefix)
        self.datasets=[]
        for entry in data.datasets.train.values():
            opts=OmegaConf.to_container(entry,resolve=True)
            opts.pop('_target_',None)
            opts['kinematics_path']=config['paths']['kinematics']
            opts['motion_frames']=frames
            self.datasets.append(BumiClosedLoopStage1Dataset(**opts))
        self.losses=build_stage1_losses(actor,train_config).to(next(actor.parameters()).device)
        self.generator=torch.Generator().manual_seed(int(config['stage9']['seed'])+1001)
        self.bc_update_steps=0
        self.batch_size=int(config['stage9'].get('bc_batch',2))
        self.batch_forward = config.get('stage10', {}).get('performance', {}).get('bc_batch_forward', False)
        if self.batch_size < 1:
            raise ValueError('BC batch must be positive')
        self.torch_rng=torch.Generator().manual_seed(int(config['stage9']['seed'])+1002).get_state()
        self.numpy_rng=np.random.RandomState(int(config['stage9']['seed'])+1003).get_state()
        self.python_rng=random.Random(int(config['stage9']['seed'])+1004).getstate()
        device=next(actor.parameters()).device
        self.cuda_rng_device = device if device.type == 'cuda' else None
        self.cuda_rng=(torch.Generator(device=device).manual_seed(int(config['stage9']['seed'])+1005).get_state()
                       if device.type=='cuda' else None)

    @contextmanager
    def _rng_scope(self, actor):
        """监督采样、数据窗口与dropout拥有独立且可恢复的随机状态。"""
        device=next(actor.parameters()).device
        cuda_devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
        old_numpy,old_python=np.random.get_state(),random.getstate()
        with torch.random.fork_rng(devices=cuda_devices):
            torch.set_rng_state(self.torch_rng)
            np.random.set_state(self.numpy_rng);random.setstate(self.python_rng)
            if cuda_devices:
                torch.cuda.set_rng_state(self.cuda_rng,cuda_devices[0])
            try:
                yield
            finally:
                self.torch_rng=torch.get_rng_state()
                self.numpy_rng=np.random.get_state();self.python_rng=random.getstate()
                if cuda_devices:
                    self.cuda_rng=torch.cuda.get_rng_state(cuda_devices[0])
                np.random.set_state(old_numpy);random.setstate(old_python)


    def backward(self, actor, weight):
        """两条监督样本一次前向，损失仍逐样本归约后等权平均；使用独立新RNG执行身份。"""
        from gem.closedloop.stage1_dataset import collate_stage1_training_samples
        device = next(actor.parameters()).device
        actor.train()
        try:
            with self._rng_scope(actor), torch.autocast(device_type=device.type, enabled=False):
                with measure('bc.data_prepare'):
                    samples = []
                    for _ in range(self.batch_size):
                        source = int(torch.multinomial(torch.tensor([.2,.35,.25,.2]), 1, generator=self.generator))
                        dataset = self.datasets[source]
                        index = int(torch.randint(len(dataset), (1,), generator=self.generator))
                        samples.append(dataset[index])
                    cpu_batch = collate_stage1_training_samples(samples)
                with measure('bc.transfer', gpu=True):
                    batch = batch_to_device(cpu_batch, device)
                with measure('bc.forward', gpu=True):
                    prediction = actor.training_forward(batch)
                    values, warmup = [], []
                    for i in range(self.batch_size):
                        single = {key: value[i:i+1] if isinstance(value, (torch.Tensor, list, tuple)) else value
                                  for key, value in batch.items()}
                        if 'B' in single:
                            single['B'] = 1
                        loss, details = self.losses(single, prediction['pred_x_start'][i:i+1],
                            prediction['static_conf_logits'][i:i+1], global_step=self.bc_update_steps)
                        values.append(loss)
                        warmup.append({key: float(value.detach()) for key, value in details.items() if key.endswith('warmup_factor')})
                    mean_loss = torch.stack(values).mean()
                    if not torch.isfinite(mean_loss):
                        raise FloatingPointError('Nonfinite batched Stage1 supervised retention loss')
                squares = []
                diagnostics=getattr(self,'micro_gradient_diagnostics',True)
                hooks = ([p.register_hook(lambda gradient: squares.append(gradient.detach().double().square().sum()))
                         for p in actor.parameters() if p.requires_grad] if diagnostics else [])
                try:
                    with measure('bc.backward', gpu=True):
                        (weight*mean_loss).backward()
                finally:
                    for hook in hooks:
                        hook.remove()
                norm = (float(torch.stack(squares).sum().sqrt()) if squares else 0.) if diagnostics else None
                self.bc_update_steps += 1
        finally:
            actor.eval()
        return dict(loss=float(mean_loss.detach()), weight=weight, bc_update_steps=self.bc_update_steps,
            samples=[[sample['meta']] for sample in samples], batch_size=self.batch_size,
            prefix_contract=self.prefix_contract, warmup_step=self.bc_update_steps-1,
            warmup_origin='independent_bc_updates_from_zero', warmup_factors=warmup,
            weighted_gradient_norm_per_microbatch=[norm], execution_contract='batched_forward_samplewise_loss_rng.v1')

    def state_dict(self):
        result = dict(generator=self.generator.get_state(),bc_update_steps=self.bc_update_steps,
                    torch_rng=self.torch_rng.clone(),numpy_rng=self.numpy_rng,python_rng=self.python_rng,
                    cuda_rng=self.cuda_rng.clone() if self.cuda_rng is not None else None,batch_size=self.batch_size)
        if getattr(self, 'batch_forward', False):
            result['execution_contract'] = 'batched_forward_samplewise_loss_rng.v1'
        return result

    def validate_state_dict(self, state):
        """用独立 RNG 对象验证断点，不拷贝 Dataset/Actor，也不触碰当前或全局随机状态。"""
        expected = 'batched_forward_samplewise_loss_rng.v1' if getattr(self, 'batch_forward', False) else None
        if state.get('execution_contract') != expected:
            raise ValueError('BC execution/RNG contract differs from checkpoint')
        if (type(state['batch_size']) is not int or state['batch_size'] != self.batch_size
                or type(state['bc_update_steps']) is not int or state['bc_update_steps'] < 0):
            raise ValueError('BC state batch or update count differs')
        torch.Generator(device='cpu').set_state(state['generator'])
        torch.Generator(device='cpu').set_state(state['torch_rng'])
        np.random.RandomState().set_state(state['numpy_rng'])
        random.Random().setstate(state['python_rng'])
        if (state['cuda_rng'] is None) != (self.cuda_rng is None):
            raise ValueError('BC CUDA RNG topology differs')
        if state['cuda_rng'] is not None:
            torch.Generator(device=self.cuda_rng_device).set_state(state['cuda_rng'])
        return True

    def load_state_dict(self,state):
        self.validate_state_dict(state)
        self.generator.set_state(state['generator']);self.bc_update_steps=int(state['bc_update_steps'])
        self.torch_rng=state['torch_rng'].clone();self.numpy_rng=state['numpy_rng'];self.python_rng=state['python_rng']
        self.cuda_rng=state['cuda_rng'].clone() if state['cuda_rng'] is not None else None


class Stage1Supervision(SupervisedAnchor):
    """独立算法可调用的原Stage1逐样本Loss；明确全局分母和成熟预训练步数。

    Dataset、Contact真值、FK及Loss均沿用Stage1。仅把旧backward拆成可显式指定
    样本数的前向；调用者负责梯度累积及归约。本类的随机流与生成监督互不污染。
    验证实例读取val清单，绝不向Replay或Critic提供这些样本。
    """
    def __init__(self, config, actor, train_config, *, source_step=595000, split="train"):
        super().__init__(config, actor, train_config)
        if split not in ("train", "val"):
            raise ValueError("Stage1 supervised split must be explicit")
        self.source_step, self.split = int(source_step), split
        if split == "val":
            from gem.closedloop.stage1_dataset import BumiClosedLoopStage1Dataset
            data = OmegaConf.load(Path(config["paths"]["genmo_repo"]) /
                                  "configs/closedloop/stage1_dataset_server1_fourset_90505_v1.yaml")
            data.server1_example.root = config["stage9"]["bc_data_root"]
            data.qpos30_stats.path = config["paths"]["stats"]
            data.dataset_defaults.kinematics_path = config["paths"]["kinematics"]
            data.sample_contract.update(prefix_min_frames=self.prefix_contract["min_frames"],
                prefix_max_frames=self.prefix_contract["max_frames"],
                prefix_zero_probability=self.prefix_contract["zero_probability"])
            self.datasets = []
            for entry in data.datasets.val.values():
                options = OmegaConf.to_container(entry, resolve=True)
                options.pop("_target_", None)
                options.update(kinematics_path=config["paths"]["kinematics"],
                               motion_frames=config["stage10"]["generation_horizon"]["frames"])
                self.datasets.append(BumiClosedLoopStage1Dataset(**options))

    def loss(self, actor, count):
        from gem.closedloop.stage1_dataset import collate_stage1_training_samples
        if count < 1:
            raise ValueError("An empty BC shard must skip forward")
        device = next(actor.parameters()).device
        with self._rng_scope(actor):
            samples = []
            for _ in range(count):
                source = int(torch.multinomial(torch.tensor([.2, .35, .25, .2]), 1, generator=self.generator))
                dataset = self.datasets[source]
                samples.append(dataset[int(torch.randint(len(dataset), (1,), generator=self.generator))])
            batch = batch_to_device(collate_stage1_training_samples(samples), device)
            prediction = actor.training_forward(batch)
            losses = []
            for index in range(count):
                single = {key: value[index:index+1] if isinstance(value, (torch.Tensor, list, tuple)) else value
                          for key, value in batch.items()}
                if "B" in single:
                    single["B"] = 1
                loss, _ = self.losses(single, prediction["pred_x_start"][index:index+1],
                    prediction["static_conf_logits"][index:index+1],
                    global_step=self.source_step+self.bc_update_steps)
                losses.append(loss)
            result = torch.stack(losses)
            if not torch.isfinite(result).all():
                raise FloatingPointError("Nonfinite original Stage1 supervision")
            return result

