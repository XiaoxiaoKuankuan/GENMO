"""当前第二阶段的Stage1装配、因果价值快照、连续GAE与原监督BC共享实现。

Actor从s595000装配并保留固定位置表；Critic价值按对应历史和版本计算，真实终止
不自举，普通片段边界不拼接跨episode或行为版本的GAE。BC使用Stage1原数据和
损失以及独立可恢复RNG。Actor更新唯一入口为full_batch_actor，本模块不再保留
旧Stage9、坐标PPO或其他单卡训练器；资源和物理推进由当前主循环负责。
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
from .returns import compute_gae
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


def populate_values(transitions, critic, device, *, critic_version=None, batch_size=1):
    """固定版本价值快照；终止置零，自举条件合批但不跨环境重组GAE。"""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('value batch_size must be positive')
    critic.eval()
    work = []
    for row in transitions:
        work.append((row, 'old_value', row.metadata['critic_context'], row.metadata['remaining_music_seconds']))
        row.next_value = 0.
        if row.next_context is not None and not row.terminated:
            work.append((row, 'next_value', row.metadata['next_critic_context'], row.metadata['next_remaining_music_seconds']))
    with torch.no_grad(), measure('critic.fixed_value_snapshot', gpu=True):
        for start in range(0, len(work), batch_size):
            chunk = work[start:start+batch_size]
            context = {key: torch.cat([item[2][key] for item in chunk]).to(device) for key in chunk[0][2]}
            remaining = torch.tensor([item[3] for item in chunk], device=device)
            values = critic(context, remaining).detach().cpu().tolist()
            if not all(math.isfinite(value) for value in values):
                raise FloatingPointError('Nonfinite frozen Critic snapshot')
            for item, value in zip(chunk, values):
                setattr(item[0], item[1], value)
    if critic_version is not None:
        for row in transitions:
            row.metadata['value_snapshot_version'] = critic_version


def fixed_targets(transitions, critic, device, *, gamma_upper=.99, lambda_upper=.95,
                  reuse_values=False, critic_version=None, normalize=False):
    if reuse_values:
        if critic_version is None:
            raise ValueError('Reusing populated values requires an explicit Critic version')
        for row in transitions:
            if row.metadata.get('value_snapshot_version') != critic_version:
                raise ValueError('Stale or missing Critic value snapshot version')
            if not math.isfinite(row.old_value) or not math.isfinite(row.next_value):
                raise FloatingPointError('Nonfinite cached Critic values')
    else:
        populate_values(transitions, critic, device, critic_version=critic_version)
    continuation = [False]*len(transitions)
    for i in range(len(transitions)-1):
        a,b=transitions[i:i+2]
        continuation[i] = (a.transition_valid and b.transition_valid and not a.terminated and not a.truncated
            and all(a.identity[k]==b.identity[k] for k in ('backend_session_id','episode_id','policy_version'))
            and a.identity.get('env_id',0)==b.identity.get('env_id',0)
            and a.control_tick_end==b.control_tick_begin)
    return compute_gae(rewards=[t.rewards for t in transitions],values=[t.old_value for t in transitions],
        next_values=[t.next_value for t in transitions],executed_steps=[t.executed_control_steps for t in transitions],
        bootstrap_mask=[not t.terminated and t.next_context is not None for t in transitions],
        continuation_mask=continuation,valid=[t.transition_valid for t in transitions],
        event_rewards=[t.metadata.get('event_reward',0.) for t in transitions],
        gamma_upper=gamma_upper,lambda_upper=lambda_upper,normalize=normalize)


def _validate_actor_transitions(policy, transitions):
    """拒绝不属于当前随机核的链；不允许用参数版本相同掩盖条件或掩码漂移。"""
    if not transitions:
        raise ValueError('Actor diagnostics require at least one transition')
    for item in transitions:
        if not item.transition_valid:
            raise ValueError('Invalid transitions cannot enter Actor optimization')
        from ..robofft import ALGORITHM
        if getattr(policy,'algorithm',None)==ALGORITHM:
            from ..robofft.records import validate_action
            validate_action(item,policy)
            continue
        if item.chain.shape != (policy.steps + 1, item.context["future_valid"].shape[1], 30):
            raise ValueError('Rollout chain differs from the configured denoising steps')
        if item.chain.dtype != torch.float32 or item.old_log_prob.dtype != torch.float64:
            raise TypeError('Rollout chain/log probability precision changed')
        if item.old_log_prob.shape != (len(policy.stochastic_steps),):
            raise ValueError('Old log probabilities must match the nineteen stochastic transitions')
        expected = item.context['future_valid'][..., None] & ~item.context['known_qpos30_mask']
        if not torch.equal(item.free_mask.cpu(), expected[0].cpu()):
            raise ValueError('Stored free mask differs from the original conditions')
        trace = item.metadata['sampler_trace']
        if (trace['kernel_config'] != policy.kernel_config
                and trace['kernel_config'] != getattr(policy,'_diagnostic_behavior_kernel',None)):
            raise ValueError('Rollout and learner stochastic kernel configuration differs')
        if trace['probability_steps'].tolist() != list(policy.stochastic_steps):
            raise ValueError('Deterministic final step cannot have a probability density')
        if trace['timestep_map'].tolist() != list(policy.timestep_map):
            raise ValueError('Rollout timestep map differs from the learner')


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
