"""按八卡全局参数梯度校准物理权重，不按Loss标量凑占比。

监督梯度包含原Stage1全部损失及加权Future辅助监督；物理梯度来自实际接受的学生H32
BPTT，先按全局有效样本数归一化。周期性测量两条梯度经过DDP平均后的范数与夹角，
使lambda*物理范数/(监督范数+lambda*物理范数)渐进接近目标。只有物理更新步使用
此口径，不声称它等同于AdamW参数变化比例或真实Newton收益。

校准步的整个前向/反向必须处于DDP.no_sync中：autograd.grad不交给DDP reducer，
最后联合backward再显式平均实际参数梯度。其他步继续原DDP路径。权重控制器不消费
随机数，完整保存EMA、起点和校准游标；零/非有限梯度不会被放大为伪物理信号。
"""

import copy
import math

import torch
from torch import distributed as dist


def _average(vector, world_size):
    if world_size > 1:
        dist.all_reduce(vector)
        vector.div_(world_size)
    return vector


def _gradient_vector(loss, named, world_size):
    gradients = torch.autograd.grad(loss, [p for _, p in named], retain_graph=True, allow_unused=True)
    vector = torch.cat([g.detach().reshape(-1).float() if g is not None else
                        torch.zeros(p.numel(), device=p.device, dtype=torch.float32)
                        for (_, p), g in zip(named, gradients)])
    return _average(vector, world_size)


def global_gradient_measurement(supervised, execution, actor, world_size, groups=()):
    """execution已乘world_size/global_valid_count；两个向量再执行DDP同义平均。"""
    named = [(n, p) for n, p in actor.named_parameters() if p.requires_grad]
    data = _gradient_vector(supervised, named, world_size)
    physics = _gradient_vector(execution, named, world_size)
    finite = torch.isfinite(data).all() & torch.isfinite(physics).all()
    norms = torch.stack([data.norm(), physics.norm(), torch.dot(data, physics), finite.float()]).tolist()
    if not norms[3] or not all(math.isfinite(v) for v in norms[:3]):
        raise FloatingPointError('全局监督或执行参数梯度非有限，未执行优化')
    module_norms = {}
    offset = 0
    sums = {group: data.new_zeros(2) for group in groups}
    for name, parameter in named:
        end = offset + parameter.numel()
        for group in groups:
            if name.startswith(group):
                sums[group] += torch.stack((data[offset:end].square().sum(), physics[offset:end].square().sum()))
        offset = end
    for group, value in sums.items():
        d, p = value.sqrt().tolist()
        module_norms[group] = {'supervised_norm': d, 'execution_norm': p}
    return {'supervised_norm': norms[0], 'execution_norm': norms[1],
            'cosine': norms[2] / max(1e-30, norms[0]*norms[1]), 'modules': module_norms}


def average_parameter_gradients(actor, world_size):
    """仅用于no_sync校准步；保持全局未用参数grad=None与AdamW原语义。"""
    if world_size <= 1:
        return
    parameters = [p for p in actor.parameters() if p.requires_grad]
    used = torch.tensor([p.grad is not None for p in parameters], device=parameters[0].device, dtype=torch.int32)
    dist.all_reduce(used, op=dist.ReduceOp.MAX)
    used = used.tolist()
    active = [p for p, flag in zip(parameters, used) if flag]
    if not active:
        return
    vector = torch.cat([p.grad.reshape(-1) if p.grad is not None else torch.zeros_like(p).reshape(-1) for p in active])
    _average(vector, world_size)
    offset = 0
    for p in active:
        view = vector[offset:offset+p.numel()].view_as(p)
        if p.grad is None:
            p.grad = view.clone()
        else:
            p.grad.copy_(view)
        offset += p.numel()


class GradientBalance:
    schema = 'global_parameter_gradient_share.v1'

    def __init__(self, config, initial_weight, start_step=0):
        self.config = copy.deepcopy(config)
        c = self.config
        if not (0 < c['target_share'] < c['maximum_measured_share'] < 1
                and c['ramp_steps'] > 0 and c['every_steps'] > 0
                and 0 <= c['ema_beta'] < 1 and c['max_change_factor'] > 1
                and 0 < c['min_weight'] <= initial_weight <= c['max_weight']):
            raise ValueError('物理梯度占比、渐进/测量周期、EMA或权重边界非法')
        self.start_step = int(start_step)
        self.weight = float(initial_weight)
        self.initial_share = None
        self.last_step = None
        self.ema_data = self.ema_physics = None
        self.metrics = {}

    def due(self, step):
        return self.last_step is None or step-self.last_step >= self.config['every_steps']

    def observe(self, step, measurement, valid_count):
        self.last_step = int(step)
        d, p = measurement['supervised_norm'], measurement['execution_norm']
        if not all(math.isfinite(v) and v >= 0 for v in (d, p)):
            raise FloatingPointError('不能用非有限参数梯度更新物理权重')
        before = self.weight*p/max(1e-30, d+self.weight*p)
        if valid_count == 0 or min(d, p) <= 1e-8:
            self.metrics = {**measurement, 'measurement_step': step, 'status': 'no_nonzero_global_gradient',
                            'valid_samples': valid_count, 'measured_share': before, 'lambda_exec': self.weight}
            return self.weight
        if self.initial_share is None:
            self.initial_share = before
        beta = self.config['ema_beta']
        self.ema_data = d if self.ema_data is None else beta*self.ema_data+(1-beta)*d
        self.ema_physics = p if self.ema_physics is None else beta*self.ema_physics+(1-beta)*p
        progress = min(1., max(0., (step-self.start_step)/self.config['ramp_steps']))
        target = self.initial_share + progress*(self.config['target_share']-self.initial_share)
        requested = target/(1-target)*self.ema_data/self.ema_physics
        change = self.config['max_change_factor']
        updated = min(self.weight*change, max(self.weight/change, requested))
        updated = min(self.config['max_weight'], max(self.config['min_weight'], updated))
        # 仅保证本次实测批次的上限；未校准批次仍由原全局梯度裁剪保护。
        cap = self.config['maximum_measured_share']
        updated = min(updated, cap/(1-cap)*d/p)
        self.weight = float(updated)
        self.metrics = {**measurement, 'measurement_step': step, 'status': 'calibrated',
            'valid_samples': valid_count, 'scope': 'physics_updates_only', 'target_share': target,
            'final_target_share': self.config['target_share'], 'ramp_progress': progress,
            'share_before_calibration': before, 'measured_share': self.weight*p/(d+self.weight*p),
            'lambda_exec': self.weight, 'requested_lambda_exec': requested,
            'ema_supervised_norm': self.ema_data, 'ema_execution_norm': self.ema_physics,
            'weight_limited': not math.isclose(self.weight, requested, rel_tol=1e-7)}
        return self.weight

    def state_dict(self):
        return copy.deepcopy({'schema': self.schema, **self.__dict__})

    def load_state_dict(self, state):
        state = copy.deepcopy(state)
        if state.pop('schema') != self.schema or state['config'] != self.config:
            raise ValueError('物理梯度控制器恢复合同不一致')
        self.__dict__.update(state)
