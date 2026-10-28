"""冻结Actor及独立WM的纯张量计算编译，保持原模块参数身份和状态字典。

仅替换同一对象的forward/step可调用函数，normalizer和参数仍属于原模块；不包装
成改变checkpoint键名的OptimizedModule。动态batch用于32至256反向微批量及真实
采集，训练/冻结WM的requires_grad不同图由PyTorch正确区分。SQP控制流、投影
证书、采集复位和Python报告不编译；任何数值/梯度对照必须使用原eager作参照。
"""

import torch


def compile_components(control, dynamics):
    if getattr(dynamics, '_components_compiled', False):
        return
    options = {'triton.cudagraphs': False}
    control.bundle.actor.forward = torch.compile(control.bundle.actor.forward, dynamic=True, fullgraph=True, options=options)
    dynamics.step = torch.compile(dynamics.step, dynamic=True, fullgraph=True, options=options)
    dynamics._components_compiled = True
