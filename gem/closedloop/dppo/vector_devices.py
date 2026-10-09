"""为Isaac多进程绑定全局GPU编号，避免CUDA可见设备重映射与Vulkan枚举冲突。

训练rank仍沿用torchrun的CUDA_VISIBLE_DEVICES顺序；物理子进程使用PCI顺序的
全局设备号，不设置单卡可见掩码。AppLauncher据cuda:N同时选择计算、PhysX和
图形设备，复用Isaac Lab原分布式启动的编号规则。这里只解析已分配八卡，不加载
torch、不创建CUDA上下文、不扩展任务使用的设备集合；后端报告实际UUID供验收。
新编号合同写入配置及恢复身份，不能对旧掩码合同的checkpoint透明续训。
"""
from __future__ import annotations
import os

VECTOR_DEVICE_CONTRACT='isaac_global_gpu_index.v1'


def bind_vector_device(config, rank):
    tokens=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if (len(tokens)!=8 or len(set(tokens))!=8 or
            any(not token.strip().isdigit() or not 0<=int(token)<8 for token in tokens) or
            type(rank)is not int or not 0<=rank<8):
        raise ValueError('Explicit eight distinct Server1 GPU indices required')
    device=int(tokens[rank])
    config['runtime'].update(physics_device=f'cuda:{device}',vector_device_contract=VECTOR_DEVICE_CONTRACT)
    return dict(CUDA_VISIBLE_DEVICES=None,CUDA_DEVICE_ORDER='PCI_BUS_ID')
