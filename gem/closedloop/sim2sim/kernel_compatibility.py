"""校验跨PyTorch版本加载的Stage2训练核与本机重建核。

服务器训练与本机sim2sim可以使用不同的PyTorch版本。冻结前段的content_sha256
除了权重和DDIM系数，还绑定包含torch_version的运行设置；只删除外层版本字段
仍会因内层摘要不同而错误拒绝加载。本模块仅将本机完整内容重新按来源版本标签
计算摘要，要求它与checkpoint中的原始摘要严格一致，随后比较所有其他核字段。

不会删除或放宽权重、系数、冻结步数、精度、CFG、Horizon和资产检查，也不改变
模型、采样配置或checkpoint。成功只代表结构及内容合同可重建，不承诺不同Torch
版本的浮点输出逐位一致；本机继续生成新的行为身份并独立进行时延标定。
"""
from __future__ import annotations

import copy
import hashlib
import json


def frozen_content_fingerprint(policy, *, torch_version=None):
    """按照冻结前段原合同重算内容摘要，可只替换摘要中的Torch版本元数据。"""
    settings = json.loads(policy._frozen_settings())
    if torch_version is not None:
        settings['kernel']['numerical_execution']['torch_version'] = torch_version
    digest = hashlib.sha256()
    model = policy.frozen_policy.actor
    for name, value in (list(model.named_parameters()) + list(model.named_buffers())
                        + list(policy._frozen_coefficients.items())):
        digest.update((name + str(value.dtype) + str(tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    digest.update(json.dumps(settings, sort_keys=True, separators=(',', ':')).encode())
    return digest.hexdigest()


def verify_source_kernel(source_kernel, policy):
    """只允许可完整证明的版本标签差异，其他核或冻结内容差异一律拒绝。"""
    old = copy.deepcopy(source_kernel)
    new = copy.deepcopy(policy.kernel_config)
    source_version = old['numerical_execution'].pop('torch_version', None)
    local_version = new['numerical_execution'].pop('torch_version', None)
    rebased = False
    old_prefix, new_prefix = old.get('frozen_prefix'), new.get('frozen_prefix')
    if (source_version is not None and local_version is not None and source_version != local_version
            and old_prefix and new_prefix
            and old_prefix.get('content_sha256') != new_prefix.get('content_sha256')):
        reconstructed = frozen_content_fingerprint(policy, torch_version=source_version)
        if reconstructed == old_prefix.get('content_sha256'):
            new_prefix['content_sha256'] = reconstructed
            rebased = True
    if old != new:
        raise ValueError('本机重建随机核与checkpoint声明不同')
    return dict(source_torch_version=source_version, local_torch_version=local_version,
                frozen_content_version_metadata_rebased=rebased,
                scope='完整核与冻结内容合同校验；跨Torch版本不声明浮点逐位等价')
