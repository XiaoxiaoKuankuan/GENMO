"""利用有限时域局部依赖精确组装SO(3)约束Jacobian和拉格朗日Hessian。

单段角速度只依赖相邻两帧的6个旋转变量，角加速度只依赖相邻三帧的9个变量。
分别在6/9/3维局部块上用原SO(3)公式自动微分，再通过固定线性映射组装30维
全局导数；历史锚点作为局部值输入，第11点halo通过2*d10-d9的同一映射进入。
该实现与原30维全量求导数学等价，不近似旋转、不放宽约束，也不改隐式KKT。
只对无分支的局部导数尝试GPU融合；编译失败可显式退回同一精确局部eager核，
最终输出仍须原FP32证书和KKT精度检查。参数VJP继续使用原约束的高阶autograd。
"""

import torch
from supermimic.rotations import so3_exp, so3_log


def build(projector, example, *, compile_kernels=False):
    unit = (projector.velocity[3:6]*.02).to(example)
    basis = torch.eye(30, device=example.device, dtype=example.dtype).reshape(30, 3, 10).permute(2, 1, 0)
    basis = basis*unit[None, :, None]
    maps = torch.cat((basis.new_zeros(2, 3, 30), basis, 2*basis[-1:]-basis[-2:-1]), 0)
    vmap = torch.cat((maps[1:-1], maps[2:]), 1)
    amap = torch.cat((maps[:-2], maps[1:-1], maps[2:]), 1)
    pmap = maps[2:12]

    def velocity(local, nominal):
        d = local.reshape(2, 3)
        corrected = nominal@so3_exp(d)
        value = so3_log(corrected[1]@corrected[0].T)/.02
        original = so3_log(nominal[1]@nominal[0].T)/.02
        return (value-original).norm()/projector.rotation_velocity-1

    def acceleration(local, nominal):
        d = local.reshape(3, 3)
        corrected = nominal@so3_exp(d)
        value = so3_log(corrected[1:]@corrected[:-1].transpose(-1, -2))/.02
        original = so3_log(nominal[1:]@nominal[:-1].transpose(-1, -2))/.02
        return ((value-original).diff(dim=0)/.02).norm()/projector.rotation_acceleration-1

    def amplitude(local, nominal):
        return local.norm()/.05-1

    functions = [velocity, acceleration, amplitude] if projector.constrain_root_angular_acceleration else [velocity, amplitude]
    mappings = [vmap, amap, pmap] if projector.constrain_root_angular_acceleration else [vmap, pmap]
    kernels = []
    for function in functions:
        gradient = torch.func.grad(function, argnums=0)
        jac = torch.func.vmap(gradient)
        hess = torch.func.vmap(torch.func.jacrev(gradient, argnums=0))
        kernels.append([jac, hess])
    if compile_kernels:
        def compile_one(function):
            compiled = torch.compile(function, dynamic=True, fullgraph=True, options={'triton.cudagraphs': False})
            failed = False
            def call(*values):
                nonlocal failed
                if failed:
                    return function(*values)
                for value in values:
                    for dim in range(1, value.ndim):
                        torch._dynamo.mark_static(value, dim)
                try:
                    return compiled(*values)
                except (torch._dynamo.exc.Unsupported, torch._dynamo.exc.BackendCompilerFailed) as error:
                    # 只降级编译能力，绝不吞掉求解器异常、OOM或非有限结果。
                    failed = True
                    projector.derivative_compile_fallback = str(error).splitlines()[0]
                    return function(*values)
            return call
        kernels = [[compile_one(f) for f in row] for row in kernels]

    def local_inputs(x, past, anchor, rotation):
        plan = x.reshape(-1, 3, 10).transpose(1, 2)*unit
        d = torch.cat((past[:, None], anchor[:, None], plan, 2*plan[:, -1:]-plan[:, -2:-1]), 1)
        local_v = torch.cat((d[:, 1:-1], d[:, 2:]), -1)
        rot_v = torch.stack((rotation[:, 1:-1], rotation[:, 2:]), 2)
        local_a = torch.cat((d[:, :-2], d[:, 1:-1], d[:, 2:]), -1)
        rot_a = torch.stack((rotation[:, :-2], rotation[:, 1:-1], rotation[:, 2:]), 2)
        rows = [(local_v, rot_v), (local_a, rot_a), (plan, rotation[:, 2:12])]
        return rows if projector.constrain_root_angular_acceleration else [rows[0], rows[2]]

    def jacobian(x, past, anchor, rotation):
        output = []
        for (local, nominal), mapping, (jac, _) in zip(local_inputs(x, past, anchor, rotation), mappings, kernels):
            j = jac(local.flatten(0, 1), nominal.flatten(0, 1)).reshape(len(x), len(mapping), -1)
            output.append(torch.einsum('bsa,sai->bsi', j, mapping))
        return torch.cat(output, 1)

    def hessian(x, past, anchor, rotation, multiplier):
        output = x.new_zeros(len(x), 30, 30)
        offset = 0
        for (local, nominal), mapping, (_, hess) in zip(local_inputs(x, past, anchor, rotation), mappings, kernels):
            h = hess(local.flatten(0, 1), nominal.flatten(0, 1)).reshape(len(x), len(mapping), local.shape[-1], local.shape[-1])
            weight = multiplier[:, offset:offset+len(mapping)]
            output = output+torch.einsum('sai,bsac,scj->bij', mapping, h*weight[..., None, None], mapping)
            offset += len(mapping)
        return output
    return torch.func.vmap(projector.angular_constraints), jacobian, hessian
