"""仅用于显式反向候选的cuBLAS残差补偿矩阵乘。

输入和最终输出保持FP32，将每个输入拆成BF16高位、低位及可选尾位，用3或6项
Tensor Core乘积直接累计到最终FP32矩阵。没有逐样本完整权重梯度，没有BF16输出
舍入，也不修改采样前向、old概率、主参数或Adam。它是需要全梯度回归的数值候选，
不是宣称等价的默认优化；失败时报告，不自动降低精度门槛或回退成另一条路径。

当前PyTorch 2.7的mm没有独立输出dtype接口，因此调用安装环境的cuBLAS GemmEx。
每线程/设备拥有独立handle，绑定当前CUDA stream；显式FP32计算和禁止低精度中间
归约，输入支持行/列连续转置。全部内存由PyTorch管理，没有设备同步或CPU数据拷贝。
注册的纯算子及fake实现允许AOT编译反向；不提供二阶导数，不用于独立可微前向。
"""
from __future__ import annotations

import atexit
import ctypes
from pathlib import Path
import threading

import torch

_LOCAL = threading.local()
_HANDLES = []
_LIBRARY = None


def _library():
    global _LIBRARY
    if _LIBRARY is None:
        path = Path(torch.__file__).resolve().parents[1]/'nvidia/cublas/lib/libcublas.so.12'
        lib = ctypes.CDLL(str(path))
        ptr, integer = ctypes.c_void_p, ctypes.c_int
        lib.cublasCreate_v2.argtypes = [ctypes.POINTER(ptr)]
        lib.cublasDestroy_v2.argtypes = [ptr]
        lib.cublasSetStream_v2.argtypes = [ptr, ptr]
        lib.cublasSetMathMode.argtypes = [ptr, integer]
        lib.cublasSetPointerMode_v2.argtypes = [ptr, integer]
        lib.cublasGemmEx.argtypes = [ptr, integer, integer, integer, integer, integer,
            ptr, ptr, integer, integer, ptr, integer, integer, ptr, ptr, integer,
            integer, integer, integer]
        _LIBRARY = lib
    return _LIBRARY


def _check(status):
    if status != 0:
        raise RuntimeError(f'Explicit compensated cuBLAS GEMM failed with status {status}')


def _handle(device):
    lib = _library()
    handles = getattr(_LOCAL, 'handles', None)
    if handles is None: handles = _LOCAL.handles = {}
    if device not in handles:
        handle = ctypes.c_void_p()
        _check(lib.cublasCreate_v2(ctypes.byref(handle)))
        _check(lib.cublasSetPointerMode_v2(handle, 0))  # alpha/beta为host标量。
        _check(lib.cublasSetMathMode(handle, 16))  # DISALLOW_REDUCED_PRECISION_REDUCTION。
        handles[device] = handle
        _HANDLES.append(handle)
    handle = handles[device]
    _check(lib.cublasSetStream_v2(handle, torch.cuda.current_stream(device).cuda_stream))
    return lib, handle


@atexit.register
def _close_handles():
    if _LIBRARY is not None:
        for handle in _HANDLES: _LIBRARY.cublasDestroy_v2(handle)
        _HANDLES.clear()


def _transpose_operand(value):
    # row-major [M,K]正是column-major [K,M]；支持权重梯度常见的转置view。
    if value.stride(1) == 1: return value, 0, value.stride(0)
    if value.stride(0) == 1: return value, 1, value.stride(1)
    value = value.contiguous()
    return value, 0, value.stride(0)


@torch.library.custom_op('stage10_numeric::compensated_gemm', mutates_args=())
def compensated_gemm(left: torch.Tensor, right: torch.Tensor, products: int) -> torch.Tensor:
    if (left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]
            or left.dtype != torch.float32 or right.dtype != torch.float32
            or not left.is_cuda or left.device != right.device or products not in (3,6)):
        raise ValueError('Compensated GEMM requires CUDA FP32 matrices and 3/6 products')
    m,k = left.shape; n = right.shape[1]
    if min(m,n,k) == 0: return left.new_zeros(m,n)
    with torch.cuda.device(left.device):
        def parts(value):
            high = value.to(torch.bfloat16)
            rest = value-high.float()
            low = rest.to(torch.bfloat16)
            return (high,low,(rest-low.float()).to(torch.bfloat16)) if products == 6 else (high,low)
        a,b = parts(left),parts(right)
        order = [(2,0),(1,1),(0,2),(1,0),(0,1),(0,0)] if products == 6 else [(1,0),(0,1),(0,0)]
        output = left.new_empty(m,n)
        lib,handle = _handle(left.device.index)
        alpha = ctypes.c_float(1.)
        for index,(i,j) in enumerate(order):
            x,tx,ldx = _transpose_operand(b[j])
            y,ty,ldy = _transpose_operand(a[i])
            beta = ctypes.c_float(float(index != 0))
            # C^T = B^T A^T，BF16输入(14)，FP32输出(0)，FP32计算(68)。
            _check(lib.cublasGemmEx(handle,tx,ty,n,m,k,ctypes.byref(alpha),
                x.data_ptr(),14,ldx,y.data_ptr(),14,ldy,ctypes.byref(beta),
                output.data_ptr(),0,n,68,-1))
        return output


@compensated_gemm.register_fake
def _fake(left, right, products):
    return left.new_empty(left.shape[0],right.shape[1])
