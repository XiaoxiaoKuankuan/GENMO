"""GPU共享场景的上层执行适配：根据真实剩余参考协商最晚提交时间。

共享物理场景里，提前采满的环境仍持续执行旧参考，剩余窗口可能短于启动校准的
保守延迟预算。固定使用该预算会在仍有可执行参考时错误要求不存在的未来前缀。
本适配仅把请求deadline收紧到真实可用边界：仍保留GMT的10控制步lookahead、
下一控制点差分支持和30Hz源网格；不在本模块修改上层时钟确定的到达时刻，也不
放宽commit的late_plan检查。实际生成超过更紧deadline仍按原规则拒绝。

preview和真实reserve使用相同快照/公式，Critic与下一次Actor得到一致的前缀。
旧单环境路径不变。该协商策略有独立版本，必须写入GPU运行及断点身份，并重新
建立GPU闭环基线；不能将改变前缀长度后的结果冒充旧CPU运行的等价恢复。
"""
from __future__ import annotations
from .env_adapter import UpperEnvironment,ExecutionIntegrityError

DEADLINE_CONTRACT='available_reference_deadline_cap.v1'


def bounded_reference_deadline(snapshot, requested, *, minimum_prefix=12):
    tick=int(snapshot['tick'])
    source_end=int(snapshot['source_end_tick'])
    reference_end=int(snapshot['reference_valid_end_tick'])
    # ceil(protected+12,20)<=source_end；deadline需额外保护10*12 tick。
    maximum=((min(reference_end,source_end-12)-120)//12)*12
    if source_end<tick+(minimum_prefix-1)*20 or maximum<tick:
        raise ExecutionIntegrityError('No causal reference horizon for a new GPU decision')
    deadline=min(requested,maximum)
    return deadline,dict(contract=DEADLINE_CONTRACT,requested_deadline_tick=requested,
        effective_deadline_tick=deadline,maximum_supported_deadline_tick=maximum,
        reduced=deadline<requested,source_end_tick=source_end,reference_valid_end_tick=reference_end)


class VectorUpperEnvironment(UpperEnvironment):
    def _execution_boundary(self, generated, arrival, ready_tick):
        from .vector_boundary import fragment_reference_boundary,FRAGMENT_CONTRACT
        supported=fragment_reference_boundary(self.snapshot)
        # 迟于已承诺deadline的票据不可能安装。到达前仅执行现有真实参考，最多到
        # 仍具有合法bootstrap条件的网格点；不得在参考耗尽后虚构GMT输入。
        cannot_extend=generated['prepared'] is None or arrival>generated['meta']['deadline_tick']
        if cannot_extend and ready_tick>supported:
            return dict(tick=supported,reason='reference_horizon_truncated',contract=FRAGMENT_CONTRACT,
                simulated_arrival_tick=arrival,deadline_tick=generated['meta']['deadline_tick'],
                pending_plan_scope='discarded_at_administrative_boundary_without_commit')
        return None

    def step(self, **kwargs):
        from .vector_boundary import fragment_reference_boundary,FRAGMENT_CONTRACT
        row=super().step(**kwargs)
        if not kwargs.get('deterministic',False) and not row.terminated:
            if fragment_reference_boundary(self.snapshot)==self.snapshot['tick']:
                row.truncated=True;row.reason='reference_horizon_truncated'
                row.metadata['reference_boundary_reset']=dict(contract=FRAGMENT_CONTRACT,tick=self.snapshot['tick'])
            if not row.executed_control_steps:
                raise ExecutionIntegrityError('GPU collection cannot count a zero-control administrative action')
        return row

    def _request(self):
        request=super()._request()
        request['deadline_tick'],self.deadline_diagnostic=bounded_reference_deadline(
            self.snapshot,request['deadline_tick'],minimum_prefix=request['min_prefix'])
        return request

    def generate(self,**kwargs):
        from .vector_generation import GENERATION_CONTRACT, generate_for_environment
        contract = self.config['runtime'].get('vector_generation_contract')
        if contract is not None and contract != GENERATION_CONTRACT:
            raise ValueError('Unknown vector generation pipeline')
        batched = contract is not None and hasattr(self.policy, 'owner') and not kwargs.get('deterministic', False)
        result = generate_for_environment(self) if batched else super().generate(**kwargs)
        result['timing']['reference_deadline']=dict(self.deadline_diagnostic)
        return result
