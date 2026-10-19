"""参考计划的跨rollout因果账本和真实奖励区间审计。

只读取GMT已完成控制步返回的权威时间轴：窗口首次观察指控制开始时，GMT输入的
21个参考点首次含该plan；实际消费指这个窗口的当前中心点属于该plan。控制结束
的reward reference另行记录，不能将未来lookahead的时间误当作消费时间。600Hz
整数时钟均为本物理episode时间；计划保留生成它的行为版本，更新Actor不重置账本。

每轮只持久化发生变化的计划摘要，未出现的首次时间为null；不存逐步大数组。
账本不进入Actor/Critic，不搬移奖励，不改变GAE。完整checkpoint恢复后重建PhysX，
因此新episode重新建账，旧物理中间状态不伪恢复。过期计划从内存移除，已经写入的
摘要仍留在各轮审计文件。奖励区间审计拒绝重叠、重复事件和同episode内时间缺口。
"""
from __future__ import annotations
import copy

CREDIT_CONTRACT='genmo.reference_plan_credit.actual_clock.v1'


class ReferenceCreditLedger:
    def __init__(self):
        self.episode=None;self.records={};self.dirty={};self.last_tick=None

    def reset(self, episode):
        for key,item in self.records.items():
            item['episode_closed']=True;self.dirty[key]=item
        self.records={};self.episode=str(episode);self.last_tick=0

    def register(self, plan_id, *, decision_tick, policy_version, parent_plan_id, horizon_ticks):
        key=str(plan_id)
        if key in self.records:raise ValueError('Duplicate reference plan identity')
        self.records[key]=dict(plan_id=key,episode_id=self.episode,policy_version=policy_version,
            parent_plan_id=parent_plan_id,decision_tick=int(decision_tick),
            last_possible_tick=int(decision_tick)+int(horizon_ticks)+120,
            commit_tick=None,first_window_observed_tick=None,first_current_consumed_tick=None,
            first_reward_reference_tick=None,episode_closed=False)
        self.dirty[key]=self.records[key]

    def committed(self, plan_id, tick):
        item=self.records[str(plan_id)]
        if item['commit_tick'] is not None:raise ValueError('Reference plan committed twice')
        item['commit_tick']=int(tick);self.dirty[str(plan_id)]=item

    def _mark(self, plan, field, tick):
        key=str(plan)
        if key not in self.records:
            if not key.startswith('bootstrap:'):raise ValueError('GMT observed an unregistered reference plan')
            self.register(key,decision_tick=0,policy_version=None,parent_plan_id=None,horizon_ticks=10**12)
            self.records[key]['commit_tick']=0
        item=self.records[key]
        if item['commit_tick'] is None or int(tick)<item['commit_tick']:
            raise ValueError('GMT observed a plan before confirmed commit')
        if item[field] is None:
            item[field]=int(tick);self.dirty[key]=item

    def observe(self, rows):
        for row in rows:
            begin,end=int(row['control_tick_begin']),int(row['tick'])
            if str(row['episode_id'])!=self.episode or begin!=self.last_tick or end!=begin+12:
                raise ValueError('Duplicate, missing, or cross-episode physical control in reference ledger')
            ticks=[int(t) for t in row['consumed_reference_ticks']]
            plans=[str(p) for p in row['consumed_plan_ids']]
            if ticks!=list(range(begin-120,begin+121,12)) or len(plans)!=21:
                raise ValueError('GMT command window must contain current and ten past/future controls')
            for plan in set(plans):self._mark(plan,'first_window_observed_tick',begin)
            self._mark(plans[10],'first_current_consumed_tick',begin)
            if int(row['reference_tick'])!=end:raise ValueError('Reward reference must match the completed control')
            reward_plan=str(row['reference_plan_id'])
            self._mark(reward_plan,'first_reward_reference_tick',end)
            self.last_tick=end
        # 所有可再次出现在GMT回看窗口的参考必须保留；结束后才清理内存。
        for key,item in list(self.records.items()):
            if item['last_possible_tick']<self.last_tick:self.records.pop(key)

    def drain_updates(self):
        result=copy.deepcopy(list(self.dirty.values()));self.dirty.clear();return result


def validate_reward_intervals(rows):
    """同一物理控制只能属于一条转移；终止事件不可重复，跨版本只切GAE而不搬奖励。"""
    last={};identities=set()
    for row in rows:
        identity=row.identity
        key=tuple(identity.get(k,0) for k in ('backend_session_id','env_id','episode_id'))
        begin,end=int(row.control_tick_begin),int(row.control_tick_end)
        if end-begin!=12*row.executed_control_steps or len(row.rewards)!=row.executed_control_steps:
            raise ValueError('Reward interval differs from actual 50Hz execution')
        token=(key,begin,end)
        if token in identities:raise ValueError('Repeated physical rewards or zero-step terminal event')
        identities.add(token)
        if key in last:
            previous_end,terminal=last[key]
            if previous_end!=begin or terminal:raise ValueError('Overlapping, missing, or post-terminal physical rewards')
        last[key]=(end,bool(row.terminated))
