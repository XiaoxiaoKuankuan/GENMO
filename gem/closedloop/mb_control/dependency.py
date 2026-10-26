"""按实际参考支持点计算冻结WM的最长调用依赖，不按计划提交时刻提前累加。

初始真实快照的状态、历史和缓存深度为零；一次WM调用在实际状态、SM历史及当前
目标所依赖的最长路径后加一。每份新计划保存其生成条件的深度，只有读取该版本
参考支持点时才进入路径。已提交但尚未影响控制的未来计划不增加当前状态深度。
前缀、速度halo与未来十帧使用ReferenceBuffer的权威支持集；整数版本仅用于计数，
不detach模型张量、不缩短BPTT、不改变参考或调度，也不自行授权训练时域。
"""


class ReferenceDependencies:
    def __init__(self, initial_reference):
        self.version_depths = {int(v): 0 for v in initial_reference.source_versions.unique().tolist()}

    def register(self, reference, depth):
        self.version_depths[reference.version] = int(depth)

    def source_depth(self, reference, ticks):
        lo, hi, fraction = reference._indices(ticks)
        versions = reference.source_versions[lo].tolist()
        versions.extend(reference.source_versions[hi[fraction != 0]].tolist())
        return max((self.version_depths[v] for v in versions), default=0)

    def control_depth(self, reference, tick, future_steps=10):
        return self.source_depth(reference, reference.dependency_support(tick, future_steps))

    def prefix_depth(self, reference, ticket):
        ticks = reference.ticks.new_tensor([ticket.request_tick + i * 20 for i in range(ticket.prefix_frames)])
        return self.source_depth(reference, ticks)

    def cache_depth(self, reference):
        return max((self.version_depths[v] for v in reference.source_versions.unique().tolist()), default=0)
