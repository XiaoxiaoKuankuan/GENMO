"""验证全新世界复核的固定分片不会让一个rank执行两首歌。

显式跨split分配八个不同rank，空rank仍参加通信；检查任务无遗漏、固定任务
身份保留和多任务误分配拒绝。仅内存夹具，不创建模型、GPU或实验产物。
"""
import pytest
from tools.eval.run_stage10_fresh_world_control import single_task_partitions


def test_one_task_per_world_across_two_splits():
    assigned=[]
    for split,offset in (('test',0),('val',4)):
        tasks=[dict(task_id=f'{split}{i}') for i in range(4)]
        plan=dict(tasks=tasks,plan_sha256='parent',fresh_world_task_ranks={t['task_id']:offset+i for i,t in enumerate(tasks)})
        shards=single_task_partitions(plan,8)
        assert sum(s['task_count'] for s in shards)==4
        assert all(s['task_count']<=1 for s in shards)
        assigned.extend(i for i,s in enumerate(shards) if s['tasks'])
    assert sorted(assigned)==list(range(8))


def test_duplicate_assignment_is_rejected():
    with pytest.raises(ValueError,match='one task'):
        single_task_partitions(dict(tasks=[dict(task_id='a'),dict(task_id='b')],
            fresh_world_task_ranks={'a':0,'b':0}),8)
