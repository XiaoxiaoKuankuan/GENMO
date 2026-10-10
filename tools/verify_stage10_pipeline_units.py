"""服务器1八rank执行本阶段正确性与恢复回归，禁止在本地替代验收。

每个rank绑定自己的GPU，分别运行列式奖励、跨块缓存、抽样权重、完整KL、世界
分页和ACK、预算、恢复及归档相关测试；pytest临时文件仅在指定独立输出目录。
各rank的完整输出和退出码保存，任何rank失败导致入口失败。测试并不替代真实
2048条闭环吞吐，且必须由调用者先确认八卡空闲，不与其他GPU基准混跑。
"""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import timedelta
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gmt-repo',type=Path,required=True)
    a=p.parse_args();rank=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=8: raise ValueError('Requires Server1 eight ranks')
    torch.cuda.set_device(rank);torch.set_num_threads(1)
    dist.init_process_group('gloo',timeout=timedelta(minutes=20))
    directory=a.output/f'rank{rank:02d}';directory.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'tests/closedloop/dppo'))
    import pytest
    paths=['test_vector_reward_adapter.py','test_performance_v3.py','test_denoising_sampling.py',
        'test_world_flow_scale.py','test_vector_fragment_boundary.py','test_journal_binary_v4.py',
        'test_updater_v2.py','test_stage10_audit_recovery.py','test_stage10_archive_audit.py',
        'test_stage10_async_archives.py','test_stage10_timing_metrics.py','test_vector_production_contract.py']
    with (directory/'pytest.log').open('w') as log, redirect_stdout(log),redirect_stderr(log):
        code=pytest.main(['-q','-p','no:cacheprovider','--basetemp',str(directory/'tmp'),
            *[str(root/'tests/closedloop/dppo'/name) for name in paths],
            str(root/'tests/closedloop/test_closedloop_protocol.py'),
            str(root/'tests/closedloop/test_closedloop_protocol_disconnect.py'),
            str(a.gmt_repo/'tests/test_vector_scheduler.py'),
            str(a.gmt_repo/'tests/test_vector_execution_journal.py')])
    results=[None]*8;dist.all_gather_object(results,dict(rank=rank,exit_code=int(code)))
    if rank==0:(a.output/'report.json').write_text(json.dumps(results,indent=2))
    dist.destroy_process_group()
    return int(any(r['exit_code'] for r in results))


if __name__=='__main__':raise SystemExit(main())
