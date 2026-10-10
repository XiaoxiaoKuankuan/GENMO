"""服务器1八rank使用同一2048条真实执行证据验证列式奖励与GAE。

完整核验已封存归档和本rank原始世界journal的SHA，从原音乐身份重建独立因果
窗口，原标量GPU奖励适配与候选逐字段比较。四个物理子步、终止、状态和时间戳
直接来自同一权威块，不重新仿真；比较包括真实奖励和完整回报/GAE，事件罚分保留。
计时分别覆盖列视图建立、旧逐步奖励和新列式奖励，不把回放微基准当成采样提速。
本工具要求首个rollout包含从600tick开始的真实环境历史；后续rollout需先重放
前序证据，缺历史时明确失败，不能用零历史替代。解包临时数据结束精确清理。
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from datetime import timedelta
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time
import numpy as np
import torch
import torch.distributed as dist
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.stage10_fixed_inputs import load_rank
from tools.train_closedloop_stage10 import configuration
from gem.runtime.trajectory_blocks import ColumnarTrace,unpack_trace
from gem.closedloop.dppo.journal_codec import decode_payload,encode_binary,encode_binary_reference
from gem.closedloop.dppo.vector_reward_adapter import VectorExecutionReward,compare_reward_evidence
from gem.closedloop.dppo.full_dataset import FullMusicCatalog
from gem.closedloop.dppo.target_activity import load_paired_activity
from gem.closedloop.dppo.trainer import fixed_targets


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('iteration','config','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--compare-codecs',action='store_true',help='核对全部原始帧字节并分别计时旧新编码，非端到端吞吐')
    p.add_argument('--compare-reward-bytes',action='store_true',
        help='额外核对全部奖励叶子的规范字节，包括浮点符号零；不放宽原数值门槛')
    a=p.parse_args();rank=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=8:raise ValueError('Requires Server1 eight ranks')
    torch.cuda.set_device(rank);torch.set_num_threads(1)
    dist.init_process_group('gloo',timeout=timedelta(minutes=30))
    directory=a.output/f'rank{rank:02d}';directory.mkdir(parents=True,exist_ok=False)
    rows,targets,source=load_rank(a.iteration,rank,directory/'unpacked',journals=True)
    cfg=configuration(a.config);root=cfg['paths']['data_root'];catalog=FullMusicCatalog(root)
    task_by_episode={r.identity['episode_id']:r.metadata['training_task'] for r in rows}
    expected={(r.identity['episode_id'],d['tick']):d for r in rows for d in r.metadata['reward_details']}
    calculators={};warm={};observed={};times=dict(column_view_seconds=0.,scalar_seconds=0.,columnar_seconds=0.)
    codec_times=dict(reference_seconds=0.,fast_seconds=0.,frames=0,bytes=0)
    reward_byte_controls=0
    journal=Path(source['rank_directory'])/'world_journal.sqlite'
    with sqlite3.connect(journal.resolve().as_uri()+'?mode=ro',uri=True) as db:
        for sha,payload in db.execute('SELECT sha256,payload FROM replies ORDER BY rowid'):
            value=decode_payload(payload,sha256=sha)
            if a.compare_codecs:
                start=time.perf_counter();old_bytes=encode_binary_reference(value);codec_times['reference_seconds']+=time.perf_counter()-start
                start=time.perf_counter();new_bytes=encode_binary(value);codec_times['fast_seconds']+=time.perf_counter()-start
                if old_bytes!=new_bytes or new_bytes!=payload:raise ValueError('Canonical journal bytes changed')
                codec_times['frames']+=1;codec_times['bytes']+=len(new_bytes)
                del old_bytes,new_bytes
            if not value.get('ok'):continue
            for reply in value['result'].get('replies',[]):
                env=reply.get('result',{})
                if not reply.get('ok') or not env.get('ok') or env.get('operation') not in ('advance','drain_fragment'):continue
                result=env['result'];blocks=result.get('trace_blocks',[])
                if 'trace_block' in result:blocks=[result['trace_block']]
                for block in blocks:
                    if not block['count']:continue
                    start=time.perf_counter();trace=ColumnarTrace([block]);times['column_view_seconds']+=time.perf_counter()-start
                    if a.compare_codecs and encode_binary_reference(list(trace))!=encode_binary_reference(unpack_trace(block)):
                        raise ValueError('Complete physical fields/dtype/signed bytes changed in column views')
                    episode=trace[0]['episode_id']
                    if episode not in task_by_episode:continue
                    if trace[-1]['tick']<=600:
                        warm[episode]=np.array(trace[-1]['joint_position_target'],copy=True);continue
                    if trace[0]['tick']<=600:raise ValueError('Unexpected warmup/reward mixed block')
                    if episode not in calculators:
                        if trace[0]['tick']!=612 or episode not in warm:raise ValueError('Missing original causal history')
                        task=task_by_episode[episode];sample=catalog._lookup[(task['split'],task['dataset'],task['sample_id'])]
                        if sample['manifest_sha256']!=task['manifest_sha256']:raise ValueError('Music manifest changed')
                        music=catalog.load_music(sample)[task['music_start_frame']:]
                        pair=[]
                        for _ in range(2):
                            target=load_paired_activity(root,sample,music_start_frame=task['music_start_frame'])
                            calc=VectorExecutionReward(cfg['stage10']['reward'],music,target_activity=target)
                            calc.seed_previous_target(warm[episode]);pair.append(calc)
                        calculators[episode]=pair
                    old,new=calculators[episode]
                    start=time.perf_counter();reference=[old.evaluate_step(row) for row in trace];times['scalar_seconds']+=time.perf_counter()-start
                    start=time.perf_counter();actual=new.evaluate_batch(trace);times['columnar_seconds']+=time.perf_counter()-start
                    if new.columnar_fallbacks:raise ValueError(f'Columnar path fell back: {new.last_columnar_fallback}')
                    for x,y in zip(actual,reference):
                        compare_reward_evidence(x,y)
                        if a.compare_reward_bytes:
                            if encode_binary_reference(x)!=encode_binary_reference(y):
                                raise ValueError(f'Complete reward canonical bytes changed: episode={episode}, tick={x["tick"]}')
                            reward_byte_controls+=1
                        key=(episode,x['tick'])
                        if key in expected:compare_reward_evidence(x,expected[key]);observed[key]=x
    if observed.keys()!=expected.keys():raise ValueError('Not all actual reward/control steps covered')
    replay=[];max_reward_difference=0.
    for row in rows:
        values=torch.tensor([observed[(row.identity['episode_id'],d['tick'])]['reward'] for d in row.metadata['reward_details']],dtype=torch.float64)
        if values.numel():values[-1]+=row.metadata.get('event_penalty_total',0.)
        torch.testing.assert_close(values,row.rewards,atol=1e-12,rtol=1e-12)
        if values.numel():max_reward_difference=max(max_reward_difference,float((values-row.rewards).abs().max()))
        replay.append(replace(row,rewards=values))
    old=fixed_targets(rows,None,'cpu',reuse_values=True,critic_version=rows[0].metadata['value_snapshot_version'],normalize=False)
    new=fixed_targets(replay,None,'cpu',reuse_values=True,critic_version=rows[0].metadata['value_snapshot_version'],normalize=False)
    for key in ('advantages_raw','returns','discounted_rewards'):torch.testing.assert_close(new[key],old[key],atol=1e-11,rtol=1e-11)
    report=dict(rank=rank,source=source,passed=True,controls=len(observed),episodes=len(calculators),timing=times,
        max_reward_difference=max_reward_difference,gae_max_abs_difference=float((new['advantages_raw']-old['advantages_raw']).abs().max()),
        columnar_fallbacks=0,all_reward_fields_compared=True,physical_controls_unchanged=True,
        full_reward_fields_byte_compared=a.compare_reward_bytes,reward_byte_controls=reward_byte_controls,
        full_physical_fields_byte_compared=a.compare_codecs,
        codecs=None if not a.compare_codecs else codec_times)
    (directory/'report.json').write_text(json.dumps(report,indent=2))
    reports=[None]*8;dist.all_gather_object(reports,report)
    if rank==0:(a.output/'report.json').write_text(json.dumps(dict(ranks=reports),indent=2))
    shutil.rmtree(directory/'unpacked');dist.destroy_process_group()


if __name__=='__main__':main()
