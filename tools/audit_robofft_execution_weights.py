#!/usr/bin/env python3
"""导出真实 RoboFFT 固定采集数据的逐计划60帧用途与不可变时间权重。

输入为生产collect-only或capture-validation生成的fixed_data_validation/iteration目录。
先核验case及每rank rollout manifest/张量SHA，再按实际GMT来源账本重新计算时间权重，
与训练准备阶段固定在MCRecord中的权重逐元素精确核对。每条计划输出中心消费、仅窗口
消费、后继覆盖、边界预测标签、自由坐标数及60帧归一化权重；已知前缀为零权重。
汇总文件和JSONL明细均要求新路径，不修改来源数据，不载入模型、不启动GPU或物理，
也不进行优化。本工具的证明范围是已保留控制区间内的来源和权重一致性；窗口包含
前瞻与回看，未观察不等于永不消费，姿态插值支持不等于完整动力学因果导数。
有限复采与之前训练的统计相同也不自动证明所有历史输入逐字节相同；模型效果仍需
独立的完整歌曲PhysX配对评估。导出报告可保留，原始临时张量按任务清理约定处理。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch

from gem.closedloop.dppo.diagnostic_rollout import load_rollout
from gem.closedloop.dppo.training_checkpoint import file_sha256
from gem.closedloop.robofft.execution_weights import VERSION, weights_for_rows


def audit_rows(rows, settings, *, rank):
    """重算与已缓存权重必须逐元素一致；不重新抽MC或替换旧损失。"""
    computed, audits=weights_for_rows(rows,settings)
    entries=[]
    for row,expected,audit in zip(rows,computed,audits):
        if row.mc_record is None or row.mc_record.time_weights is None:
            raise ValueError('Execution audit requires real cached weighted MC records')
        row.mc_record.validate()
        if not torch.equal(expected,row.mc_record.time_weights):
            raise ValueError('Cached execution weights differ from the actual retained provenance')
        if row.metadata.get('execution_weighting')!=audit:
            raise ValueError('Cached per-plan consumption audit differs')
        entries.append(dict(rank=rank,identity=row.identity,audit=audit,
            free_coordinates_by_frame=row.free_mask.long().sum(-1).tolist(),
            normalized_frame_weights=expected.tolist()))
    return entries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixed-case',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plans',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.plans.exists():raise FileExistsError('Audit outputs must be new paths')
    if args.output.resolve()==args.plans.resolve():raise ValueError('Summary and plans require different paths')
    root=args.fixed_case.resolve(strict=True);case=json.loads((root/'case.json').read_text())
    if case.get('schema')!='genmo.stage2.robofft_fixed_learning_case.v1':
        raise ValueError('Actual RoboFFT fixed learning case required')
    if file_sha256(root/'learning_state.pt')!=case['learning_state_sha256']:
        raise ValueError('Fixed learning state SHA differs')
    all_entries=[];identities=[]
    for item in case['ranks']:
        directory=root/f"rank{item['rank']:02d}"
        if file_sha256(directory/'manifest.json')!=item['rollout_manifest_sha256']:
            raise ValueError('Rollout manifest SHA differs from fixed case')
        data=load_rollout(directory)
        identities.append(data['identity'].get('actor_initialization'))
        all_entries.extend(audit_rows(data['transitions'],
            data['identity']['training']['robofft']['execution_weighting'],rank=item['rank']))
    if len(all_entries)!=case['real_global_actions'] or any(x!=identities[0] for x in identities):
        raise ValueError('Audit must cover all real actions with one Actor source')
    audits=[v['audit'] for v in all_entries]
    totals={k:sum(v[k] for v in audits) for k in ('near_free_frames','middle_free_frames','far_free_frames',
        'raw_weight_mass','normalized_weight_mass','free_coordinate_count')}
    for field in ('current_generated_frames','window_only_generated_frames','unobserved_free_frames_overwritten_by_committed_child'):
        totals[field]=sum(len(v[field]) for v in audits)
    result=dict(schema='genmo.robofft.execution_weights_audit.v1',weight_version=VERSION,
        fixed_case=str(root),case_sha256=file_sha256(root/'case.json'),actions=len(audits),
        source_actor=identities[0],exact_cached_weights_recomputed=True,totals=totals,
        predicted_actions=sum(v['scope'].startswith('predicted') for v in audits),
        incomplete_source_lineage_actions=sum(not v['source_lineage_complete'] for v in audits),
        scope='real_retained_provenance_and_fixed_loss_weights_not_motion_quality',plans=str(args.plans.resolve()))
    args.plans.parent.mkdir(parents=True,exist_ok=True);args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.plans.open('x') as handle:
        for entry in all_entries:handle.write(json.dumps(entry,ensure_ascii=False,allow_nan=False)+'\n')
    result['plans_sha256']=file_sha256(args.plans)
    with args.output.open('x') as handle:json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(actions=result['actions'],exact_match=True,totals=totals),ensure_ascii=False))


if __name__=='__main__':main()
