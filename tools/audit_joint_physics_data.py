"""统计原数据与v2离线请求覆盖的独立命令。

读取配置、原split清单及不可变离线分片，分别给出H16/H32/H64合法窗口、缺缓存来源和
运动/真实接触类型数量。只进行CPU metadata与张量统计，不运行Newton、不生成训练梯度，
也不把数据覆盖统计视作世界模型认证。输出使用新建目录，保留先前报告。
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from gem.closedloop.joint_physics.config import read_config, build_adapter
from gem.closedloop.joint_physics.data import SourceCatalog, SidecarStore
from gem.closedloop.joint_physics.requests import PhysicsRequestStream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--set',action='append',default=[])
    parser.add_argument('--split',choices=('train','val','test'),default='train')
    parser.add_argument('--output',required=True)
    parser.add_argument('--optimized-index', action='store_true', help='核对全库身份并持久化H1/H32只读请求表，不运行Newton/WM')
    args = parser.parse_args()
    cfg = read_config(args.config,args.set)
    cfg['data'].get('loading', {}).pop('supervision_cache', None)
    import torch
    torch.set_num_threads(cfg['runtime']['cpu_threads'])
    adapter = build_adapter(cfg,'cpu')
    from gem.closedloop.joint_physics.prepare_rollouts import asset_contract
    catalog = SourceCatalog(cfg,args.split)
    assets = asset_contract(cfg,adapter)
    store = SidecarStore(cfg['data']['sidecar'],args.split,assets,sample_contract=cfg['data']['sample_contract'],
                         cache_bytes=16*1024**3)
    if args.optimized_index:
        from gem.closedloop.joint_physics.contracts import source_key, execution_contract
        from gem.closedloop.joint_physics.execution import ValidatedSupport
        from gem.datasets.music_dance.music_dance_bumi import resolve_contract_path
        sources = []
        for dataset, row in catalog.sources():
            record = dataset.rows[row]
            meta = dict(dataset_id=dataset.dataset_name, split=dataset.split, sample_id=str(record['sample_id']))
            meta.update({field:str(resolve_contract_path(dataset.reader.root,record[field],field,meta['sample_id']))
                         for field in ('motion_path','music_feature_path')})
            episode = store.episode(meta)
            if not torch.isfinite(episode['state']).all() or not torch.isfinite(episode['reference_qpos']).all():
                raise ValueError('离线真实数据非有限: '+source_key(meta))
            sources.append(dict(source=source_key(meta), dataset=dataset.dataset_name, split=dataset.split,
                real_seconds=max(0,len(episode['state'])-1)/50, source_seconds=episode['source_frames']/30,
                future_valid=int(episode['offline_requests']['valid'].sum()),
                termination=episode.get('termination',episode.get('termination_reason'))))
        keys = [v['source'] for v in sources]
        if len(set(keys)) != len(keys) or set(keys) != set(store.index['sources']):
            raise ValueError('全库source有重复、缺失或额外来源')
        support = ValidatedSupport(cfg['physics']['validation_report'],assets,32,'cpu',execution_contract(cfg))
        result = {'sources':sources,'source_count':len(sources),'future_valid':sum(v['future_valid'] for v in sources),
                  'assets':assets,'sample_contract':cfg['data']['sample_contract'],'indexes':cfg['data']['sidecar']}
        for h in (32,1):
            stream = PhysicsRequestStream(catalog,store,adapter.bundle.spec.joint_names,h,cfg['training']['seed'],
                certification=support if h==32 else None, historical_mode='diagnostic',
                noise_levels=cfg['physics'].get('noise_levels') if h==32 else None,
                cache_directory=cfg['data']['loading']['request_cache'])
            result[str(h)] = stream.inventory
            result[f'H{h}_legal_requests'] = sum(v['valid'] for v in stream.inventory)
            print(json.dumps({'horizon':h,'requests':result[f'H{h}_legal_requests'],'sources':len(stream.keys)}),flush=True)
    else:
        result = {str(h):PhysicsRequestStream(catalog,store,adapter.bundle.spec.joint_names,h,cfg['training']['seed']).inventory for h in (16,32,64)}
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    (output/'inventory.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if isinstance(v,(int,float))}),flush=True)


if __name__=='__main__':
    main()
