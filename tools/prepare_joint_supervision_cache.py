"""一次性CPU监督缓存生成入口：复用已存在数据，绝不启动Newton或WM。

原四来源逐source处理，线程仅负责确定性CPU计算和原子文件写入；完成后发布包含SHA的
总索引。已完成文件可校验后复用，临时文件不进入索引；输出目录必须专用于该缓存。
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from gem.closedloop.joint_physics.config import read_config, ROOT
from gem.closedloop.joint_physics.data import SourceCatalog
from gem.closedloop.joint_physics.supervision_cache import identity, write_source
from gem.robots.bumi.endecoder import BumiEndecoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = read_config(args.config)
    cfg['data'].get('loading', {}).pop('supervision_cache', None)
    catalog = SourceCatalog(cfg, 'train')
    contract = identity(cfg, catalog)
    options = dict(cfg['stage1']['endecoder']); options['kinematics_path'] = str(ROOT/options['kinematics_path'])
    endecoder = BumiEndecoder(**options)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    started, sources = time.perf_counter(), {}
    def prepare(item):
        return write_source(output, *item, endecoder, contract)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for key, row in pool.map(prepare, catalog.sources()):
            sources[key] = row
            if len(sources)%100 == 0:
                print(json.dumps({'sources':len(sources), 'seconds':time.perf_counter()-started}), flush=True)
    value = {'contract': contract, 'sources': sources, 'seconds': time.perf_counter()-started,
             'bytes': sum(r['bytes'] for r in sources.values())}
    temporary = output/'index.partial'
    temporary.write_text(json.dumps(value, ensure_ascii=False))
    temporary.replace(output/'index.json')
    print(json.dumps({'complete':len(sources), 'bytes':value['bytes'], 'seconds':value['seconds']}), flush=True)

if __name__ == '__main__':
    main()
