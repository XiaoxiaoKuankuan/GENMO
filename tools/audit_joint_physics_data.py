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
    args = parser.parse_args()
    cfg = read_config(args.config,args.set)
    import torch
    torch.set_num_threads(cfg['runtime']['cpu_threads'])
    adapter = build_adapter(cfg,'cpu')
    from gem.closedloop.joint_physics.prepare_rollouts import asset_contract
    catalog = SourceCatalog(cfg,args.split)
    store = SidecarStore(cfg['data']['sidecar'],args.split,asset_contract(cfg,adapter),sample_contract=cfg['data']['sample_contract'])
    result = {str(h):PhysicsRequestStream(catalog,store,adapter.bundle.spec.joint_names,h,cfg['training']['seed']).inventory for h in (16,32,64)}
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    (output/'inventory.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
