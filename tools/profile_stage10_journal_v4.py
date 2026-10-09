"""服务器1八rank对真实advance回复重复测量JSON与二进制journal的数据通路。

同一份已封存物理回复只读解码，分别计量规范编码、SHA256、SQLite FULL事务与
带SHA校验的解码；逐字段核验dtype、shape、内容和完整四子步。每次事务使用独立
测试数据库保留真实原始执行身份，绝不伪造新物理执行或在正式journal追加回复。
8个rank同步开始，反映服务器同盘并发写入成本；不运行物理、不宣称局部倍数等于
整轮提速。测试数据库在完成逐字段验证后精确删除，只保留各rank统计和输入SHA。
"""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.distributed as dist
from gem.closedloop.dppo.buffer import StepJournal, EncodedJournalReply, _journal_value
from gem.closedloop.dppo.journal_codec import FORMAT, encode_binary, decode_payload
from tools.profile_closedloop_rpc import compare
from tools.train_closedloop_stage10_8gpu import _available_gpus


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reply',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args()
    rank,world=int(os.environ['RANK']),int(os.environ['WORLD_SIZE'])
    if world!=8 or rank!=int(os.environ['LOCAL_RANK']):raise ValueError('Exactly eight local ranks required')
    dist.init_process_group('gloo',timeout=timedelta(minutes=3))
    status=[None]
    if rank==0:
        try:
            if args.output_dir.exists():raise FileExistsError(args.output_dir)
            status[0]=dict(devices=_available_gpus())
        except Exception as error:status[0]=dict(error=str(error))
    dist.broadcast_object_list(status,0)
    if 'error' in status[0]:raise RuntimeError(status[0]['error'])
    torch.cuda.set_device(rank)
    torch.empty(1,device=f'cuda:{rank}')  # 显式绑定所属卡；下列journal工作仍诚实报告为CPU/磁盘。
    output=args.output_dir/f'rank{rank:02d}';output.mkdir(parents=True)
    raw=args.reply.read_bytes();value=decode_payload(raw)
    identity=StepJournal.encode_result(value).identity
    results={}
    for name,format,encode in [('json','json.v1',lambda x:json.dumps(_journal_value(x),ensure_ascii=False,
            sort_keys=True,separators=(',',':'),allow_nan=False).encode()),('binary',FORMAT,encode_binary)]:
        times=[]
        for repeat in range(12):
            path=output/f'{name}_{repeat}.sqlite'
            try:
                with StepJournal(path,format=format) as journal:
                    if journal.connection.execute('PRAGMA synchronous').fetchone()[0]!=2:raise ValueError('FULL required')
                    dist.barrier(); start=time.perf_counter()
                    payload=encode(value);encoded=time.perf_counter()
                    sha=hashlib.sha256(payload).hexdigest();hashed=time.perf_counter()
                    if not journal.append_encoded(EncodedJournalReply(identity,payload,sha,format)):raise ValueError('Missing fresh write')
                    written=time.perf_counter()
                    decoded=decode_payload(payload,sha256=sha);end=time.perf_counter()
                    compare(value,decoded)
                    stored=journal.connection.execute('SELECT payload,sha256 FROM replies').fetchone()
                    compare(value,decode_payload(stored[0],sha256=stored[1]))
                    if repeat>=2:times.append([encoded-start,hashed-encoded,written-hashed,end-written])
            finally:
                for suffix in ('','-wal','-shm'):
                    Path(str(path)+suffix).unlink(missing_ok=True)
        data=np.asarray(times)
        results[name]=dict(bytes=len(payload),sha256=sha,repeats=10,exact_fields_dtype_shape_values=True,
            **{key:dict(mean=float(data[:,i].mean()),p50=float(np.percentile(data[:,i],50)),p95=float(np.percentile(data[:,i],95)))
               for i,key in enumerate(('encoding_seconds','sha_seconds','full_transaction_seconds','decode_with_sha_seconds'))})
    report=dict(rank=rank,source_sha256=hashlib.sha256(raw).hexdigest(),cases=results)
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    reports=[None]*8;dist.all_gather_object(reports,report)
    if rank==0:(args.output_dir/'report.json').write_text(json.dumps(dict(scope='eight_concurrent_real_reply_journal_only',ranks=reports),ensure_ascii=False,indent=2)+'\n')
    dist.destroy_process_group()


if __name__=='__main__':main()
