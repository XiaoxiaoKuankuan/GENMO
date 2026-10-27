"""把服务器全量Stage1评价的报告持续同步到用户本地目录。

只通过用户指定SSH主机读取本次独立评价根目录中的报告、统计与状态；不发送训练
控制命令，不同步大模型或修改远端轨迹。每60秒增量同步，收到CPU分析完成文件后
结束；网络失败保留日志并重试，最多24小时。远端任务未完成时绝不写成评价已完成。
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--host',required=True);p.add_argument('--remote',required=True);p.add_argument('--local',required=True)
    args=p.parse_args();local=Path(args.local).expanduser().resolve();local.mkdir(parents=True,exist_ok=True)
    names=['manifest.json','summary.json','per_music_results.json','optimization_space_analysis.json',
           'progress.json','supervisor.json','supervisor_complete.json','analysis_complete.json',
           'launch.json','contract_tests.log','previous_training_stop_verified.json',
           'Stage1全量音乐评价.html','Stage1优化空间分析.html']
    command=['rsync','-az']+[f'--include=/{name}' for name in names]+['--exclude=*',f'{args.host}:{args.remote.rstrip("/")}/',str(local)+'/']
    deadline=time.monotonic()+86400
    while time.monotonic()<deadline:
        try:
            run=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=55)
            status={'updated_utc':datetime.now(timezone.utc).isoformat(),'returncode':run.returncode,'message':run.stdout[-1000:]}
        except Exception as error:status={'updated_utc':datetime.now(timezone.utc).isoformat(),'error':repr(error)}
        if (local/'analysis_complete.json').exists() and status.get('returncode')==0:
            status['final_analysis']=json.loads((local/'analysis_complete.json').read_text());status['sync_finished']=True
        (local/'报告同步状态.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
        if status.get('sync_finished'):return
        time.sleep(60)
    (local/'报告同步超时.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'remote_job_not_stopped':True}))


if __name__=='__main__':main()
