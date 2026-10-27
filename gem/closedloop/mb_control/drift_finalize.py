"""现有独立50更新实验的有界证据收尾器，不启动或停止任何训练。

等待指定目录的第50次完整真实评价落盘后，CPU重算同口径A/B/C/D并刷新报告。
输入是已经授权的现有实验，不创建新Actor更新、不修改优化器、不向训练发送信号。
超出等待预算、评价失败或报告失败都记录真实状态；记录两小时截止时刻，不能把
截止后完成的结果倒填为按时完成。保存日志和原始证据，不根据质量正负删除文件。
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--wait-seconds',type=int,default=3600)
    args=parser.parse_args();root=Path(args.root)
    output=root/'finalization_status.json'
    def record(**kwargs):
        output.write_text(json.dumps({'recorded_utc':datetime.now(timezone.utc).isoformat(),
            'training_started_or_stopped':False,**kwargs},ensure_ascii=False,indent=2))
    summary=root/'episode_continuation/evaluations/actor_000050/physical_summary.json'
    failure=root/'episode_continuation/evaluation_000050_failure.json'
    deadline=datetime.fromisoformat('2026-10-27T05:38:54+00:00').timestamp()
    started=time.monotonic();record(status='waiting_existing_evaluation_50',deadline_utc='2026-10-27T05:38:54+00:00')
    while not summary.exists():
        if failure.exists():record(status='real_evaluation_failed',failure=str(failure));return
        if time.monotonic()-started>args.wait_seconds:record(status='wait_budget_elapsed',evaluation_50_complete=False);return
        time.sleep(15)
    manifest=json.loads((root/'episode_multianchor/manifest.json').read_text())
    run=manifest['config']['diagnostic']['source_run']
    try:
        destination=root/'new50_trajectory_audit'
        if not destination.exists():
            with (root/'new50_trajectory_audit.log').open('w') as log:
                subprocess.run([sys.executable,'tools/diagnose_stage2_music_drift.py','--offline','--run',run,
                    '--output',str(destination),'--versions','50','--evaluation-folders',str(root/'new_evaluation_folders.json'),
                    '--skip-log-audit'],stdout=log,stderr=subprocess.STDOUT,check=True)
        if not (destination/'summary.json').exists():raise RuntimeError('既有50重算目录不完整，保留失败现场，不覆盖')
        with (root/'report_generation.log').open('a') as log:
            subprocess.run([sys.executable,'-m','gem.closedloop.mb_control.drift_document','--root',str(root)],stdout=log,stderr=subprocess.STDOUT,check=True)
        record(status='complete',evaluation_50_complete=True,summary=str(summary),
            evaluation_completed_utc=datetime.fromtimestamp(summary.stat().st_mtime,timezone.utc).isoformat(),
            evaluation_completed_within_two_hours=summary.stat().st_mtime<=deadline,
            evaluation_late_seconds=max(0.,summary.stat().st_mtime-deadline))
    except Exception as error:
        record(status='report_failed',error=repr(error),evaluation_50_complete=summary.exists())
        raise


if __name__=='__main__':main()
