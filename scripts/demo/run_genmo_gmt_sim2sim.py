#!/usr/bin/env python3
"""本机GENMO＋冻结GMT的实时MuJoCo动力学入口。

--audio接受WAV/MP3或音频目录，--audio-dir明确指定递归音乐文件夹；
--playlist接受既有selection/manifest或音频清单；
--actor-checkpoint选择明确的Stage2权重，不追随latest。窗口通过空格/R/N/P/
1/2/G/Esc交互，默认只绑定Stage1基础模型和当前测试模型；单画面叠加半透明
蓝色GENMO运动学参考与实体动力学机器人；
G显示/隐藏参考；--headless用于有界自动验收；--record在执行结束后编码视频。
所有运行产物写入outputs/sim2sim下独立目录；不使用Isaac、不启动训练、不改模型。
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import signal
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/closedloop/sim2sim_mujoco.yaml')
    source=parser.add_mutually_exclusive_group()
    source.add_argument('--audio',type=Path,help='单首WAV/MP3或递归音频文件夹')
    source.add_argument('--audio-dir',type=Path,help='递归扫描目录中的WAV/MP3，按相对路径排序')
    source.add_argument('--playlist',type=Path)
    parser.add_argument('--actor-checkpoint',type=Path);parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--record',action='store_true');parser.add_argument('--headless',action='store_true')
    parser.add_argument('--mute',action='store_true',help='显式关闭声音设备；特征和音乐时钟照常工作')
    parser.add_argument('--max-seconds',type=float,help='仅本机验收的实际仿真总秒数上限，记为行政结束')
    parser.add_argument('--output',type=Path,help='须位于配置output_root内的未存在目录')
    args=parser.parse_args()
    if args.audio_dir is not None and not args.audio_dir.expanduser().is_dir():
        parser.error('--audio-dir必须是存在的文件夹')
    # 本机PortAudio以用户级依赖装在venv/lib，无需sudo或改系统环境。
    # 动态加载器只在进程启动时读取LD_LIBRARY_PATH，因此在导入模型前最多重启一次。
    local_library=Path(sys.prefix)/'lib'
    if not args.mute and (local_library/'libportaudio.so.2').exists():
        paths=os.environ.get('LD_LIBRARY_PATH','').split(os.pathsep)
        if str(local_library) not in paths:
            env=dict(os.environ,LD_LIBRARY_PATH=os.pathsep.join([str(local_library),*filter(None,paths)]))
            os.execve(sys.executable,[sys.executable,'-B',str(Path(__file__).resolve()),*sys.argv[1:]],env)
    if args.max_seconds is not None and args.max_seconds<=0:parser.error('--max-seconds必须大于0')
    if args.headless:os.environ.setdefault('MUJOCO_GL','egl')
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
    from gem.closedloop.sim2sim.config import load_config
    config=load_config(args.config)
    if args.mute:config['runtime']['audio']=False
    if args.actor_checkpoint:
        config['models']=[dict(name=args.actor_checkpoint.stem,checkpoint=str(args.actor_checkpoint.resolve()))]
    if not 1<=len(config['models'])<=3:raise ValueError('必须绑定1至3个明确模型')
    from gem.closedloop.sim2sim.music import load_playlist
    from gem.closedloop.sim2sim.runtime import Sim2SimRuntime
    from gem.closedloop.sim2sim.viewer import Sim2SimViewer,encode_video
    from gem.closedloop.sim2sim.diagnostics import write_json
    tracks=load_playlist(args.audio or args.audio_dir or args.playlist or config['playlist'],config['paths']['dataset_root'])
    root=Path(config['paths']['output_root']).resolve()
    output=(args.output or root/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])).resolve()
    if output==root or not output.is_relative_to(root):raise ValueError('产物须位于outputs/sim2sim下独立目录')
    runtime=Sim2SimRuntime(config,tracks,output,seed=args.seed,headless=args.headless,record=args.record,max_seconds=args.max_seconds)
    write_json(output/'launch.json',dict(argv=sys.argv,config=str(args.config.resolve()),cwd=os.getcwd()))
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:runtime.command('stop'))
    print('[OUTPUT] '+str(output),flush=True)
    runtime.start()
    try:
        if args.headless:
            while not runtime.finished.wait(.2):pass
        else:Sim2SimViewer(runtime).run()
    finally:runtime.close()
    if args.record:
        for job in runtime.video_jobs:
            print('[VIDEO] '+str(encode_video(*job,config)),flush=True)
    if runtime.failure:return 1
    if any(e['reason'].startswith(('realtime_','generation_','reference_')) for e in runtime.episodes):return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
