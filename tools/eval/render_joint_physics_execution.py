#!/usr/bin/env python3
"""把既定GENMO预览与真实Newton记录组成可直接浏览的三栏交付目录。

输入是同一清单的原始动作、固定模型生成动作及独立Newton执行输出。先核对生成文件SHA、
真实执行证据、一次reset和禁止WM替代的记录，再调用现有MuJoCo离屏渲染器；这里不重新
生成舞蹈、不运行动力学、不修改原双栏视频。复制原轨迹和音乐到独立目录，便于完整下载。
每首保留整曲原参考/生成画面；Newton提前终止之后由渲染器留空，禁止复制终帧伪装完成。

逐首核验H264/AAC、帧数、分辨率及完整解码，汇总真实完成率与结束原因，生成中文HTML和
ZIP。输出目录允许已有runtime/execution，但videos必须不存在，防止覆盖已有结果。
工具适合由独立systemd用户会话启动；不操作训练进程、不修改模型或任何训练配置。
"""

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def verify_video(path, frames):
    result = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
        '-show_streams', '-show_format', '-of', 'json', str(path)]))
    video = next(s for s in result['streams'] if s['codec_type'] == 'video')
    audio = next(s for s in result['streams'] if s['codec_type'] == 'audio')
    assert video['codec_name'] == 'h264' and audio['codec_name'] == 'aac'
    assert (video['width'], video['height'], video['r_frame_rate']) == (1920, 726, '30/1')
    assert int(video['nb_read_frames']) == frames
    assert abs(float(video['duration']) - frames/30) < .04
    subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(path), '-f', 'null', '-'], check=True)
    return dict(frames=frames, duration=frames/30, full_decode=True, sha256=digest(path), bytes=path.stat().st_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--execution', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mjcf', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.inputs/'inputs/manifest.json').read_text())
    execution = json.loads((args.execution/'manifest.json').read_text())
    assert execution['frozen_unchanged'] and execution['world_model_forward_calls'] == 0
    assert execution['max_control_steps'] is None
    assert execution['checkpoint_sha256'] == manifest['checkpoint_sha256']
    assert execution['reference_manifest_sha256'] == digest(args.inputs/'inputs/manifest.json')
    results = {s['number']: s for s in execution['sources']}
    assert set(results) == {s['number'] for s in manifest['sources']}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output/'videos').mkdir(exist_ok=False)
    started = time.perf_counter()
    checks, cards = [], []
    labels = {'music_end': '完整音乐结束', 'measured_hard_joint_limit': '实际关节超出硬限位',
              'sm_physical_termination': 'SM原终止条件触发', 'nonfinite_action': '动作非有限',
              'nonfinite_measured_state': '实际状态非有限',
              'solver_capacity_or_contact_failure': '求解器容量或接触异常'}
    for source in manifest['sources']:
        number, directory = source['number'], source['directory']
        result = results[number]
        origin = args.inputs/'videos'/directory
        measured = args.execution/directory
        assert digest(origin/'generated.pt') == result['generated_reference_sha256']
        assert digest(measured/'physics.pt') == result['physics_sha256']
        assert result['reset_count'] == 1
        target = output/'videos'/directory
        target.mkdir()
        for filename in ('original.pt', 'generated.pt', 'generated.npz', 'audio.wav'):
            shutil.copy2(origin/filename, target/filename)
        for filename in ('physics.pt', 'result.json'):
            shutil.copy2(measured/filename, target/filename)
        command = [sys.executable, str(ROOT/'tools/eval/render_bumi_motion.py'),
            '--motion', str(target/'original.pt'), '--compare-motion', str(target/'generated.pt'), str(target/'physics.pt'),
            '--mjcf', str(args.mjcf), '--audio', str(target/'audio.wav'), '--font',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', '--output', str(target/'comparison.mp4'),
            '--width', '640', '--height', '540', '--title', f"{number:02d} {source['split']} / {source['dataset']}"]
        subprocess.run(command, check=True, env=dict(os.environ, MUJOCO_GL='egl', PYTHONDONTWRITEBYTECODE='1'))
        check = dict(number=number, **verify_video(target/'comparison.mp4', source['frames']))
        checks.append(check)
        reason = labels.get(result['end_reason'], result['end_reason'])
        text = f"真实执行 {result['physical_seconds']:.2f} / {source['frames']/30:.2f}秒；{reason}"
        cards.append(f'<article><h2>{number:02d} · {html.escape(source["split"])} · {html.escape(source["dataset"])}</h2>'
            f'<p>{html.escape(source["source"])}</p><p>{html.escape(text)}</p>'
            f'<video controls preload="none" poster="videos/{directory}/comparison_frame_0000.jpg" '
            f'src="videos/{directory}/comparison.mp4"></video><p><a href="videos/{directory}/comparison.mp4" download>下载视频</a>'
            f' · <a href="videos/{directory}/physics.pt" download>真实物理轨迹</a></p></article>')
        print(json.dumps(dict(number=number, video_verified=True, physical_seconds=result['physical_seconds'],
                              end_reason=result['end_reason'], wall_seconds=time.perf_counter()-started)), flush=True)
        write_json(output/'delivery_checks.json', checks)
    summary = {split: dict(songs=sum(s['split']==split for s in execution['sources']),
        completed=sum(s['split']==split and s['complete_song'] for s in execution['sources']),
        physical_seconds=sum(s['physical_seconds'] for s in execution['sources'] if s['split']==split)) for split in ('train','test')}
    write_json(output/'summary.json', dict(step=manifest['step'], split=summary, sources=execution['sources'],
        frozen_unchanged=True, world_model_forward_calls=0, renderer_wall_seconds=time.perf_counter()-started))
    title = f"GENMO {manifest['step']:,}步：10首训练集＋5首测试集三栏对照"
    explanation = ('左：原始参考；中：完整20步DDIM纯噪声生成；右：冻结SM在真实Newton中的执行。'
        '采用已固定的15首与生成轨迹，每首仅初始化一次。真实执行终止后右栏留空，原参考和生成栏继续完整播放。'
        '这是离线固定参考的真实跟踪，不是异步在线生成时延测试，也不是硬件实测。')
    body = f'<h1>{title}</h1><p>{explanation}</p><p>训练集完成{summary["train"]["completed"]}/10；测试集完成{summary["test"]["completed"]}/5。</p>'
    body += '<p><a href="15首三栏视频与真实轨迹.zip" download>下载全部视频和真实轨迹</a> · <a href="summary.json">执行汇总</a></p>'
    (output/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{title}</title><style>body{{background:#101a2a;color:#e8eef8;font-family:sans-serif;margin:24px auto;max-width:1500px;padding:0 20px;line-height:1.7}}'
        'article{border:1px solid #40536b;padding:16px;margin:24px 0;border-radius:12px}video{width:100%;background:#070c13}a{color:#8dc6ff}p{overflow-wrap:anywhere}</style>'
        +body+''.join(cards)+'</html>', encoding='utf-8')
    (output/'说明.txt').write_text(title+'\n'+explanation+'\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with zipfile.ZipFile(output/'15首三栏视频与真实轨迹.zip', 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in sorted((output/'videos').rglob('*')):
            if path.is_file(): archive.write(path, path.relative_to(output))
        for name in ('index.html','说明.txt','summary.json','delivery_checks.json'):
            archive.write(output/name,name)
    print(json.dumps(dict(delivered=len(checks), split=summary, wall_seconds=time.perf_counter()-started)),flush=True)


if __name__ == '__main__':
    main()
