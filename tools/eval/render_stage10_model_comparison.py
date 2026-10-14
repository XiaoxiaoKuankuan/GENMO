"""把三模型同任务的Isaac真实执行轨迹并排渲染成带原音乐的MuJoCo视频。

只读加载50Hz实际qpos，逐帧mj_forward运动学回放，不运行MuJoCo动力学，不插值、
不修改姿态、不补最后一帧。提前失败/结束的面板明确变成结束提示，其余模型继续
按真实音乐时刻播放。三个视角保持相同角度和距离、仅跟随各自根平移，同时显示
根位移、真实活动度、配对目标、近软边界关节数和累计回报。视频用同一条原音频。

输入严格核验轨迹SHA、任务/seed/group一致性、原生21关节顺序、wxyz四元数及
源MJCF SHA；输出视频、缩略图和逐项身份JSON。渲染不接触训练代码、资产和权重，
不能把此视频称作重新在MuJoCo执行冻结GMT的动力学实验。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault('MUJOCO_GL','egl')
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def sha(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):d.update(block)
    return d.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectories',type=Path,nargs=3,required=True)
    parser.add_argument('--labels',nargs=3,default=['Stage1 s595000','第147轮','第246轮'])
    parser.add_argument('--mjcf',type=Path,required=True)
    parser.add_argument('--kinematics',type=Path,required=True)
    parser.add_argument('--audio',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--panel-width',type=int,default=640)
    parser.add_argument('--panel-height',type=int,default=480)
    parser.add_argument('--font',type=Path,default=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'))
    args=parser.parse_args()
    output=args.output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists() or output.with_suffix('.json').exists():raise FileExistsError(output)
    metadata=[json.loads(p.read_text()) for p in args.trajectories]
    if len({m['task_id'] for m in metadata})!=1 or any(m['task']!=metadata[0]['task'] for m in metadata):
        raise ValueError('Three trajectories must use the identical task, music and seed')
    kin=json.loads(args.kinematics.read_text())
    if sha(args.mjcf)!=kin['source_mjcf_sha256']:raise ValueError('Rendering MJCF differs from verified kinematics')
    arrays=[]
    for p,m in zip(args.trajectories,metadata):
        file=p.parent/m['trajectory_file']
        if sha(file)!=m['sha256']:raise ValueError('Actual trajectory SHA differs')
        with np.load(file) as z:a={k:z[k] for k in z.files}
        if m['fps']!=50 or m['quaternion_convention']!='wxyz' or m['qpos_order']!='mujoco_native':
            raise ValueError('Actual trajectory convention differs')
        if a['qpos'].shape!=(m['actual_control_steps'],28) or not np.isfinite(a['qpos']).all():raise ValueError('Invalid actual qpos')
        if np.max(np.abs(np.linalg.norm(a['qpos'][:,3:7],axis=1)-1))>1e-5:raise ValueError('Non-unit actual root quaternion')
        if not np.all(np.diff(a['tick'])==12):raise ValueError('Missing real controls')
        arrays.append(a)
    task=metadata[0]['task'];frames=max(len(a['tick']) for a in arrays);fps=50
    audio_sha=sha(args.audio)
    if audio_sha!=task['sample']['row']['source_audio_sha256']:
        raise ValueError('Audio file differs from task manifest')
    w,h=args.panel_width,args.panel_height
    if w%2 or h%2:raise ValueError('Even render dimensions required')
    header,footer=90,96;size=(w*3,h+header+footer)
    model=mujoco.MjModel.from_xml_path(str(args.mjcf))
    names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(model.njnt) if model.jnt_qposadr[i]>=7]
    if model.nq!=28 or any(m['joint_names']!=names for m in metadata):raise ValueError('MJCF native joint order differs')
    model.vis.global_.offwidth=max(w,model.vis.global_.offwidth);model.vis.global_.offheight=max(h,model.vis.global_.offheight)
    renderer=mujoco.Renderer(model,height=h,width=w);data=mujoco.MjData(model)
    camera=mujoco.MjvCamera();mujoco.mjv_defaultFreeCamera(model,camera)
    camera.distance=2.45;camera.azimuth=135;camera.elevation=-13
    title_font=ImageFont.truetype(str(args.font),24);text_font=ImageFont.truetype(str(args.font),18)
    small_font=ImageFont.truetype(str(args.font),16)
    temporary=output.with_name(output.stem+'.partial.mp4')
    if temporary.exists():raise FileExistsError(temporary)
    # qpos[0]是首个真实控制区间结束（0.02s），音频对齐同一时刻，不把重置预热混入歌曲。
    command=['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{size[0]}x{size[1]}',
        '-framerate','50','-i','pipe:0','-ss','0.02','-i',str(args.audio),'-map','0:v:0','-map','1:a:0',
        '-t',str(frames/fps),'-c:v','libx264','-preset','veryfast','-crf','21','-threads','2','-pix_fmt','yuv420p',
        '-c:a','aac','-b:a','160k','-af','apad','-movflags','+faststart',str(temporary)]
    encoder=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    colors=['#78b9ff','#68dfaa','#ffb36a'];thumbnail=min(frames-1,max(0,frames//3));started=time.perf_counter()
    cumulative=[np.cumsum(a['reward']) for a in arrays]
    try:
        for index in range(frames):
            canvas=Image.new('RGB',size,'#101a2a');draw=ImageDraw.Draw(canvas)
            draw.text((16,4),f"{task['split']} / {task['dataset']} / {task['sample_id']}    seed={task['seed']}    音乐 t={(index+1)/50:.2f}s",font=text_font,fill='white')
            draw.text((16,29),'Isaac真实执行轨迹 · MuJoCo运动学回放 · 相同视角，跟随根平移 · 非MuJoCo动力学重跑',font=small_font,fill='#b7c5d9')
            for column,(a,m,label,color) in enumerate(zip(arrays,metadata,args.labels,colors)):
                left=column*w;draw.text((left+16,56),label,font=title_font,fill=color)
                if index<len(a['tick']):
                    data.qpos[:]=a['qpos'][index];mujoco.mj_forward(model,data)
                    camera.lookat[:]=[data.qpos[0],data.qpos[1],.62]
                    renderer.update_scene(data,camera=camera)
                    canvas.paste(Image.fromarray(renderer.render()),(left,header))
                    safe=a['safe_limits'][0];hard=a['hard_limits'][0];q=a['actual_joint_gmt'][index]
                    near=int(np.sum((q<safe[:,0])|(q>safe[:,1])|
                        ((safe[:,0]==hard[:,0])&(q<=hard[:,0]))|((safe[:,1]==hard[:,1])&(q>=hard[:,1]))))
                    outside=int(np.sum((q<hard[:,0]-1e-6)|(q>hard[:,1]+1e-6)))
                    displacement=float(np.linalg.norm(a['qpos'][index,:2]-a['qpos'][0,:2]))
                    draw.text((left+12,header+h+5),f"活动 {a['activity'][index]:.3f} / 目标 {a['target_activity'][index]:.3f} rad/s",font=text_font,fill='white')
                    draw.text((left+12,header+h+32),f"近边界 {near}/21  硬越界 {outside}/21  根位移 {displacement:.2f}m",font=text_font,fill='#ffad96' if outside else '#c5d0df')
                    draw.text((left+12,header+h+59),f"累计回报 {cumulative[column][index]:.2f}  活动门控 {a['activity_gate'][index]:.2f}",font=text_font,fill='#c5d0df')
                else:
                    reason='音乐结束' if m['end_reason']=='music_end' else '执行已终止：'+str(m['end_reason'])
                    draw.text((left+25,header+h//2-35),reason,font=title_font,fill=color)
                    draw.text((left+25,header+h//2+10),f"实际执行 {len(a['tick'])/50:.2f}s；结束后不补帧",font=text_font,fill='#c5d0df')
                draw.line((left,50,left,size[1]),fill='#536278',width=2)
            if index==thumbnail:canvas.save(output.with_suffix('.jpg'),quality=92)
            encoder.stdin.write(np.asarray(canvas,dtype=np.uint8).tobytes())
        encoder.stdin.close();error=encoder.stderr.read().decode(errors='replace');result=encoder.wait()
        if result:raise RuntimeError(f'ffmpeg failed {result}: {error}')
        os.link(temporary,output);temporary.unlink()
    except BaseException:
        if encoder.stdin and not encoder.stdin.closed:encoder.stdin.close()
        if encoder.poll() is None:encoder.terminate();encoder.wait()
        temporary.unlink(missing_ok=True)
        raise
    finally:renderer.close()
    manifest=dict(schema='genmo.stage10.three_model_mujoco_replay.v1',task=task,labels=args.labels,
        trajectory_metadata=[dict(path=str(p),sha256=sha(p),npz_sha256=m['sha256']) for p,m in zip(args.trajectories,metadata)],
        video_sha256=sha(output),fps=50,frames=frames,duration_seconds=frames/50,
        panel_duration_seconds=[len(a['tick'])/50 for a in arrays],audio_sha256=audio_sha,audio_start_seconds=.02,
        mjcf_sha256=sha(args.mjcf),mujoco_version=mujoco.__version__,
        scope='kinematic_replay_of_actual_Isaac_qpos_not_new_MuJoCo_dynamics',
        camera=dict(distance=2.45,azimuth=135,elevation=-13,follow_root_translation=True,follow_rotation=False),
        ended_panels='explicit_status_without_pose_padding',wall_seconds=time.perf_counter()-started)
    with output.with_suffix('.json').open('x') as f:json.dump(manifest,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(output=str(output),frames=frames,seconds=frames/50,wall_seconds=manifest['wall_seconds']),ensure_ascii=False))


if __name__=='__main__':main()
