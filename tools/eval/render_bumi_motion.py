#!/usr/bin/env python3
"""离线渲染 BUMI qpos28 轨迹（仅运动学，不执行动力学仿真）。

--compare-motion扩展为原参考/生成参考/真实Newton记录的三栏同步回放；按30/50Hz原时间
读取，终止之后留空并标记，不复制最后姿态。原音频、中文标签、关节名称映射和成品元数据
一并验证；仅在内存给渲染模型加地板/光照，不修改源机器人或真实物理参数。

工具严格校验轨迹的机器人标识、四元数约定、MuJoCo 原生关节顺序和 30 Hz 帧率，
再逐帧执行 ``mj_forward`` 并写入 H.264 视频。视频帧采用流式编码，避免完整音乐对应的
数千帧 RGB 图像同时驻留内存；因此既适合短片验证，也适合数分钟完整音频的批量评测。
``--max-frames`` 可在加载并完成轨迹契约校验后只渲染前缀帧，供固定时长网页验收避免先渲染
整段长动作再由 ffmpeg 截断；该选项不循环、不补帧，也不会改变源动作产物。
``--follow-root`` 为长音乐对比提供可选的根平移跟随相机：保持默认相机的距离、
方位角和俯角，仅逐帧把观察中心移到根位置，避免机器人随世界系位移走出画面。
它不跟随根旋转，不修改 qpos、地面或动作指标；不传此选项时保留原来的固定相机。
离屏缓冲在内存中按请求分辨率扩容，不改写带SHA身份的源XML及任何动力学参数。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_qpos(path: Path, key: str) -> tuple[np.ndarray, int, dict[str, Any]]:
    payload = torch_load(path)
    if not isinstance(payload, dict):
        raise ValueError(f"BUMI motion artifact must be a dictionary: {path}")
    expected = {
        "robot_name": "bumi",
        "quaternion_convention": "wxyz",
        "qpos_order": "mujoco_native",
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"{path}: {field} must be {value!r}, got {payload.get(field)!r}")
    qpos = torch.as_tensor(payload.get(key)).detach().cpu().double()
    if qpos.ndim != 2 or qpos.shape[1] != 28 or qpos.shape[0] <= 0:
        raise ValueError(f"{path}: {key} must have shape [T,28], got {qpos.shape}")
    if not bool(torch.isfinite(qpos).all()):
        raise ValueError(f"{path}: {key} contains NaN or Inf")
    norm = torch.linalg.vector_norm(qpos[:, 3:7], dim=-1)
    if float((norm - 1.0).abs().max()) > 1.0e-3:
        raise ValueError(f"{path}: root quaternion is not normalized wxyz")
    return qpos.numpy(), int(payload.get("fps", 30)), payload


def mujoco_joint_order(model: mujoco.MjModel) -> list[str]:
    ids = []
    for joint_id in range(model.njnt):
        address = int(model.jnt_qposadr[joint_id])
        if address >= 7:
            ids.append(joint_id)
    ids.sort(key=lambda value: int(model.jnt_qposadr[value]))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, value) for value in ids]
    if any(name is None for name in names):
        raise ValueError("Every BUMI actuated MJCF joint must have a name")
    return [str(name) for name in names]


def render_comparison(args):
    """同音乐三栏：原参考、生成参考、真实物理记录；终止后留空，禁止重复末帧。"""
    import json
    import math
    import os
    import xml.etree.ElementTree as ET
    from PIL import Image, ImageDraw, ImageFont
    paths = [args.motion, *args.compare_motion]
    loaded = [load_qpos(path, args.qpos_key) for path in paths]
    if [item[1] for item in loaded] != [30,30,50]:
        raise ValueError("三栏合同需要30Hz原参考、30Hz生成参考、50Hz实测轨迹")
    if loaded[2][2].get('evidence') != 'measured_newton_state':
        raise ValueError("动力学栏必须是Newton真实记录")
    source = args.mjcf.resolve()
    xml = ET.parse(source).getroot()
    compiler = xml.find('compiler')
    meshdir = source.parent / (compiler.get('meshdir','.') if compiler is not None else '.')
    asset = xml.find('asset')
    if asset is None:
        asset = ET.SubElement(xml,'asset')
    for mesh in asset.findall('mesh'):
        if mesh.get('file'):
            mesh.set('file',str((meshdir/mesh.get('file')).resolve()))
    if compiler is not None:
        compiler.attrib.pop('meshdir',None)
    world = xml.find('worldbody')
    if not any(geom.get('type')=='plane' for geom in world.findall('geom')):
        ET.SubElement(asset,'texture',name='comparison_grid',type='2d',builtin='checker',rgb1='.85 .87 .9',rgb2='.65 .69 .74',width='512',height='512')
        ET.SubElement(asset,'material',name='comparison_ground',texture='comparison_grid',texrepeat='4 4')
        ET.SubElement(world,'geom',name='comparison_floor',type='plane',size='0 0 .01',material='comparison_ground',contype='0',conaffinity='0')
    ET.SubElement(world,'light',pos='0 0 6',dir='0 0 -1',directional='true',diffuse='.8 .8 .8')
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'))
    order = mujoco_joint_order(model)
    arrays=[]
    for qpos,fps,meta in loaded:
        names=meta.get('joint_names',[])
        if len(names)!=21 or len(set(names))!=21 or set(names)!=set(order):
            raise ValueError("三栏关节名称集合必须与机器人完全匹配")
        arrays.append(np.concatenate((qpos[:,:7],qpos[:,[7+names.index(name) for name in order]]),axis=-1))
    if len(arrays[0])!=len(arrays[1]):
        raise ValueError("原参考与生成参考必须覆盖相同音乐时长")
    n=len(arrays[0]); fps=30
    w,h=args.width,args.height
    top,bottom=94,92
    model.vis.global_.offwidth=max(model.vis.global_.offwidth,w)
    model.vis.global_.offheight=max(model.vis.global_.offheight,h)
    model.vis.headlight.ambient[:]=.4
    renderer=mujoco.Renderer(model,height=h,width=w)
    data=mujoco.MjData(model)
    camera=mujoco.MjvCamera();mujoco.mjv_defaultFreeCamera(model,camera)
    camera.distance=2.6;camera.azimuth=135;camera.elevation=-15
    font=ImageFont.truetype(str(args.font),23)
    small=ImageFont.truetype(str(args.font),18)
    output=args.output.resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError(output)
    staged=output.with_name(output.stem+'.partial.mp4')
    if staged.exists():raise FileExistsError(staged)
    size=(3*w,top+h+bottom)
    command=['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{size[0]}x{size[1]}',
        '-framerate','30','-i','pipe:0','-i',str(args.audio),'-map','0:v:0','-map','1:a:0','-af','apad',
        '-t',str(n/30),'-c:v','libx264','-preset','fast','-crf','20','-threads','2','-pix_fmt','yuv420p',
        '-c:a','aac','-b:a','160k','-movflags','+faststart',str(staged)]
    encoder=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    labels=['原始数据参考','GENMO生成参考 · 10000步','SM + Newton真实执行']
    colors=['#8dc6ff','#78e0b4','#ffbd83']
    measured=loaded[2][2]
    end=float(measured['physical_end_seconds'])
    reason=measured['end_reason']
    ended_text={'measured_hard_joint_limit':'实际关节超硬限位','sm_physical_termination':'SM原终止条件触发',
        'reference_support_exhausted':'参考支持耗尽','music_end':'音乐结束'}.get(reason,reason)
    try:
        for i in range(n):
            t=i/30
            canvas=Image.new('RGB',size,'#101a2a');draw=ImageDraw.Draw(canvas)
            draw.text((16,7),f'{args.title}  |  音乐 {t:.2f} / {n/30:.2f}s  |  1倍速',font=font,fill='white')
            draw.text((16,39),'左/中：运动学动画；右：真实Newton状态回放。纯噪声DDIM20，非去噪重建；不测试在线时延。',font=small,fill='#c4cfdf')
            for column,(qpos,loaded_item) in enumerate(zip(arrays,loaded)):
                x=column*w
                draw.text((x+14,67),labels[column],font=font,fill=colors[column])
                if column==2 and t>end+1e-9:
                    draw.text((x+30,top+h//2-40),'执行已终止',font=font,fill=colors[column])
                    draw.text((x+30,top+h//2),f'{end:.2f}s：{ended_text}',font=small,fill='white')
                    draw.text((x+30,top+h//2+32),'后续没有物理记录，不补帧',font=small,fill='#c4cfdf')
                else:
                    index=min(len(qpos)-1,int(math.floor(t*loaded_item[1]+1e-8)))
                    data.qpos[:]=qpos[index]
                    mujoco.mj_forward(model,data)
                    camera.lookat[:]=[qpos[index,0],qpos[index,1],.55]
                    renderer.update_scene(data,camera=camera)
                    canvas.paste(Image.fromarray(renderer.render()),(x,top))
                if column==0:
                    lines=['原音乐配对动作，完整序列','仅作内容对照，不是生成目标轨迹']
                elif column==1:
                    lines=['前0.4秒保持单个初态，随后自主生成','自身历史代理；无真实缓存，Future无效']
                else:
                    lines=[f'实际执行 {end:.2f}s / 音乐 {n/30:.2f}s',f'结束原因：{ended_text}']
                for j,line in enumerate(lines):
                    draw.text((x+14,top+h+10+j*29),line,font=small,fill='#c4cfdf')
                draw.line((x,65,x,size[1]),fill='#40536b',width=2)
            if i in {0,min(n-1,30),n//2}:
                canvas.save(output.with_name(output.stem+f'_frame_{i:04d}.jpg'),quality=92)
            encoder.stdin.write(np.asarray(canvas,dtype=np.uint8).tobytes())
        encoder.stdin.close()
        error=encoder.stderr.read().decode(errors='replace')
        if encoder.wait():raise RuntimeError(error)
        os.link(staged,output);staged.unlink()
    except BaseException:
        if encoder.stdin and not encoder.stdin.closed:encoder.stdin.close()
        if encoder.poll() is None:encoder.terminate();encoder.wait()
        staged.unlink(missing_ok=True)
        raise
    finally:
        renderer.close()
    output.with_suffix('.render.json').write_text(json.dumps(dict(frames=n,fps=30,seconds=n/30,
        physical_seconds=end,end_reason=reason,inputs=[str(p.resolve()) for p in paths],
        rendering='mujoco_kinematic_replay_of_separate_reference_and_real_newton_states',
        camera='same_distance_angle_each_panel_follows_root_translation_no_pose_realignment'),ensure_ascii=False,indent=2))
    print(f'Rendered comparison {output}',flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", required=True, type=Path)
    parser.add_argument("--mjcf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--qpos-key", default="qpos")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera")
    parser.add_argument("--compare-motion",type=Path,nargs=2,help="生成参考和真实Newton记录，与--motion原参考组成三栏")
    parser.add_argument("--audio",type=Path)
    parser.add_argument("--font",type=Path)
    parser.add_argument("--title",default="联合GENMO三栏对比")
    parser.add_argument(
        "--follow-root", action="store_true", help="相机中心逐帧跟随根位置；不改变动作"
    )
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    if args.compare_motion:
        if args.audio is None or args.font is None:
            parser.error("三栏模式需要原音乐--audio与中文--font")
        render_comparison(args)
        return
    if args.follow_root and args.camera is not None:
        parser.error("--follow-root 与命名 --camera 不能同时使用")
    qpos, fps, payload = load_qpos(args.motion.expanduser().resolve(), args.qpos_key)
    if args.max_frames is not None:
        if args.max_frames <= 0:
            raise ValueError("--max-frames 必须为正整数")
        qpos = qpos[: args.max_frames]
    if fps != 30:
        raise ValueError(f"BUMI render input must be 30 FPS, got {fps}")
    model = mujoco.MjModel.from_xml_path(str(args.mjcf.expanduser().resolve()))
    if model.nq != 28:
        raise ValueError(f"BUMI MJCF nq must be 28, got {model.nq}")
    artifact_joint_names = tuple(map(str, payload.get("joint_names", ())))
    mjcf_joint_names = tuple(mujoco_joint_order(model))
    if artifact_joint_names != mjcf_joint_names:
        raise ValueError(
            "Motion joint_names do not exactly match MuJoCo-native MJCF qpos order; "
            "rendering refuses to guess a reorder"
        )
    data = mujoco.MjData(model)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera: str | int | mujoco.MjvCamera = -1 if args.camera is None else args.camera
    if args.follow_root:
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, camera)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("BUMI 视频渲染需要 ffmpeg")
    encoder = subprocess.Popen(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{args.width}x{args.height}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    rendered_frames = 0
    try:
        assert encoder.stdin is not None
        for frame_qpos in qpos:
            data.qpos[:] = frame_qpos
            mujoco.mj_forward(model, data)
            if args.follow_root:
                camera.lookat[:] = frame_qpos[:3]
            renderer.update_scene(data, camera=camera)
            frame = np.ascontiguousarray(renderer.render(), dtype=np.uint8)
            if frame.shape != (args.height, args.width, 3):
                raise RuntimeError(f"MuJoCo 返回了异常画面形状：{frame.shape}")
            encoder.stdin.write(frame.tobytes())
            rendered_frames += 1
        encoder.stdin.close()
        assert encoder.stderr is not None
        error_text = encoder.stderr.read().decode("utf-8", errors="replace").strip()
        returncode = encoder.wait()
        if returncode != 0:
            raise RuntimeError(f"ffmpeg 编码失败（{returncode}）：{error_text}")
    except BaseException:
        if encoder.stdin is not None and not encoder.stdin.closed:
            encoder.stdin.close()
        if encoder.poll() is None:
            encoder.terminate()
            encoder.wait()
        output.unlink(missing_ok=True)
        raise
    finally:
        renderer.close()
    print(f"Rendered {rendered_frames} frames to {output}")


if __name__ == "__main__":
    main()
