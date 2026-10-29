"""将基线与训练后Adapter的真实执行及参考影子左右渲染，输出视频与对比图。

输入是同一次整歌配对评价导出的两个JSON/NPZ：左侧零Adapter，右侧已训练Adapter。
每侧实色机器人使用真实Newton物理执行qpos，蓝色半透明机器人使用同一控制步下发的
参考第一帧；原始GENMO名义目标仍用于误差和世界坐标轨迹图。两侧保持同一音乐时间、
相同相机以及共同名义目标的镜头中心，不单独跟随实际机器人而掩盖位置漂移。
只通过mj_forward重建渲染几何，不推进MuJoCo动力学，不修改/平滑/插值轨迹。真实执行
提前结束的面板显示终止信息，不补机器人姿态；视频保留原音乐，图片选择配对根误差
差异的一秒滑动平均峰值，并写明选择时刻。文件SHA、关节名称、初态、物理参数、音频
和50Hz时钟在输入边界检查，所有产物写入新路径，禁止覆盖已有视频或图片。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectories', type=Path, nargs=2, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mjcf', type=Path)
    parser.add_argument('--audio', type=Path)
    parser.add_argument('--labels', nargs=2, default=['基线：GENMO + 冻结SuperMimic', '最好模型：Adapter GPU2 / u700'])
    parser.add_argument('--panel-width', type=int, default=800)
    parser.add_argument('--panel-height', type=int, default=580)
    parser.add_argument('--max-frames', type=int, help='仅用于有限渲染验收，正式整歌不设置')
    parser.add_argument('--font', type=Path, default=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'))
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if any(output.with_suffix(ext).exists() for ext in ('.mp4', '.png', '.json')):
        raise FileExistsError(output)
    metadata = [json.loads(path.read_text()) for path in args.trajectories]
    first, second = metadata
    for key in ('task', 'initial_state_sha256', 'physical_xi_sha256', 'joint_names', 'mjcf_sha256', 'audio_start_seconds'):
        if first[key] != second[key]:
            raise ValueError(f'配对渲染输入不一致：{key}')
    if not first['zero_adapter'] or second['zero_adapter']:
        raise ValueError('左右输入必须依次为零Adapter基线、训练后模型')
    mjcf = args.mjcf or Path(first['mjcf'])
    audio = args.audio or Path(first['task']['audio_path'])
    if sha(mjcf) != first['mjcf_sha256'] or sha(audio) != first['task']['audio_sha256']:
        raise ValueError('MJCF或音频不匹配实际评价资产')
    arrays = []
    for path, meta in zip(args.trajectories, metadata):
        npz = path.parent/meta['trajectory_file']
        if sha(npz) != meta['sha256']:
            raise ValueError('轨迹SHA不符')
        with np.load(npz) as archive:
            row = {key: archive[key] for key in archive.files}
        n = meta['actual_control_steps']
        if meta['fps'] != 50 or meta['quaternion_convention'] != 'wxyz' or meta['qpos_order'] != 'mujoco_native':
            raise ValueError('轨迹时钟或姿态约定不符')
        for key in ('qpos', 'nominal_qpos', 'corrected_qpos'):
            if row[key].shape != (n, 28) or not np.isfinite(row[key]).all():
                raise ValueError(f'无效姿态：{key}')
            if n and np.max(np.abs(np.linalg.norm(row[key][:, 3:7], axis=1)-1)) > 1e-5:
                raise ValueError('根四元数不是单位四元数')
        if not np.array_equal(row['tick'], np.arange(1, n+1)*12):
            raise ValueError('真实50Hz控制时钟缺帧')
        arrays.append(row)
    common = min(len(row['tick']) for row in arrays)
    if common:
        np.testing.assert_allclose(arrays[0]['nominal_qpos'][:common], arrays[1]['nominal_qpos'][:common], atol=1e-6, rtol=1e-6)
    frames = max(len(row['tick']) for row in arrays)
    if args.max_frames is not None:
        frames = min(frames, args.max_frames)
    if not frames:
        raise ValueError('没有可渲染的真实执行帧')
    roots = np.concatenate([row[key][:, :2] for row in arrays for key in ('qpos', 'nominal_qpos')])
    map_mid = (roots.min(0)+roots.max(0))/2
    map_span = max(float(np.ptp(roots, axis=0).max()), 1.)*1.2
    peak_count = min(common, frames)
    if peak_count:
        delta = np.abs(np.sqrt(arrays[0]['raw_error_root_pos'][:peak_count])-np.sqrt(arrays[1]['raw_error_root_pos'][:peak_count]))
        window = min(50, peak_count)
        snapshot = int(np.argmax(np.convolve(delta, np.ones(window)/window, mode='valid'))+window//2)
    else:
        snapshot = frames//2
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
             for i in range(model.njnt) if model.jnt_qposadr[i] >= 7]
    if model.nq != 28 or names != first['joint_names']:
        raise ValueError('渲染关节顺序与评价轨迹不符')
    width, height = args.panel_width, args.panel_height
    header, footer = 146, 116
    size = (2*width, header+height+footer)
    model.vis.global_.offwidth = max(width, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(height, model.vis.global_.offheight)
    renderer = mujoco.Renderer(model, height=height, width=width, max_geom=10000)
    actual, ghost = mujoco.MjData(model), mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.distance, camera.azimuth, camera.elevation = 2.8, 135, -17
    option, perturb = mujoco.MjvOption(), mujoco.MjvPerturb()
    title_font = ImageFont.truetype(str(args.font), 25)
    font = ImageFont.truetype(str(args.font), 20)
    small = ImageFont.truetype(str(args.font), 17)
    colors = [(0.96, .59, .23, 1.), (.24, .82, .63, 1.)]
    ink = ['#ffb266', '#61dfb1']
    temp = output.with_name(output.stem+'.partial.mp4')
    if temp.exists():
        raise FileExistsError(temp)
    command = ['ffmpeg', '-nostdin', '-v', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb24',
        '-video_size', f'{size[0]}x{size[1]}', '-framerate', '50', '-i', 'pipe:0',
        '-ss', str(first['audio_start_seconds']), '-i', str(audio), '-map', '0:v:0', '-map', '1:a:0',
        '-t', str(frames/50), '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-threads', '2',
        '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '160k', '-af', 'apad', '-movflags', '+faststart', str(temp)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    started = time.perf_counter()
    try:
        for index in range(frames):
            canvas = Image.new('RGB', size, '#101a28')
            draw = ImageDraw.Draw(canvas)
            task = first['task']
            draw.text((20, 9), f"训练集 / {task['source']} / {task['id']}", font=font, fill='white')
            draw.text((20, 41), '实色：真实仿真执行   蓝色影子：该方法下发参考   两侧同一音乐时刻、同一镜头', font=font, fill='#bdcadd')
            draw.text((20, 71), f'Newton真实轨迹回放 · 50 Hz · 音乐 {(index+1)/50:.2f} s · 原始名义目标用于计算误差', font=small, fill='#97acc3')
            available = next(row for row in arrays if index < len(row['tick']))
            center = available['nominal_qpos'][index, :3]
            camera.lookat[:] = [center[0], center[1], .67]
            for column, (row, meta) in enumerate(zip(arrays, metadata)):
                left = column*width
                draw.text((left+20, 107), args.labels[column], font=title_font, fill=ink[column])
                if index < len(row['tick']):
                    actual.qpos[:] = row['qpos'][index]
                    ghost.qpos[:] = row['corrected_qpos'][index]
                    mujoco.mj_forward(model, actual)
                    mujoco.mj_forward(model, ghost)
                    renderer.update_scene(actual, camera=camera, scene_option=option)
                    scene = renderer.scene
                    for geom in scene.geoms[:scene.ngeom]:
                        if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and model.geom_bodyid[geom.objid] > 0:
                            geom.rgba[:] = colors[column]
                    first_ghost = scene.ngeom
                    mujoco.mjv_addGeoms(model, ghost, option, perturb, mujoco.mjtCatBit.mjCAT_DYNAMIC, scene)
                    for geom in scene.geoms[first_ghost:scene.ngeom]:
                        geom.rgba[:] = [.12, .66, 1., .32]
                    ground = scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(ground, mujoco.mjtGeom.mjGEOM_PLANE, np.array([25., 25., .05]),
                        np.zeros(3), np.eye(3).flatten(), np.array([.20, .25, .29, 1.]))
                    scene.ngeom += 1
                    cx, cy = np.floor(center[:2]).astype(int)
                    for offset in range(-4, 6):
                        for start, end in [([cx+offset, cy-5, .001], [cx+offset, cy+5, .001]),
                                           ([cx-5, cy+offset, .001], [cx+5, cy+offset, .001])]:
                            geom = scene.geoms[scene.ngeom]
                            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, np.ones(3), np.zeros(3),
                                np.eye(3).flatten(), np.array([.38, .44, .49, 1.]))
                            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 1., np.array(start), np.array(end))
                            scene.ngeom += 1
                    canvas.paste(Image.fromarray(renderer.render()), (left, header))
                    root_error = float(np.sqrt(row['raw_error_root_pos'][index]))
                    q_error = float(np.sqrt(row['raw_joint_q_mse'][index]))
                    draw.text((left+18, header+height+7), f'根位置误差 {root_error*100:.2f} cm    关节RMSE {q_error:.4f} rad', font=font, fill='white')
                    draw.text((left+18, header+height+39), f"本帧Tracking {row['raw_track'][index]:.5f}    已执行 {(index+1)/50:.2f} s", font=font, fill='#bdcadd')
                    full_root = np.sqrt(np.mean(row['raw_error_root_pos']))
                    draw.text((left+18, header+height+71), f'整段根RMSE {full_root*100:.2f} cm    {meta["end_reason"]}', font=small, fill=ink[column])
                    mx, my = left+width-190, header+height-175
                    draw.rectangle((mx, my, mx+176, my+161), fill='#152334', outline='#6d839b')
                    draw.text((mx+8, my+5), '世界位置 / 同一比例', font=small, fill='white')
                    for key, color in [('nominal_qpos', '#4ba9ef'), ('qpos', ink[column])]:
                        xy = (row[key][:index+1, :2]-map_mid)/map_span
                        points = [(mx+88+float(p[0])*144, my+94-float(p[1])*112) for p in xy]
                        if len(points) > 1:
                            draw.line(points, fill=color, width=2)
                        x, y = points[-1]
                        draw.ellipse((x-3, y-3, x+3, y+3), fill=color)
                else:
                    draw.text((left+40, header+height//2), f'执行已结束：{meta["end_reason"]}', font=title_font, fill=ink[column])
                    draw.text((left+40, header+height//2+42), f'实际执行 {len(row["tick"])/50:.2f} 秒，结束后不补姿态', font=font, fill='white')
                draw.line((left, 101, left, size[1]), fill='#58718b', width=2)
            if index == snapshot:
                canvas.save(output.with_suffix('.png'))
            encoder.stdin.write(np.asarray(canvas, dtype=np.uint8).tobytes())
            if (index+1) % 500 == 0:
                print(json.dumps(dict(frames=index+1, total=frames, wall_seconds=time.perf_counter()-started)), flush=True)
        encoder.stdin.close()
        error = encoder.stderr.read().decode(errors='replace')
        if encoder.wait():
            raise RuntimeError(error)
        os.link(temp, output)
        temp.unlink()
    except BaseException:
        if not encoder.stdin.closed:
            encoder.stdin.close()
        if encoder.poll() is None:
            encoder.terminate()
            encoder.wait()
        temp.unlink(missing_ok=True)
        raise
    finally:
        renderer.close()
    manifest = dict(task=first['task'], labels=args.labels, frames=frames, fps=50, duration_seconds=frames/50,
        panel_actual_seconds=[len(row['tick'])/50 for row in arrays], snapshot_seconds=(snapshot+1)/50,
        snapshot_selection='one_second_smoothed_absolute_root_error_difference_peak',
        camera=dict(distance=2.8, azimuth=135, elevation=-17, shared_nominal_center=True),
        scope='kinematic_render_of_actual_Newton_physics_and_committed_reference_no_new_dynamics',
        video_sha256=sha(output), image_sha256=sha(output.with_suffix('.png')),
        trajectory_metadata=[dict(path=str(p), metadata_sha256=sha(p), npz_sha256=m['sha256'])
                             for p, m in zip(args.trajectories, metadata)],
        mjcf_sha256=sha(mjcf), audio_sha256=sha(audio), audio_start_seconds=first['audio_start_seconds'],
        wall_seconds=time.perf_counter()-started, max_frames=args.max_frames)
    output.with_suffix('.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(dict(output=str(output), frames=frames, snapshot_seconds=manifest['snapshot_seconds']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
