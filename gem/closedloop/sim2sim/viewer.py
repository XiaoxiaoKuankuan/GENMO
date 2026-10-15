"""MuJoCo原生GLFW交互窗口与结束后的真实状态视频编码。

窗口持有独立MjData，只从物理线程复制qpos用于绘制；绝不写入正在运行的
物理MjData。单一画面中的实体机器人显示GMT控制下的MuJoCo完整动力学状态；
半透明蓝色机器人显示同一时刻的GENMO已提交参考，只在独立MjData中做运动学
正向计算和几何体渲染，不执行GMT、PD或mj_step。参考保留原始三维网格、
光照和22%蓝色半透明外观，按真实深度遮挡叠加；不会把轮廓涂成平面色块，
也不平滑或修改参考姿态。两者使用同一快照和相机，
参考姿态不做平滑、平移或时间偏移，渲染掉帧不改变真实控制步。键盘事件
仅入控制队列，暂停/重置/切歌/切模型均由物理线程在合法控制边界处理。
视频在所有控制停止后由逐50Hz真实状态生成并混入原音频，保留明确的
physics_backend=mujoco来源；这段离线编码不计作实时动力学吞吐。
"""
from __future__ import annotations
from pathlib import Path
import subprocess
import time
import numpy as np
import mujoco


class Sim2SimViewer:
    def __init__(self,runtime):
        import glfw
        self.glfw=glfw;self.runtime=runtime
        if not glfw.init():raise RuntimeError('GLFW初始化失败；无显示器时使用--headless')
        self.window=glfw.create_window(1280,800,'GENMO + GMT | MuJoCo real physics',None,None)
        if not self.window:glfw.terminate();raise RuntimeError('无法创建MuJoCo交互窗口')
        glfw.make_context_current(self.window);glfw.swap_interval(1)
        self.model=mujoco.MjModel.from_xml_path(runtime.config['paths']['mjcf'])
        self.data=mujoco.MjData(self.model);self.ref_data=mujoco.MjData(self.model)
        self.scene=mujoco.MjvScene(self.model,maxgeom=3000);self.ghost_scene=mujoco.MjvScene(self.model,maxgeom=1500)
        # 与Actor共用4090时关闭窗口阴影，离线视频仍可有阴影；不更改模型/物理。
        self.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
        self.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=False
        self.ghost_scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
        self.ghost_scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=False
        self.context=mujoco.MjrContext(self.model,mujoco.mjtFontScale.mjFONTSCALE_150)
        self.camera=mujoco.MjvCamera();self.options=mujoco.MjvOption();self.perturb=mujoco.MjvPerturb()
        self.camera.distance=2.5;self.camera.azimuth=125;self.camera.elevation=-14
        self.show_reference=True;self.mouse_last=None;self.mouse_button=None
        self.captured_episodes=set()
        glfw.set_key_callback(self.window,self._key)
        glfw.set_scroll_callback(self.window,self._scroll)
        glfw.set_mouse_button_callback(self.window,self._button)
        glfw.set_cursor_pos_callback(self.window,self._mouse)

    def _key(self,window,key,scancode,action,mods):
        g=self.glfw
        if action!=g.PRESS:return
        mapping={g.KEY_SPACE:'toggle_pause',g.KEY_R:'restart',g.KEY_N:'next',g.KEY_P:'previous',
                 g.KEY_ESCAPE:'stop'}
        if key in mapping:self.runtime.command(mapping[key])
        elif key==g.KEY_G:self.show_reference=not self.show_reference
        elif key in (g.KEY_1,g.KEY_2,g.KEY_3):
            index=key-g.KEY_1
            if index<len(self.runtime.models):self.runtime.command(f'model:{index}')

    def _scroll(self,window,x,y):self.camera.distance=float(np.clip(self.camera.distance*np.exp(-.08*y),.5,12.))
    def _button(self,window,button,action,mods):
        self.mouse_button=button if action==self.glfw.PRESS else None
        self.mouse_last=self.glfw.get_cursor_pos(window)
    def _mouse(self,window,x,y):
        if self.mouse_button is not None and self.mouse_last:
            dx,dy=x-self.mouse_last[0],y-self.mouse_last[1]
            self.camera.azimuth-=dx*.3;self.camera.elevation=float(np.clip(self.camera.elevation-dy*.2,-85,0))
        self.mouse_last=(x,y)

    def _reference_scene(self,state):
        """仅在独立MjData中计算GENMO参考的运动学姿态，随后复制可见机器人geom。"""
        self.ref_data.qpos[:]=state['reference_qpos'];mujoco.mj_forward(self.model,self.ref_data)
        mujoco.mjv_updateScene(self.model,self.ref_data,self.options,self.perturb,self.camera,
                             mujoco.mjtCatBit.mjCAT_DYNAMIC,self.ghost_scene)

    def _ghost(self,state):
        if not self.show_reference or 'reference_qpos' not in state:return
        self._reference_scene(state)
        for source in self.ghost_scene.geoms[:self.ghost_scene.ngeom]:
            if source.objtype!=mujoco.mjtObj.mjOBJ_GEOM or source.rgba[3]<=0:continue
            if self.scene.ngeom>=self.scene.maxgeom:break
            dest=self.scene.geoms[self.scene.ngeom]
            mujoco.mjv_initGeom(dest,source.type,source.size.astype(np.float64),source.pos.astype(np.float64),
                               source.mat.astype(np.float64).reshape(9),
                               np.array([.15,.65,.95,.22],dtype=np.float32))
            for field in ('dataid','objtype','objid','texcoord','modelrbound'):
                setattr(dest,field,getattr(source,field))
            dest.matid=-1;dest.category=mujoco.mjtCatBit.mjCAT_DECOR
            dest.transparent=1;self.scene.ngeom+=1

    def _render(self,rect,state):
        """恢复原三维半透明网格外观；实体与蓝色参考在同一个场景中绘制。"""
        self._ghost(state)
        mujoco.mjr_render(rect,self.scene,self.context)

    def run(self):
        g=self.glfw
        try:
            while not self.runtime.finished.is_set() and not g.window_should_close(self.window):
                frame_started=time.monotonic()
                self.runtime.viewer_heartbeat=frame_started
                state=self.runtime.snapshot()
                if 'qpos' in state:self.data.qpos[:]=state['qpos']
                mujoco.mj_forward(self.model,self.data)
                self.camera.lookat[:]=self.data.qpos[:3]
                self.camera.lookat[2]=max(.4,self.camera.lookat[2])
                mujoco.mjv_updateScene(self.model,self.data,self.options,self.perturb,self.camera,
                                      mujoco.mjtCatBit.mjCAT_ALL,self.scene)
                width,height=g.get_framebuffer_size(self.window)
                rect=mujoco.MjrRect(0,0,width,height)
                self._render(rect,state)
                reference_label='Blue ghost: GENMO kinematics' if self.show_reference else 'Blue ghost: hidden (G)'
                if self.show_reference and state.get('reference_source')=='fallback_hold':
                    reference_label='Blue ghost: HOLD fallback (not GENMO)'
                mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                    rect,'Solid: GMT + MuJoCo dynamics\n'+reference_label,'',self.context)
                title=f'{state.get("model","Loading")} | {state.get("music","")} | {state.get("status","")}'
                g.set_window_title(self.window,title)
                text=(f'Status: {state.get("status", "loading")}\n'
                    f'Music: {state.get("music_seconds",0):.2f} / {state.get("music_duration",0):.2f} s\n'
                    f'Activity: {state.get("activity",0):.3f} rad/s  Yaw error: {state.get("yaw_error",0):.3f} rad\n'
                    f'Reference reserve: {state.get("reference_remaining",0):.3f} s\n'
                    f'GENMO: {state.get("inference_seconds",0)*1000:.1f} ms  Real time: {state.get("real_time_factor",0):.3f}x\n'
                    f'Warning: {state.get("last_warning") or "-"}\n'
                    f'End: {state.get("reason") or "-"}')
                mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    rect,text,'',self.context)
                mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                    rect,'SPACE pause | R replay | N/P song | 1/2/3 model\n'
                    'G blue reference | ESC exit | physics 200 Hz / GMT 50 Hz',
                    '',self.context)
                if (self.runtime.record and state.get('status')=='finished' and
                        state.get('episode_id') not in self.captured_episodes):
                    # 只保存本应用的渲染缓冲，不截取用户桌面；物理已停，不干扰控制计时。
                    from PIL import Image
                    rgb=np.empty((height,width,3),dtype=np.uint8)
                    mujoco.mjr_readPixels(rgb,None,rect,self.context)
                    Image.fromarray(rgb[::-1]).save(self.runtime.output/(state['episode_id']+'_window.png'))
                    self.captured_episodes.add(state['episode_id'])
                g.swap_buffers(self.window);g.poll_events()
                # 窗口25Hz，物理仍200Hz；避免渲染无上限争抢GENMO所在的同一张GPU。
                time.sleep(max(.001,1/self.runtime.config['runtime']['viewer_fps']-(time.monotonic()-frame_started)))
        except Exception as exc:
            self.runtime.failure=dict(phase='viewer',type=type(exc).__name__,message=str(exc))
            self.runtime.command('abort:viewer_exception')
            raise
        finally:
            if not self.runtime.finished.is_set():self.runtime.command('stop')
            self.context.free();g.destroy_window(self.window);g.terminate()


def encode_video(directory,audio_path,initial_qpos,music_start_tick,config):
    """编码完成的真实状态轨迹；插音频前加实际站立预热的静音时段。"""
    directory=Path(directory)
    with np.load(directory/'control_trace.npz',allow_pickle=False) as trace:
        states=trace['qpos'].copy()
    settings=config['runtime'];fps=settings['video_fps'];width=settings['video_width'];height=settings['video_height']
    seconds=len(states)*.02
    model=mujoco.MjModel.from_xml_path(config['paths']['mjcf'])
    model.vis.global_.offwidth=max(model.vis.global_.offwidth,width)
    model.vis.global_.offheight=max(model.vis.global_.offheight,height)
    data=mujoco.MjData(model);camera=mujoco.MjvCamera()
    camera.distance=2.5;camera.azimuth=125;camera.elevation=-14
    raw=directory/'physics_video.partial.mp4';output=directory/'physics_with_audio.mp4'
    if raw.exists() or output.exists():raise FileExistsError('拒绝覆盖已有视频')
    cmd=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
         '-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast',
         '-crf','20','-pix_fmt','yuv420p','-threads','2',str(raw)]
    process=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    try:
        with mujoco.Renderer(model,height=height,width=width) as renderer:
            for frame in range(math_ceil_frames(seconds,fps)):
                index=int(round(frame/fps/.02))-1
                data.qpos[:]=initial_qpos if index<0 else states[min(index,len(states)-1)]
                mujoco.mj_forward(model,data);camera.lookat[:]=data.qpos[:3]
                renderer.update_scene(data,camera=camera)
                process.stdin.write(renderer.render().tobytes())
        process.stdin.close()
        if process.wait()!=0:raise RuntimeError('ffmpeg实际状态编码失败')
        delay=int(round(music_start_tick/600*1000))
        subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-i',str(raw),'-i',str(audio_path),
            '-filter_complex',f'[1:a]adelay={delay}:all=1,apad[a]','-map','0:v','-map','[a]',
            '-c:v','copy','-c:a','aac','-t',str(seconds),'-movflags','+faststart',
            '-metadata','comment=physics_backend=mujoco; online GENMO+GMT; recorded actual dynamics',str(output)],check=True)
        from .diagnostics import write_json
        write_json(directory/'video.json',dict(physics_backend='mujoco',source='control_trace.npz',
            audio=str(audio_path),warmup_silence_seconds=delay/1000,seconds=seconds,fps=fps,
            video=str(output),encoding_after_control_stopped=True))
    finally:
        if process.poll() is None:process.kill();process.wait()
        raw.unlink(missing_ok=True)
    return output


def math_ceil_frames(seconds,fps):
    import math
    return max(1,int(math.ceil(seconds*fps-1e-9)))
