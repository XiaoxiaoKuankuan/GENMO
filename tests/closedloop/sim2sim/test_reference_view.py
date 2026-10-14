"""参考显示与真实状态隔离的MuJoCo场景回归。

使用内存构建的微型刚体模型，无GPU、声音设备或窗口。验证分屏参考只改变显示
颜色，不改输入qpos/相机坐标/实际MjData；叠加层仅复制可见机器人geom，避免将
辅助site或隐藏碰撞形状重新变成蓝色实体。实际GL渲染由独立有界验收覆盖。
"""
import numpy as np
import mujoco
from gem.closedloop.sim2sim.viewer import Sim2SimViewer


def scene_view():
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="floor" type="plane" size="3 3 .1"/>
      <body name="robot" pos="0 0 1"><freejoint/>
        <geom name="body" type="box" size=".1 .1 .2"/>
        <geom name="hidden" type="sphere" size=".05" rgba="1 0 0 0"/>
        <site name="sensor" size=".02"/>
      </body></worldbody></mujoco>''')
    view=Sim2SimViewer.__new__(Sim2SimViewer)
    view.model=model;view.data=mujoco.MjData(model);view.ref_data=mujoco.MjData(model)
    view.scene=mujoco.MjvScene(model,maxgeom=50);view.ghost_scene=mujoco.MjvScene(model,maxgeom=50)
    view.options=mujoco.MjvOption();view.perturb=mujoco.MjvPerturb();view.camera=mujoco.MjvCamera()
    view.show_reference=True
    mujoco.mj_forward(model,view.data)
    mujoco.mjv_updateScene(model,view.data,view.options,view.perturb,view.camera,
                          mujoco.mjtCatBit.mjCAT_ALL,view.scene)
    return view


def test_split_reference_preserves_pose_without_touching_actual_state():
    view=scene_view();before=view.data.qpos.copy();reference=before.copy();reference[0]=.2
    expected=mujoco.MjData(view.model);expected.qpos[:]=reference;mujoco.mj_forward(view.model,expected)
    state=dict(reference_qpos=reference.copy())
    view._opaque_reference(state)
    np.testing.assert_array_equal(view.data.qpos,before)
    np.testing.assert_array_equal(view.ref_data.qpos,reference)
    np.testing.assert_array_equal(state['reference_qpos'],reference)
    np.testing.assert_array_equal(view.ref_data.geom_xpos,expected.geom_xpos)
    body=view.model.geom('body').id
    geom=next(g for g in view.ghost_scene.geoms[:view.ghost_scene.ngeom]
              if g.objtype==mujoco.mjtObj.mjOBJ_GEOM and g.objid==body)
    assert geom.rgba[3]==1 and not geom.transparent and geom.matid==-1


def test_overlay_does_not_turn_sites_or_hidden_geoms_into_blue_shapes():
    view=scene_view();count=view.scene.ngeom
    view._ghost(dict(reference_qpos=view.data.qpos.copy()))
    assert view.scene.ngeom-count==1
    geom=view.scene.geoms[count]
    assert geom.objtype==mujoco.mjtObj.mjOBJ_GEOM and geom.objid==view.model.geom('body').id
