# BUMI 纯运动学显示资源

源文件取自经核验的 robot_retargeter BUMI3 资产。`mjcf/bumi3_retarget.xml` 保持原文，
SHA256 为 `fe93472dd764704fe8389b0f82052ae84ed8bc90f6d71b1467872f86e08a9ad3`，
与 s350000 模型的 kinematics 记录一致。`meshes/` 仅携带 XML 实际引用的 22 个文件。

`manifest.json` 记录所有 XML/mesh 字节数和 SHA256，以及匹配的 kinematics SHA256。
不要直接编辑、替换 XML 或 mesh 而绕过指纹校验；更换机器人资产时需要重新验证关节
顺序、坐标约定和正向运动学，再生成对应清单。运行时不读取来源仓库的绝对路径。

地面、灯光和机器人外观来自源 XML。预览程序只设置 qpos 并执行 mj_forward，不执行
mj_step；XML 中保留的惯量、接触和关节动力学参数不代表已经进行动力学仿真。
