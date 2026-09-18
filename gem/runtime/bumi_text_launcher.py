"""独立部署包配置启动器。

只从deployment.ini读取模型后端、资产、T5和窗口开关，使用当前包的Python启动常驻
文本控制台。部署清单校验模型资产，配置不纳入不可变模型指纹，因此用户可以直接改文件。
没有ROS、Redis、GMT地址或任何控制器网络端口。
"""

from pathlib import Path
import configparser
import os
import sys


def command(root):
    root = Path(root).resolve()
    config = configparser.ConfigParser()
    if not config.read(root / "deployment.ini"):
        raise FileNotFoundError("部署目录缺少deployment.ini")
    runtime = config["runtime"]

    def path(name):
        return str((root / Path(runtime[name]).expanduser()).resolve())

    if not Path(path("t5_model")).is_dir():
        raise FileNotFoundError("请在deployment.ini的t5_model填写本地T5-3B完整目录")
    args = [
        sys.executable,
        "-B",
        "-u",
        str(root / "scripts/demo/demo_bumi_text.py"),
        "--console",
        "--backend",
        runtime["backend"],
        "--device",
        runtime["device"],
        "--deployment-manifest",
        path("manifest"),
        "--t5-model",
        path("t5_model"),
        "--output-root",
        path("output_root"),
        "--num-frames",
        runtime["num_frames"],
        "--ddim-steps",
        runtime["ddim_steps"],
        "--seed",
        runtime["seed"],
    ]
    if config.getboolean("preview", "enabled", fallback=False):
        args.append("--preview")
    return args


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    args = command(root) + sys.argv[1:]
    os.execv(sys.executable, args)
