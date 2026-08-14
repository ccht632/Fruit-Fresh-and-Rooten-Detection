"""
camera_app.py

团队共用的摄像头demo统一入口。
通过 --model 参数选择用SSD还是YOLO做检测，
两套实现各自独立成文件（camera_app_ssd.py / camera_app_yolo.py），
避免两套依赖库（torchvision vs ultralytics）互相干扰，也避免多人同时改一个文件造成git冲突。

用法：
    python camera_app.py --model ssd
    python camera_app.py --model yolo
"""

import argparse
import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["ssd", "yolo"], required=True,
                         help="选择用哪个模型做实时检测")
    # 其余参数直接透传给对应脚本（比如 --threshold, --nms-thresh 等）
    args, extra_args = parser.parse_known_args()

    if args.model == "ssd":
        target_script = os.path.join(SCRIPT_DIR, "camera_app_ssd.py")
    else:
        target_script = os.path.join(SCRIPT_DIR, "camera_app_yolo.py")

    if not os.path.exists(target_script):
        print(f"⚠️ 找不到脚本: {target_script}")
        return

    # 用subprocess调用对应脚本，而不是直接import，
    # 这样即使当前环境没装ultralytics（或没装torch），
    # 只要没选到那个模型，就不会因为import失败而报错
    subprocess.run([sys.executable, target_script] + extra_args)


if __name__ == "__main__":
    main()