# -*- coding: utf-8 -*-
"""
计算 AVI 视频的帧差（支持跳帧差分），并用伪彩色显示 / 保存结果。

依赖:
    pip install opencv-python numpy

说明:
    - 帧差 = |当前帧灰度 - 参考帧灰度|
    - 参考帧默认为前一帧；当 FRAME_INTERVAL > 1 时，参考帧为 N 帧之前的那一帧
      （即跳帧差分，用于检测更慢/更长的运动变化）
    - 伪彩映射使用 cv2.COLORMAP_JET（冷色=差异小，暖色=差异大）
    - 由于帧差异通常很小，用 DIFF_SCALE 放大后再映射，效果更明显
"""

import os
from collections import deque

import cv2
import numpy as np

# ==================== 配置（路径写死在这里） ====================
VIDEO_PATH = r"./Supplementary_Video_1.avi"   # 输入 AVI 路径
OUTPUT_PATH = r"D:\videos\frame_diff.avi"         # 输出结果路径
SHOW = True                                       # 是否实时显示
SAVE = False                                      # 是否保存结果视频
DIFF_SCALE = 5.0                                  # 帧差放大系数（1.0 表示不放大）
THRESHOLD = 0                                     # 二值化阈值，0 表示不做二值化
FRAME_INTERVAL = 15                                # 跳帧间隔：1=相邻帧，N=与 N 帧前的帧做差
# ===============================================================


def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"[错误] 视频文件不存在: {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频: {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 240:
        fps = 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频信息: {w}x{h}, {fps:.2f} fps, 约 {total} 帧, 跳帧间隔 = {FRAME_INTERVAL}")

    interval = max(1, int(FRAME_INTERVAL))

    # 创建输出视频写入器
    writer = None
    if SAVE:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))
        if not writer.isOpened():
            print("[警告] 无法创建输出视频，将只显示不保存")
            writer = None

    if SHOW:
        cv2.namedWindow("Frame Diff (Pseudo Color)", cv2.WINDOW_NORMAL)

    delay = max(1, int(1000 / fps))

    # 用队列缓存最近 interval 帧的灰度图；队首即为"参考帧"
    buffer = deque(maxlen=interval)

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1) 计算帧差：只有当缓冲区里已经有 interval 帧时才做有效差分
        if len(buffer) == interval:
            ref_gray = buffer[0]                 # 队首 = interval 帧之前的那一帧
            diff = cv2.absdiff(gray, ref_gray)
            ref_idx = idx - interval
        else:
            # 前 interval 帧没有足够的参考帧，输出全黑
            diff = np.zeros_like(gray)
            ref_idx = None

        # 2) 可选：阈值处理，去掉微小噪声
        if THRESHOLD > 0:
            _, diff = cv2.threshold(diff, THRESHOLD, 255, cv2.THRESH_BINARY)

        # 3) 放大差异，让伪彩更明显（自动饱和到 255）
        diff = cv2.convertScaleAbs(diff, alpha=DIFF_SCALE, beta=0)

        # 4) 伪彩映射
        color = cv2.applyColorMap(diff, cv2.COLORMAP_JET)

        # 5) 叠加帧号信息
        if ref_idx is not None:
            text = f"Frame: {idx} vs {ref_idx}"
        else:
            text = f"Frame: {idx} (warming up)"
        cv2.putText(color, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 6) 更新缓存：加入当前帧；超过 maxlen 会自动弹出最老的
        buffer.append(gray)

        if writer is not None:
            writer.write(color)

        if SHOW:
            cv2.imshow("Frame Diff (Pseudo Color)", color)
            key = cv2.waitKey(delay) & 0xFF
            if key == ord("q") or key == 27:   # 按 q 或 ESC 退出
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print(f"处理完成，共处理 {idx} 帧")
    if SAVE and writer is not None:
        print(f"结果已保存到: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
