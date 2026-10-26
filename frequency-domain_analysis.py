# -*- coding: utf-8 -*-
"""
pip install opencv-python numpy PyWavelets
"""

import os
import cv2
import numpy as np
import pywt

IMAGE_PATH   = r"D:\images\frame_00002.jpg"
OUT_DIR      = r"D:\wavelet_output"
WAVELET      = "haar"            # 小波基: 'haar','db2','sym4','bior1.3' ...
LEVEL        = 2                 # 分解层数
# ============================================================


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """把有正负、动态范围大的小波系数归一化到 0-255 便于保存为图像"""
    arr = arr.astype(np.float32)
    amin, amax = arr.min(), arr.max()
    if amax - amin < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - amin) / (amax - amin) * 255.0).astype(np.uint8)


def wavelet_decompose_and_save(image_path: str,
                               out_dir: str = "wavelet_output",
                               wavelet: str = "haar",
                               level: int = 2) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # ---------- 1. 读取图像 ----------
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    print(f"[√] 已读取图像, shape={img.shape}, dtype={img.dtype}")

    # ---------- 2. 转灰度 ----------
    if img.ndim == 2:
        gray = img                              # 本来就是灰度
    elif img.ndim == 3:
        if img.shape[2] == 4:                   # 去掉 alpha 通道
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("不支持的图像维度")

    print(f"[√] 已转灰度图, shape={gray.shape}")

    # ---------- 3. 小波分解 ----------
    coeffs = pywt.wavedec2(gray.astype(np.float32),
                           wavelet=wavelet, level=level,
                           mode="symmetric")

    # 最深层低频 LL
    cv2.imwrite(os.path.join(out_dir, f"gray_L{level}_LL.png"),
                normalize_to_uint8(coeffs[0]))

    # 各层细节 (LH / HL / HH)
    for i, (cH, cV, cD) in enumerate(coeffs[1:], start=1):
        lvl = level - i + 1
        cv2.imwrite(os.path.join(out_dir, f"gray_L{lvl}_LH.png"),
                    normalize_to_uint8(cH))
        cv2.imwrite(os.path.join(out_dir, f"gray_L{lvl}_HL.png"),
                    normalize_to_uint8(cV))
        cv2.imwrite(os.path.join(out_dir, f"gray_L{lvl}_HH.png"),
                    normalize_to_uint8(cD))

    # ---------- 4. 最外层子带拼成一张金字塔可视化图 ----------
    cH, cV, cD = coeffs[1]
    LL_vis = normalize_to_uint8(coeffs[0])
    if LL_vis.shape != cH.shape:               # 多层时 LL 尺寸更小，先放大
        LL_vis = cv2.resize(LL_vis, (cH.shape[1], cH.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
    top = np.hstack([LL_vis, normalize_to_uint8(cH)])
    bot = np.hstack([normalize_to_uint8(cV), normalize_to_uint8(cD)])
    cv2.imwrite(os.path.join(out_dir, "gray_pyramid.png"),
                np.vstack([top, bot]))

    # ---------- 5. 顺便把灰度图本身也存一份，便于对照 ----------
    cv2.imwrite(os.path.join(out_dir, "gray_input.png"), gray)

    print(f"[√] 所有子带已保存到: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    wavelet_decompose_and_save(IMAGE_PATH,
                               out_dir=OUT_DIR,
                               wavelet=WAVELET,
                               level=LEVEL)
