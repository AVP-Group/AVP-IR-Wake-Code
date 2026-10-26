import os
import cv2
import numpy as np


# ==================== RCA 算法实现 ====================
class RetinaCellularAutomaton:
    """视网膜启发式元胞自动机 - 显式运动建模（Magnocellular 通路）"""

    def __init__(self, height, width):
        self.H = height
        self.W = width

        # ---- 论文中给出的固定参数 (Section 3.2) ----
        self.theta_p = 0.1
        self.g_p = 1.5
        self.sigma_h = 0.3
        self.theta_b = 0.2
        self.g_b = 2.0
        self.alpha = 0.8
        self.beta = 1.2
        self.gamma_a = 0.5
        self.gamma_r = 0.7
        self.g_m = 2.5
        self.theta_m = 0.3
        self.eta_m = 0.7
        self.gamma_p = 0.8

        # ---- 固定卷积核 ----
        k1d = cv2.getGaussianKernel(3, 1.0)
        self.K_h = (k1d @ k1d.T).astype(np.float32)
        self.K_m = self._make_mexican_hat(size=5, sigma1=1.0,
                                          sigma2=2.0, w_surr=0.5)

        # ---- 跨帧内部状态 ----
        self.reset()

    @staticmethod
    def _make_mexican_hat(size, sigma1, sigma2, w_surr):
        r = size // 2
        ax = np.arange(-r, r + 1, dtype=np.float32)
        xx, yy = np.meshgrid(ax, ax)
        dog = (np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma1 ** 2))
               - w_surr * np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma2 ** 2)))
        dog = dog / np.sum(np.abs(dog))
        return dog.astype(np.float32)

    def reset(self):
        """在序列开始时重置所有状态"""
        self.S_prev_b = np.zeros((self.H, self.W), dtype=np.float32)
        self.S_prev_a = np.zeros((self.H, self.W), dtype=np.float32)
        self._first_frame = True

    @staticmethod
    def _adapt(I, theta_p, g_p):
        out = np.where(I > theta_p, g_p * np.tanh(I - theta_p), 0.1 * I)
        return out.astype(np.float32)

    def _enhance(self, x):
        x = np.maximum(x, 0.0) ** self.gamma_p
        mx = x.max()
        if mx > 1e-12:
            x = x / mx
        x = cv2.bilateralFilter(x.astype(np.float32), 5, 0.1, 0.1)
        mx = x.max()
        if mx > 1e-12:
            x = x / mx * 255.0
        return x

    def run(self, frame_gray):
        """处理一帧，返回该帧对应的运动图 M_t（float32, [0,255]）"""
        if frame_gray.dtype == np.uint8:
            I = frame_gray.astype(np.float32) / 255.0
        else:
            I = frame_gray.astype(np.float32)

        # Layer 1: Photoreceptors
        S_p = self._adapt(I, self.theta_p, self.g_p)

        # Layer 2: Horizontal cells
        N = cv2.filter2D(S_p, -1, self.K_h, borderType=cv2.BORDER_REPLICATE)
        S_h = np.maximum(S_p - self.sigma_h * N, 0.0)

        # Layer 3: Bipolar cells (ON/OFF)
        S_ON = np.maximum(self.g_b * (S_h - self.theta_b), 0.0)
        S_OFF = np.maximum(self.g_b * (-S_h - self.theta_b), 0.0)
        C_t = S_ON + S_OFF

        # Layer 4: Amacrine cells
        if self._first_frame:
            gx = cv2.Sobel(C_t, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(C_t, cv2.CV_32F, 0, 1, ksize=3)
            R_t = self.beta * np.sqrt(gx * gx + gy * gy)
        else:
            R_t = self.beta * np.abs(C_t - self.S_prev_b)
        S_a = self.alpha * self.S_prev_a + (1.0 - self.alpha) * R_t

        # Layer 5: Magnocellular ganglion cells
        I_t = C_t + self.gamma_a * S_a
        M_s = cv2.filter2D(I_t, -1, self.K_m, borderType=cv2.BORDER_REPLICATE)
        M_r = self.gamma_r * S_a
        x_thr = M_s + M_r
        S_m = self.g_m * np.maximum(0.0, np.tanh(x_thr - self.theta_m))

        # Output generation
        M_t = self._enhance(self.eta_m * S_m + (1.0 - self.eta_m) * S_a)

        # Update temporal memory
        self.S_prev_b = C_t.copy()
        self.S_prev_a = S_a.copy()
        self._first_frame = False

        return M_t


VIDEO_PATH = r"./20250918-061225.avi"          
OUTPUT_ROOT = r"./output/20250918-061225/"    

SAVE_GRAY = True      # 是否保存原始灰度帧
SAVE_MOTION = True    # 是否保存 RCA 运动图
SAVE_HEAT = True      # 是否保存热力图
SAVE_COMPARE = True   # 是否保存三联对比图


def ensure_dir(path):
    """目录不存在则新建（含多级目录）"""
    os.makedirs(path, exist_ok=True)
    return path


def main():
    if not os.path.isfile(VIDEO_PATH):
        print(f"[错误] 视频文件不存在: {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频: {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[信息] 视频: {w}x{h} @ {fps:.2f} FPS, 共 {total} 帧")

    # ---- 建立输出子目录（不存在自动新建）----
    ensure_dir(OUTPUT_ROOT)
    dir_gray = ensure_dir(os.path.join(OUTPUT_ROOT, "gray")) if SAVE_GRAY else None
    dir_motion = ensure_dir(os.path.join(OUTPUT_ROOT, "motion")) if SAVE_MOTION else None
    dir_heat = ensure_dir(os.path.join(OUTPUT_ROOT, "heat")) if SAVE_HEAT else None
    dir_compare = ensure_dir(os.path.join(OUTPUT_ROOT, "compare")) if SAVE_COMPARE else None

    # 初始化 RCA
    rca = RetinaCellularAutomaton(height=h, width=w)

    idx = 0
    energies = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1

        # 转灰度（红外视频本身就是单通道）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        # ---- RCA 前向 ----
        M_t = rca.run(gray)
        energies.append(float(M_t.mean()))

        M_disp = np.clip(M_t, 0, 255).astype(np.uint8)

        # ---- 逐帧保存为 PNG（文件名带 6 位序号，便于按时间顺序拼接） ----
        name = f"{idx:06d}.png"

        if SAVE_GRAY:
            cv2.imwrite(os.path.join(dir_gray, name), gray)

        if SAVE_MOTION:
            cv2.imwrite(os.path.join(dir_motion, name), M_disp)

        if SAVE_HEAT:
            M_heat = cv2.applyColorMap(M_disp, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(dir_heat, name), M_heat)

        if SAVE_COMPARE:
            frame_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            M_3ch = cv2.cvtColor(M_disp, cv2.COLOR_GRAY2BGR)
            M_heat = cv2.applyColorMap(M_disp, cv2.COLORMAP_JET)
            compare = np.hstack([frame_3ch, M_3ch, M_heat])
            cv2.imwrite(os.path.join(dir_compare, name), compare)

        if idx % 30 == 0 or idx == total:
            print(f"[进度] {idx}/{total}  当前运动强度: {energies[-1]:.2f}")

    cap.release()

    # ---- 统计 ----
    if energies:
        arr = np.array(energies)
        print("\n================ 统计 ================")
        print(f"总帧数:       {idx}")
        print(f"平均运动强度: {arr.mean():.2f}")
        print(f"最大运动强度: {arr.max():.2f}  (第 {arr.argmax()+1} 帧)")
        print(f"最小运动强度: {arr.min():.2f}  (第 {arr.argmin()+1} 帧)")
        csv_path = os.path.join(OUTPUT_ROOT, "motion_energy.csv")
        np.savetxt(csv_path, arr, fmt="%.4f", header="motion_energy", comments="")
        print(f"运动强度曲线已保存: {csv_path}")

    print("\n================ 输出目录 ================")
    print(f"根目录:   {os.path.abspath(OUTPUT_ROOT)}")
    if SAVE_GRAY:    print(f"  原始灰度: {os.path.abspath(dir_gray)}")
    if SAVE_MOTION:  print(f"  RCA 运动图: {os.path.abspath(dir_motion)}")
    if SAVE_HEAT:    print(f"  热力图:   {os.path.abspath(dir_heat)}")
    if SAVE_COMPARE: print(f"  三联对比: {os.path.abspath(dir_compare)}")


if __name__ == "__main__":
    main()
