# -*- coding: utf-8 -*-
"""
pip install numpy
"""

from pathlib import Path
import csv
import numpy as np


CSV_PATH = r"./data/Fig3-b.csv"        
OUT_PREFIX = r"./correlation" 
MAX_LAG = 500                                       
SUMMARY_DELIM = "\t"                           
# ===============================================================


def load_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        if "\t" in sample:
            delim = "\t"
        elif "," in sample:
            delim = ","
        elif ";" in sample:
            delim = ";"
        else:
            delim = None
        reader = csv.reader(f, delimiter=delim) if delim else csv.reader(f)
        rows = [r for r in reader if r and any(c.strip() for c in r)]

    header = [h.strip() for h in rows[0]]

    body = []
    for r in rows[1:]:
        try:
            body.append([float(v) for v in r])
        except ValueError:
            continue

    data = np.asarray(body, dtype=float)
    if data.ndim != 2:
        raise ValueError("数据读取失败")

    # 自动识别并跳过时间/帧号/索引列
    ignore_keys = ["time", "frame", "index", "unnamed"]
    skip_idx = set()
    for i, h in enumerate(header):
        h_lower = h.lower().strip()
        if not h_lower or any(k in h_lower for k in ignore_keys):
            skip_idx.add(i)
    
    # 如果没识别出要跳过的列，为了防呆，默认跳过第一列（假设是帧号）
    if not skip_idx:
        skip_idx.add(0)

    keep_idx = [i for i in range(len(header)) if i not in skip_idx]
    if not keep_idx:
        raise ValueError("未能找到有效的数据列，请检查CSV表头")

    series = data[:, keep_idx].T
    labels = [header[i] for i in keep_idx]
    return series, labels


def autocorrelation(y, maxlag):
    y = y - np.mean(y)
    den = float(np.dot(y, y))
    if den == 0:
        den = 1e-12
    maxlag = min(maxlag, len(y) - 1)
    lags = np.arange(maxlag + 1)
    ac = np.zeros_like(lags, dtype=float)
    for i, l in enumerate(lags):
        if l == 0:
            ac[i] = 1.0
        else:
            ac[i] = np.dot(y[:-l], y[l:]) / den
    return lags, ac


def crosscorrelation(y1, y2, maxlag):
    y1 = y1 - np.mean(y1)
    y2 = y2 - np.mean(y2)
    den = float(np.sqrt(np.dot(y1, y1) * np.dot(y2, y2)))
    if den == 0:
        den = 1e-12
    maxlag = min(maxlag, len(y1) - 1, len(y2) - 1)
    lags = np.arange(-maxlag, maxlag + 1)
    out = np.zeros(len(lags))
    for k, l in enumerate(lags):
        if l > 0:
            out[k] = np.dot(y1[:-l], y2[l:]) / den
        elif l < 0:
            m = -l
            out[k] = np.dot(y1[m:], y2[:-m]) / den
        else:
            out[k] = np.dot(y1, y2) / den
    return lags, out


def save_autocorrelation(series, labels):
    ac = []
    for y in series:
        _, a = autocorrelation(y, MAX_LAG)
        ac.append(a)
    lag = np.arange(ac[0].size)
    ac = np.column_stack([lag] + ac)
    np.savetxt(
        OUT_PREFIX + "_autocorrelation.csv",
        ac, delimiter=",",
        header="lag," + ",".join(["AC_" + s for s in labels]),
        comments="",
    )


def save_crosscorrelation(series, labels):
    rows = []
    for i in range(len(series)):
        for j in range(i + 1, len(series)):
            lag, c = crosscorrelation(series[i], series[j], MAX_LAG)
            for l, v in zip(lag, c):
                rows.append([labels[i], labels[j], int(l), float(v)])

    np.savetxt(
        OUT_PREFIX + "_crosscorrelation.csv",
        np.asarray(rows, dtype=object),
        fmt="%s", delimiter=",",
        header="A,B,lag,corr", comments="",
    )


def save_matrices(series, labels):
    n = len(series)
    pos = np.zeros((n, n))
    neg = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                pos[i, j] = 1.0
                neg[i, j] = 1.0
                continue
            lag, corr = crosscorrelation(series[i], series[j], MAX_LAG)
            vp = corr[lag > 0]
            vn = corr[lag < 0]
            # 严格根据公式 (5)：正取最大，负取最小
            pos[i, j] = np.max(vp) if vp.size else 0.0
            neg[i, j] = np.min(vn) if vn.size else 0.0

    header_lbl = ",".join(labels)
    np.savetxt(
        OUT_PREFIX + "_positive_matrix.csv",
        pos, delimiter=",", fmt="%.6f",
        header=header_lbl, comments="",
    )
    np.savetxt(
        OUT_PREFIX + "_negative_matrix.csv",
        neg, delimiter=",", fmt="%.6f",
        header=header_lbl, comments="",
    )


def _write_pair_summary(rows, suffix):
    rows.sort(key=lambda r: (r[0], r[1]))
    out_path = OUT_PREFIX + "_pair_summary_" + suffix + ".csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=SUMMARY_DELIM)
        w.writerow(["point_1", "point_2", "correlation", "lag"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.8f}", r[3]])


def save_pair_summary(series, labels):
    pos_rows = []
    neg_rows = []

    for i in range(len(series)):
        for j in range(i + 1, len(series)):
            lag, corr = crosscorrelation(series[i], series[j], MAX_LAG)

            # 修正 1: positive 取最大值 (而不是绝对值最大)
            mask_p = lag > 0
            if np.any(mask_p):
                idxs = np.where(mask_p)[0]
                best = idxs[np.argmax(corr[idxs])] 
                pos_rows.append([labels[i], labels[j],
                                 float(corr[best]), int(lag[best])])

            # 修正 2: negative 取最小值 (而不是绝对值最大)
            mask_n = lag < 0
            if np.any(mask_n):
                idxs = np.where(mask_n)[0]
                best = idxs[np.argmin(corr[idxs])] 
                neg_rows.append([labels[i], labels[j],
                                 float(corr[best]), int(lag[best])])

    _write_pair_summary(pos_rows, "positive")
    _write_pair_summary(neg_rows, "negative")


def main():
    series, labels = load_csv(CSV_PATH)
    print("series shape:", series.shape)
    print("labels:", labels)

    save_autocorrelation(series, labels)
    save_crosscorrelation(series, labels)
    save_matrices(series, labels)
    save_pair_summary(series, labels)

    print("Finished")


if __name__ == "__main__":
    main()
