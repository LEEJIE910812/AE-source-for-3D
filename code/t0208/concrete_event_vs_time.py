import os
import numpy as np
import matplotlib.pyplot as plt
from concrete_common import load_results_pickle

# ===== 固定路徑與平滑參數 =====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
in_path = os.path.join(SCRIPT_DIR, "results_test_2_90.pkl")
OUT_DIR = os.path.dirname(in_path)

# Event rate 曲線的取樣數與平滑視窗；通常不需要改。
DERIV_NPTS = 2000
SMOOTH_WIN = 9          # 奇數，=1 表示不平滑
EPS_T = 1e-9            # 避免 0 長度

os.makedirs(OUT_DIR, exist_ok=True)

# ===== 讀檔 =====
data = load_results_pickle(in_path)
all_results = data["results"]


# ===== 萃取每個 CSV 的 Grid / TRF / HypoDD 資料 =====
file_names = []
method_data_list = []
kept_results = []

for res in all_results:
    events = res["events"]

    def grab(method):
        xs, ys, zs, t0s = [], [], [], []

        for e in events:
            if method == "grid":
                x, y, z = res["g0_res"][e]
                t0 = res["g0_t0"][e]

            elif method == "trf":
                x, y, z = res["g1_res"].get(e, res["g0_res"][e])
                t0 = res["g1_t0"].get(e, res["g0_t0"][e])

            else:  # hypodd
                if e in res["hypodd"]:
                    x, y, z, t0 = res["hypodd"][e]
                else:
                    x, y, z = res["g1_res"].get(e, res["g0_res"][e])
                    t0 = res["g1_t0"].get(e, res["g0_t0"][e])

            xs.append(x)
            ys.append(y)
            zs.append(z)
            t0s.append(float(t0))

        return (
            np.array(xs),
            np.array(ys),
            np.array(zs),
            np.array(t0s, float)
        )

    md = {
        "grid": grab("grid"),
        "trf": grab("trf"),
        "hypodd": grab("hypodd")
    }

    # 檢查這個 CSV 是否有有效 HypoDD 時間
    hypodd_t0s = md["hypodd"][3]
    good = np.isfinite(hypodd_t0s)

    if not good.any():
        print(f"Skip empty CSV segment: {os.path.basename(res['file'])}")
        continue

    kept_results.append(res)
    method_data_list.append(md)
    file_names.append(os.path.basename(res["file"]))

all_results = kept_results
n_csv = len(method_data_list)


# ===== 計算 cumulative event number 和 event rate =====
def cumulative_and_rate(t_events, t0, t1, npts=2000, smooth_win=9):
    t = np.asarray(t_events, float)
    t = t[np.isfinite(t)]
    t = t[(t >= t0) & (t <= t1)]

    if t.size:
        t.sort()

    npts = max(3, int(npts))
    tg = np.linspace(t0, t1, npts)

    N = np.searchsorted(t, tg, side="right").astype(float)

    dt = (t1 - t0) / (npts - 1) if npts > 1 else 1.0

    rate = np.zeros_like(N)

    if npts >= 3:
        rate[1:-1] = (N[2:] - N[:-2]) / (2 * dt)
        rate[0] = (N[1] - N[0]) / dt
        rate[-1] = (N[-1] - N[-2]) / dt

    # 平滑 event rate
    if smooth_win >= 3 and smooth_win % 2 == 1:
        k = smooth_win // 2
        pad = np.pad(rate, (k, k), mode="edge")
        csum = np.cumsum(pad)
        rate = (csum[2 * k:] - csum[:-2 * k]) / (2 * k)

    return tg, N, rate


# ===== 每個 test 分開畫 HypoDD：Events vs Time + Event Rate =====
for i in range(n_csv):

    csv_name = file_names[i]
    csv_label = os.path.splitext(csv_name)[0]

    # 取這個 test 的 HypoDD t0
    _, _, _, t0s = method_data_list[i]["hypodd"]
    t0s = np.asarray(t0s, float)

    # 去掉 nan
    mask = np.isfinite(t0s)
    t0s = t0s[mask]

    if t0s.size == 0:
        print(f"Skip empty HypoDD file: {csv_name}")
        continue

    # 按時間排序
    t0s = np.sort(t0s)

    # ===== 讓每個 test 的時間從 0 秒開始 =====
    t_ref = t0s[0]
    hypodd_t_local = t0s - t_ref

    # 這個 test 自己的 event index
    hypodd_y_local = np.arange(1, len(hypodd_t_local) + 1)

    # 這個 test 的 local time 範圍
    t0_local = 0.0
    t1_local = max(EPS_T, float(np.nanmax(hypodd_t_local)))

    # ===== 建立圖 =====
    fig, axes = plt.subplots(
        2, 1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw=dict(hspace=0.18)
    )

    # ===== 上圖：Events vs Time =====
    axes[0].plot(
        hypodd_t_local,
        hypodd_y_local,
        marker="^",
        ms=1.8,
        alpha=0.75,
        ls="None",
        color="tab:green"
    )

    axes[0].set_xlim(t0_local, t1_local)
    axes[0].set_ylim(0.5, len(hypodd_y_local) + 0.5)
    axes[0].set_ylabel("Event index")
    axes[0].set_title("Events vs Time")
    axes[0].grid(True, axis="x", alpha=0.35)

    if len(hypodd_y_local) > 25:
        step = max(1, len(hypodd_y_local) // 25)
        axes[0].set_yticks(np.arange(1, len(hypodd_y_local) + 1, step))

    # ===== 下圖：Event Rate =====
    tg, N, rate = cumulative_and_rate(
        hypodd_t_local,
        t0_local,
        t1_local,
        npts=DERIV_NPTS,
        smooth_win=SMOOTH_WIN
    )

    axes[1].plot(
        tg,
        rate,
        lw=1.3,
        color="tab:green"
    )

    axes[1].set_xlim(t0_local, t1_local)
    axes[1].set_ylabel("dN/dt (events/s)")
    axes[1].set_xlabel("Local time (s)")
    axes[1].set_title("Event Rate")
    axes[1].grid(True, alpha=0.35)

    fig.suptitle(f"{csv_label}", fontsize=14)

    # ===== 存圖 =====
    out_file = os.path.join(
        OUT_DIR,
        f"{csv_label}_event_time_rate.png"
    )

    fig.savefig(out_file, dpi=220, bbox_inches="tight")
    print("Saved:", out_file)

plt.show()
