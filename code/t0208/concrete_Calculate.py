from __future__ import annotations

from itertools import combinations
from pathlib import Path
import csv
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import diags, lil_matrix
from scipy.sparse.linalg import lsqr

from concrete_common import (
    build_default_geometry,
    point_mask_for_geometry,
    project_to_geometry,
    save_results_pickle,
)


# ========================== 使用者設定：通常只需要改這幾行 ==============================
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_DIR = SCRIPT_DIR / "csv"
OUTPUT_PATH = SCRIPT_DIR / "results_test_2_90.pkl"

CSV_PATTERN = "test_2_90.csv"

# 至少幾個 channel 有 pick time 才定位。
MIN_CHANNELS = 4

# Grid search 的初始速度。
V0 = 3400.0

# False: 只記錄速度品質不好的 block，仍繼續定位；True: 直接跳過。
SKIP_BAD_VELOCITY_BLOCKS = False


# ========================== 固定定位參數：一般不用改 ==============================

# 速度品質門檻。
MIN_VELOCITY = 3000.0
MAX_VELOCITY_RMSE = 1000.0

# Grid search 網格設定。
COARSE_SHAPE = (16, 16, 16)
GLOBAL_SHAPE = (50, 50, 50)
LOCAL_SHAPE = (16, 16, 16)
REFINE_FACTOR = 4
GRID_TOL_DIST = 0.001
GRID_MAX_ITER = 3

# TRF least-squares 設定。
TRF_MAX_NFEV = 1000
TRF_XTOL = 1e-9
TRF_FTOL = 1e-9
TRF_GTOL = 1e-9

# HypoDD 設定。
HYPODD_MAX_OUTER_ITER = 3
HYPODD_MAX_INNER_ITER = 2
HYPODD_TOL_POS = 1e-3
HYPODD_TOL_RW = 1e-4
HYPODD_MAX_STEP = 0.005
BOUNDARY_EPS = 1e-6
FALLBACK_BOUNDARY_HYPODD_TO_TRF = True


# ========================== 試體與 sensor ==============================
geometry = build_default_geometry()
sensor_positions = geometry.sensor_positions
station_ids = sorted(sensor_positions.keys())
cuboids = geometry.cuboids
specimen = geometry.specimen
cuboid_bounds = geometry.cuboid_bounds
xmin, xmax = geometry.xmin, geometry.xmax
ymin, ymax = geometry.ymin, geometry.ymax
zmin, zmax = geometry.zmin, geometry.zmax

specimen_diagonal = float(np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin]))
full_bounds = [(xmin, xmax), (ymin, ymax), (zmin, zmax)]
grid_cache: dict[tuple[tuple[tuple[float, float], ...], tuple[int, int, int], tuple[int, ...]], tuple[np.ndarray, np.ndarray]] = {}


# ========================== 基本計算函式 ==============================
def read_pick_csv(csv_path):
    grouped: dict[str, dict[int, float]] = {}
    all_blocks: dict[str, set[int]] = {}
    event_metadata: dict[str, dict[str, object]] = {}

    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        has_onset_column = "Onset" in (reader.fieldnames or [])
        for row in reader:
            try:
                source_text = str(row["Block"]).strip()
                channel = int(row["Channel"])
            except (KeyError, TypeError, ValueError):
                continue
            if not source_text:
                continue

            # 新格式直接用 118-1、118-2 表示同一原始 block 的不同事件。
            # 同時相容前一版曾經加入 Onset 欄位的 CSV，讀取時自動轉成相同編號。
            old_onset = str(row.get("Onset", "")).strip() if has_onset_column else ""
            event_id = (
                f"{source_text}-{old_onset}"
                if old_onset and "-" not in source_text
                else source_text
            )
            base_text, _, subevent_text = event_id.partition("-")
            event_metadata[event_id] = {
                "block": base_text,
                "subevent": int(subevent_text or 1),
            }
            all_blocks.setdefault(event_id, set())
            try:
                pick_time = float(row["AIC_Pick_Time_s"])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(pick_time):
                continue
            all_blocks[event_id].add(channel)
            grouped.setdefault(event_id, {})[channel] = pick_time

    obs_dict = {
        block: picks
        for block, picks in grouped.items()
        if sum(channel in picks for channel in station_ids) >= MIN_CHANNELS
    }
    pick_counts = {block: len(channels) for block, channels in all_blocks.items()}
    incomplete_blocks = {
        block: {
            "pick_count": sum(channel in channels for channel in station_ids),
            "available_channels": sorted(channels),
            "missing_channels": [channel for channel in station_ids if channel not in channels],
        }
        for block, channels in all_blocks.items()
        if sum(channel in channels for channel in station_ids) < MIN_CHANNELS
    }
    return obs_dict, pick_counts, incomplete_blocks, event_metadata


def event_sort_key(event) -> tuple[int, int, str]:
    text = str(event)
    base_text, _, subevent_text = text.partition("-")
    try:
        base = int(base_text)
    except ValueError:
        base = 10**12
    try:
        subevent = int(subevent_text or 1)
    except ValueError:
        subevent = 1
    return base, subevent, text


def compute_res_jac(m, obs, use_station_ids, V):
    x, y, z, t0 = np.asarray(m, dtype=float)
    n = len(use_station_ids)
    res = np.zeros(n, dtype=float)
    J = np.zeros((n, 4), dtype=float)

    for i, sid in enumerate(use_station_ids):
        xi, yi, zi = sensor_positions[sid]
        d = max(float(np.linalg.norm([x - xi, y - yi, z - zi])), 1e-12)
        t_pred = t0 + d / V
        res[i] = obs[sid] - t_pred
        J[i, 0] = -(x - xi) / (V * d)
        J[i, 1] = -(y - yi) / (V * d)
        J[i, 2] = -(z - zi) / (V * d)
        J[i, 3] = -1.0

    return res, J


def xyz_to_cyl(m):
    p = project_to_geometry(np.asarray(m, dtype=float), geometry)
    cx, cy = geometry.cylinder_center_xy
    dx = p[0] - cx
    dy = p[1] - cy
    r = float(np.hypot(dx, dy))
    theta = float(np.arctan2(dy, dx)) if r > 1e-12 else 0.0
    return np.asarray([r, theta, p[2], p[3]], dtype=float)


def cyl_to_xyz(m):
    r, theta, z, t0 = np.asarray(m, dtype=float)
    cx, cy = geometry.cylinder_center_xy
    r = float(np.clip(r, 0.0, geometry.cylinder_radius))
    return np.asarray([cx + r * np.cos(theta), cy + r * np.sin(theta), z, t0], dtype=float)


def compute_cyl_res_jac(m, obs, use_station_ids, V):
    r, theta, _, _ = np.asarray(m, dtype=float)
    xyz_model = cyl_to_xyz(m)
    res, J_xyz = compute_res_jac(xyz_model, obs, use_station_ids, V)
    chain = np.asarray(
        [
            [np.cos(theta), -r * np.sin(theta), 0.0, 0.0],
            [np.sin(theta), r * np.cos(theta), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return res, J_xyz @ chain


# ========================== Grid search ==============================
# 單事件定位最重要的初始值：用很多候選點比較 station 到時差，找誤差最小的位置。
def vec_search_fixedV(bounds, shape, arr, use_station_ids, V):
    cache_key = (
        tuple((round(a, 8), round(b, 8)) for a, b in bounds),
        tuple(shape),
        tuple(use_station_ids),
    )

    if cache_key in grid_cache:
        pts, D = grid_cache[cache_key]
    else:
        (xm, xM), (ym, yM), (zm, zM) = bounds
        X, Y, Z = np.meshgrid(
            np.linspace(xm, xM, shape[0]),
            np.linspace(ym, yM, shape[1]),
            np.linspace(zm, zM, shape[2]),
            indexing="ij",
        )
        pts = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        pts = pts[point_mask_for_geometry(pts, geometry)]
        if len(pts) == 0:
            raise ValueError("Grid search has no points inside the specimen.")

        S = np.asarray([sensor_positions[s] for s in use_station_ids], dtype=float)
        D = np.linalg.norm(pts[:, None, :] - S[None, :, :], axis=2)
        grid_cache[cache_key] = (pts, D)

    errs = np.zeros(len(pts), dtype=float)
    for i, j in combinations(range(len(use_station_ids)), 2):
        dt_obs = arr[j] - arr[i]
        dt_pred = (D[:, j] - D[:, i]) / V
        errs += (dt_pred - dt_obs) ** 2

    idx = int(np.argmin(errs))
    x, y, z = pts[idx]
    t0 = float(np.median(arr - D[idx] / V))
    return float(x), float(y), float(z), t0


def estimate_velocity(x0, y0, z0, t0, arr, use_station_ids):
    Vs = []
    for i, sid in enumerate(use_station_ids):
        xi, yi, zi = sensor_positions[sid]
        di = float(np.linalg.norm([x0 - xi, y0 - yi, z0 - zi]))
        dt = arr[i] - t0
        if dt > 0:
            Vs.append(di / dt)
    return float(np.mean(Vs)) if Vs else float("nan")


def estimate_velocity_residual_norm(x0, y0, z0, t0, arr, use_station_ids, V):
    residuals = []
    for i, sid in enumerate(use_station_ids):
        xi, yi, zi = sensor_positions[sid]
        di = float(np.linalg.norm([x0 - xi, y0 - yi, z0 - zi]))
        dt = arr[i] - t0
        if dt > 0:
            residuals.append((di / dt - V) ** 2)
    return float(np.sqrt(np.mean(residuals))) if residuals else float("inf")


# 把點投影回試體內部。圓柱試體會投影到圓柱內，不會只用外接方盒。
def project_to_specimen_full(pt4):
    return project_to_geometry(np.asarray(pt4, dtype=float), geometry)


def near_specimen_boundary(point):
    p = np.asarray(point, dtype=float)
    if geometry.specimen_shape == "cylinder" and geometry.cylinder_radius is not None:
        cx, cy = geometry.cylinder_center_xy
        r = float(np.hypot(p[0] - cx, p[1] - cy))
        return (
            r >= geometry.cylinder_radius - BOUNDARY_EPS
            or abs(p[2] - zmin) <= BOUNDARY_EPS
            or abs(p[2] - zmax) <= BOUNDARY_EPS
        )

    return any(
        abs(p[0] - xm) <= BOUNDARY_EPS
        or abs(p[0] - xM) <= BOUNDARY_EPS
        or abs(p[1] - ym) <= BOUNDARY_EPS
        or abs(p[1] - yM) <= BOUNDARY_EPS
        or abs(p[2] - zm) <= BOUNDARY_EPS
        or abs(p[2] - zM) <= BOUNDARY_EPS
        for xm, xM, ym, yM, zm, zM in cuboid_bounds
    )


# ========================== HypoDD ==============================
def refine_hypodd(events, obs_dict, g1_res, g1_t0, V):
    ev = [e for e in events if e in g1_res and e in g1_t0]
    N = len(ev)
    if N < 2:
        return {}, {}

    curr_est = {e: np.asarray(g1_res[e], dtype=float).copy() for e in ev}
    curr_t0 = {e: float(g1_t0[e]) for e in ev}

    stages = [
        {"max_dt": 0.005, "max_dist": specimen_diagonal, "min_sta": 3, "alpha": 0.005, "damp": 1e-3},
        {"max_dt": 0.001, "max_dist": specimen_diagonal * 0.5, "min_sta": 4, "alpha": 0.001, "damp": 5e-4},
    ]

    for stage in stages:
        for _ in range(HYPODD_MAX_OUTER_ITER):
            rows = []

            # 兩兩事件比較同一 sensor 的到時差，建立 double-difference 方程。
            for i in range(N):
                ei = ev[i]
                pi = curr_est[ei]
                ti = curr_t0[ei]
                for j in range(i + 1, N):
                    ej = ev[j]
                    pj = curr_est[ej]
                    tj = curr_t0[ej]

                    if np.linalg.norm(pi - pj) > stage["max_dist"]:
                        continue

                    common = [s for s in station_ids if s in obs_dict[ei] and s in obs_dict[ej]]
                    if len(common) < stage["min_sta"]:
                        continue

                    for sid in common:
                        sx, sy, sz = sensor_positions[sid]
                        di = max(float(np.linalg.norm(pi - [sx, sy, sz])), 1e-12)
                        dj = max(float(np.linalg.norm(pj - [sx, sy, sz])), 1e-12)

                        dt_obs = obs_dict[ei][sid] - obs_dict[ej][sid]
                        dt_pred = (ti - tj) + (di - dj) / V
                        residual = dt_obs - dt_pred
                        if abs(residual) > stage["max_dt"]:
                            continue

                        left = [
                            +(pi[0] - sx) / (V * di),
                            +(pi[1] - sy) / (V * di),
                            +(pi[2] - sz) / (V * di),
                            +1.0,
                        ]
                        right = [
                            -(pj[0] - sx) / (V * dj),
                            -(pj[1] - sy) / (V * dj),
                            -(pj[2] - sz) / (V * dj),
                            -1.0,
                        ]

                        cols = list(range(4 * i, 4 * i + 4)) + list(range(4 * j, 4 * j + 4))
                        vals = left + right
                        rows.append((cols, vals, residual))

            data_rows = len(rows)
            if data_rows == 0:
                break

            # 加入零均值約束，避免整群事件一起漂移。
            for axis in range(4):
                cols = [4 * i + axis for i in range(N)]
                vals = [1.0] * N
                rows.append((cols, vals, 0.0))

            A = lil_matrix((len(rows), 4 * N), dtype=float)
            b = np.zeros(len(rows), dtype=float)
            for row, (cols, vals, residual) in enumerate(rows):
                A[row, cols] = vals
                b[row] = residual

            A = A.tocsc()
            x_prev = np.zeros(4 * N, dtype=float)

            # re-weighted LSQR：殘差大的資料權重變小。
            for _ in range(HYPODD_MAX_INNER_ITER):
                residual = A.dot(x_prev) - b
                weights = np.ones(A.shape[0], dtype=float)
                weights[:data_rows] = 1.0 / (1.0 + (np.abs(residual[:data_rows]) / stage["alpha"]) ** 2)
                W = diags(weights)
                x_new = lsqr(W.dot(A), W.dot(b), damp=stage["damp"], iter_lim=1000)[0]

                if np.linalg.norm(x_new - x_prev) < HYPODD_TOL_RW:
                    x_prev = x_new
                    break
                x_prev = x_new

            max_move = 0.0
            for i, e in enumerate(ev):
                delta = x_prev[4 * i : 4 * i + 4].copy()
                step = float(np.linalg.norm(delta[:3]))
                if step > HYPODD_MAX_STEP:
                    delta *= HYPODD_MAX_STEP / step

                before = np.asarray([*curr_est[e], curr_t0[e]], dtype=float)
                after = before + delta

                # 若一步走出試體，就縮短步長直到留在圓柱內。
                if not bool(point_mask_for_geometry(after[:3], geometry)[0]):
                    low, high = 0.0, 1.0
                    for _ in range(30):
                        mid = 0.5 * (low + high)
                        trial = before + delta * mid
                        if bool(point_mask_for_geometry(trial[:3], geometry)[0]):
                            low = mid
                        else:
                            high = mid
                    after = before + delta * low

                after = project_to_specimen_full(after)
                max_move = max(max_move, float(np.linalg.norm(after[:3] - curr_est[e])))
                curr_est[e] = after[:3]
                curr_t0[e] = float(after[3])

            if max_move < HYPODD_TOL_POS:
                break

    return curr_est, curr_t0


# ========================== Run analysis for each CSV ==============================
csv_files = [
    path
    for path in sorted(CSV_DIR.glob(CSV_PATTERN))
    if "_backup" not in path.stem.lower()
]
file_colors = plt.get_cmap("coolwarm")(np.linspace(0, 1, max(len(csv_files), 1)))
results = []

print(f"CSV files: {[path.name for path in csv_files]}")

for idx_file, csv_path in enumerate(csv_files):
    print(f"\nProcessing {csv_path.name}", flush=True)
    color = file_colors[idx_file]

    obs_dict, pick_counts, incomplete_blocks, event_metadata = read_pick_csv(csv_path)
    events = sorted(obs_dict.keys(), key=event_sort_key)
    print(f"usable events = {len(events)}", flush=True)

    g0_res, g0_t0 = {}, {}
    g1_res, g1_t0 = {}, {}
    V_est_dict: dict[str, float] = {}
    quality_failed_blocks: dict[str, dict[str, object]] = {}
    station_ids_by_event: dict[str, list[int]] = {}

    if not events:
        results.append(
            {
                "file": csv_path.name,
                "events": [],
                "all_blocks": sorted(pick_counts.keys(), key=event_sort_key),
                "complete_blocks": [],
                "incomplete_blocks": incomplete_blocks,
                "quality_failed_blocks": {},
                "manual_review_blocks": sorted(incomplete_blocks.keys(), key=event_sort_key),
                "pick_counts": pick_counts,
                "g0_res": {},
                "g0_t0": {},
                "g1_res": {},
                "g1_t0": {},
                "hypodd": {},
                "color": color,
                "velocity_estimates": {},
                "network_velocity": float("nan"),
                "event_metadata": event_metadata,
            }
        )
        continue

    # ------------------------------- Grid search ------------------------------- #
    for i, e in enumerate(events, start=1):
        source = event_metadata.get(e, {"block": e, "subevent": 1})
        print(
            f"  event {i}/{len(events)} -> block {source['block']}-{source['subevent']}",
            flush=True,
        )

        obs = obs_dict[e]
        use_station_ids = [sid for sid in station_ids if sid in obs]
        station_ids_by_event[e] = use_station_ids
        arr = np.asarray([obs[sid] for sid in use_station_ids], dtype=float)

        # 先用小網格反覆縮小範圍，同時用到時計算該事件的波速。
        V_curr = V0
        bounds = list(full_bounds)
        x_c = y_c = z_c = t_c = float("nan")
        dx = (xmax - xmin) / REFINE_FACTOR
        dy = (ymax - ymin) / REFINE_FACTOR
        dz = (zmax - zmin) / REFINE_FACTOR

        for k in range(GRID_MAX_ITER):
            x_c, y_c, z_c, t_c = vec_search_fixedV(bounds, COARSE_SHAPE, arr, use_station_ids, V_curr)
            V_next = estimate_velocity(x_c, y_c, z_c, t_c, arr, use_station_ids)
            if math.isfinite(V_next):
                V_curr = V_next

            dx = (bounds[0][1] - bounds[0][0]) / REFINE_FACTOR
            dy = (bounds[1][1] - bounds[1][0]) / REFINE_FACTOR
            dz = (bounds[2][1] - bounds[2][0]) / REFINE_FACTOR
            if max(dx, dy, dz) < GRID_TOL_DIST:
                break

            bounds = [(x_c - dx, x_c + dx), (y_c - dy, y_c + dy), (z_c - dz, z_c + dz)]

        if not math.isfinite(V_curr):
            continue

        V_rmse = estimate_velocity_residual_norm(x_c, y_c, z_c, t_c, arr, use_station_ids, V_curr)
        if V_curr < MIN_VELOCITY or V_rmse > MAX_VELOCITY_RMSE:
            quality_failed_blocks[e] = {
                "velocity": float(V_curr),
                "velocity_rmse": float(V_rmse),
                "reason": "velocity_below_min" if V_curr < MIN_VELOCITY else "velocity_rmse_above_max",
            }
            if SKIP_BAD_VELOCITY_BLOCKS:
                continue

        # 全域較密 grid search 找大概位置。
        x_c, y_c, z_c, t_c = vec_search_fixedV(full_bounds, GLOBAL_SHAPE, arr, use_station_ids, V_curr)
        V_curr2 = estimate_velocity(x_c, y_c, z_c, t_c, arr, use_station_ids)
        if not math.isfinite(V_curr2):
            V_curr2 = V_curr

        # 再在附近做局部 grid search，作為 TRF 的初始位置 g0。
        bounds_f = [(x_c - dx, x_c + dx), (y_c - dy, y_c + dy), (z_c - dz, z_c + dz)]
        x0, y0, z0, t0 = vec_search_fixedV(bounds_f, LOCAL_SHAPE, arr, use_station_ids, V_curr2)
        p0 = project_to_specimen_full(np.asarray([x0, y0, z0, t0], dtype=float))

        g0_res[e] = p0[:3]
        g0_t0[e] = float(p0[3])
        V_est_dict[e] = float(V_curr2)

    if not g0_res:
        results.append(
            {
                "file": csv_path.name,
                "events": [],
                "all_blocks": sorted(pick_counts.keys(), key=event_sort_key),
                "complete_blocks": events,
                "incomplete_blocks": incomplete_blocks,
                "quality_failed_blocks": quality_failed_blocks,
                "manual_review_blocks": sorted(
                    set(incomplete_blocks.keys()) | set(quality_failed_blocks.keys()),
                    key=event_sort_key,
                ),
                "pick_counts": pick_counts,
                "g0_res": {},
                "g0_t0": {},
                "g1_res": {},
                "g1_t0": {},
                "hypodd": {},
                "color": color,
                "velocity_estimates": V_est_dict,
                "network_velocity": float("nan"),
                "event_metadata": event_metadata,
            }
        )
        continue

    V_all = float(np.median(list(V_est_dict.values())))
    print(f"median velocity = {V_all:.1f} m/s", flush=True)

    # ------------------------------- TRF return results ------------------------------- #
    lm_thr = specimen_diagonal * 2.0
    for e in sorted(g0_res.keys(), key=event_sort_key):
        obs = obs_dict[e]
        use_station_ids = station_ids_by_event[e]

        max_distance = specimen_diagonal
        low_t0 = min(obs.values()) - max_distance / max(V_all, 1.0)
        high_t0 = max(obs.values())
        start = project_to_specimen_full(np.asarray([*g0_res[e], g0_t0[e]], dtype=float))
        start[3] = np.clip(start[3], low_t0, high_t0)

        if geometry.specimen_shape == "cylinder" and geometry.cylinder_radius is not None:
            start_cyl = xyz_to_cyl(start)
            start_cyl[0] = np.clip(start_cyl[0], 0.0, geometry.cylinder_radius)
            start_cyl[2] = np.clip(start_cyl[2], zmin, zmax)
            start_cyl[3] = np.clip(start_cyl[3], low_t0, high_t0)
            low_b = [0.0, -np.inf, zmin, low_t0]
            high_b = [geometry.cylinder_radius, np.inf, zmax, high_t0]

            sol = least_squares(
                fun=lambda m: compute_cyl_res_jac(m, obs, use_station_ids, V_all)[0],
                jac=lambda m: compute_cyl_res_jac(m, obs, use_station_ids, V_all)[1],
                x0=start_cyl,
                method="trf",
                bounds=(low_b, high_b),
                xtol=TRF_XTOL,
                ftol=TRF_FTOL,
                gtol=TRF_GTOL,
                max_nfev=TRF_MAX_NFEV,
                x_scale="jac",
            )
            p1 = project_to_specimen_full(cyl_to_xyz(sol.x))
        else:
            low_b = [xmin, ymin, zmin, low_t0]
            high_b = [xmax, ymax, zmax, high_t0]
            start = np.clip(start, low_b, high_b)

            sol = least_squares(
                fun=lambda m: compute_res_jac(m, obs, use_station_ids, V_all)[0],
                jac=lambda m: compute_res_jac(m, obs, use_station_ids, V_all)[1],
                x0=start,
                method="trf",
                bounds=(low_b, high_b),
                xtol=TRF_XTOL,
                ftol=TRF_FTOL,
                gtol=TRF_GTOL,
                max_nfev=TRF_MAX_NFEV,
                x_scale="jac",
            )
            p1 = project_to_specimen_full(sol.x)

        if sol.success and np.linalg.norm(p1[:3] - g0_res[e]) <= lm_thr:
            g1_res[e] = p1[:3]
            g1_t0[e] = float(p1[3])
        else:
            g1_res[e] = g0_res[e]
            g1_t0[e] = g0_t0[e]

    # ------------------------------- HypoDD return results ------------------------------- #
    pos_dict, t0_dict = refine_hypodd(
        sorted(g1_res.keys(), key=event_sort_key),
        obs_dict,
        g1_res,
        g1_t0,
        V_all,
    )

    hypodd = {}
    hypodd_boundary_fallback_blocks = []
    for e, pos in pos_dict.items():
        p4 = project_to_specimen_full(np.asarray([*pos, t0_dict[e]], dtype=float))

        if (
            FALLBACK_BOUNDARY_HYPODD_TO_TRF
            and e in g1_res
            and near_specimen_boundary(p4[:3])
            and not near_specimen_boundary(g1_res[e])
        ):
            p4 = np.asarray([*g1_res[e], g1_t0[e]], dtype=float)
            hypodd_boundary_fallback_blocks.append(e)

        hypodd[e] = (p4[0], p4[1], p4[2], p4[3])

    results.append(
        {
            "file": csv_path.name,
            "events": sorted(g0_res.keys(), key=event_sort_key),
            "all_blocks": sorted(pick_counts.keys(), key=event_sort_key),
            "complete_blocks": events,
            "incomplete_blocks": incomplete_blocks,
            "quality_failed_blocks": quality_failed_blocks,
            "hypodd_boundary_fallback_blocks": hypodd_boundary_fallback_blocks,
            "manual_review_blocks": sorted(
                set(incomplete_blocks.keys()) | set(quality_failed_blocks.keys()),
                key=event_sort_key,
            ),
            "pick_counts": pick_counts,
            "g0_res": g0_res,
            "g0_t0": g0_t0,
            "g1_res": g1_res,
            "g1_t0": g1_t0,
            "hypodd": hypodd,
            "color": color,
            "velocity_estimates": V_est_dict,
            "network_velocity": V_all,
            "event_metadata": event_metadata,
        }
    )

    print(f"located={len(g0_res)}, hypodd={len(hypodd)}", flush=True)


# ------------------------------- Saving results -------------------------------
written_path = save_results_pickle(OUTPUT_PATH, results, geometry)
print(f"\nalready save: {written_path}")
