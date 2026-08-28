import pickle
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


# ======================== 使用者設定：通常只需要改這幾行 ========================
# interactive_html：輸出可以手動旋轉、拖曳時間軸的 3D HTML。
# static_overview：輸出一張沒有時間軸、顯示所有 AE 點的靜態 HTML。
RUN_MODE = "interactive_html"
METHOD = "hypodd"  

# 只要改這裡：要輸出哪些 results pkl。
RESULTS_FILES = (
    "results_test_1_90.pkl",
    "results_test_2_90.pkl",
)

# 點顏色模式："time" 代表 AE 時間；"depth" 代表從圓柱表面往內部的深度。
POINT_COLOR_MODES = ("time", "depth")
POINT_COLOR_MODE = POINT_COLOR_MODES[0]

# HTML 按鈕用的篩選條件：只隱藏「很貼邊」而且「殘差很大」的 AE 點。
# 深度是從圓柱外表往內算，單位 cm；殘差用 calculate 存在 pkl 裡的 velocity_rmse。
EDGE_FILTER_DEPTH_CM = 0.10
EDGE_FILTER_RESIDUAL_RMSE = 1000.0


# ======================== 固定輸出與介面設定：一般不用改 ========================

# 0 表示每個 AE 時間點都保留；大型 test 可抽樣，避免 GitHub Pages 檔案太大。
MAX_TIME_STEPS_BY_TEST = {
}

# "cdn" keeps HTML smaller for GitHub Pages. Use True only if the file must
# work without internet.
PLOTLY_JS_MODE = "cdn"

# 時間軸最多取幾個時間點。每一格都會更新 AE 點、PCA 平面與兩套裂縫 network。
# 大型資料均勻保留 60 格，維持完整動態效果並避免 HTML 再膨脹到約 1 GB。
MAX_TIME_STEPS = 60

# AE 點與 sensor marker 的顯示大小。
POINT_SIZE = 4
SENSOR_SIZE = 5

# ======================== 介面版面設定 ========================
# VIEW_HEIGHT：Plotly 圖高；VIEW_MAX_WIDTH：整個互動頁最大寬度。
VIEW_HEIGHT = 980
VIEW_MAX_WIDTH = 1560

# 介面字型；優先用 Segoe UI，再用微軟正黑體與 Noto Sans TC。
FONT_FAMILY = "Segoe UI, Microsoft JhengHei, Noto Sans TC, Arial, sans-serif"

# 3D 座標軸額外留白比例；調大可避免放大或旋轉時模型被切到。
SCENE_RANGE_PADDING = 0.45

# ======================== 裂縫與試體顯示開關 ========================
# True 會畫表面裂縫網路；這是用靠近表面的 AE 點做 graph/skeleton，不是 theta band。
SHOW_SURFACE_CRACK_NETWORK = True

# True 會畫內部破裂面；這是用 AE 點雲做 PCA 擬合出的半透明斜平面。
SHOW_INTERNAL_CRACK_PLANE = True

# True 會畫內部裂縫網路；這是用 AE 點的 3D 距離做 graph/skeleton。
SHOW_INTERNAL_CRACK_NETWORK = True

# True 會畫淡淡的透明圓柱外表，讓旋轉時仍看得出試體形狀。
SHOW_SPECIMEN_TRANSLUCENT_SURFACE = True

# 試體線框控制：上下圓環保留；直向線與側邊直線目前都關閉。
SHOW_SPECIMEN_RING_LINES = True
SHOW_SPECIMEN_VERTICAL_LINES = False
SHOW_SPECIMEN_SIDE_VERTICAL_LINES = False

# ======================== 透明圓柱外表設定 ========================
# 透明外表的不透明度；越大圓柱越明顯，但可能遮住 AE 點。
SPECIMEN_SURFACE_OPACITY = 0.12

# 圓柱外表網格解析度；數值越大越圓滑，但 HTML 會稍大。
SPECIMEN_SURFACE_THETA_STEPS = 96
SPECIMEN_SURFACE_Z_STEPS = 24

# ======================== 表面裂縫 network 設定 ========================
# 只取半徑大於 R*0.70 的 AE 點當表面附近點；越大越貼近表面。
SURFACE_CRACK_MIN_RADIUS_RATIO = 0.70

# 表面裂縫線稍微畫在圓柱外側，避免被透明外表蓋住。
SURFACE_CRACK_OUTSET_RATIO = 1.035

# 表面 AE 點少於這個數量就不畫裂縫網路。
SURFACE_CRACK_MIN_POINTS = 6

# 每個 AE 點最多連到幾個最近鄰點，形成初始 graph。
SURFACE_CRACK_K_NEIGHBORS = 3

# 判斷局部密度的距離半徑，單位 m；越大會保留更多稀疏點。
SURFACE_CRACK_DENSITY_RADIUS = 0.035

# 點附近至少要有幾個鄰居才算局部密度足夠。
SURFACE_CRACK_MIN_LOCAL_NEIGHBORS = 2

# 兩個 AE 點之間可連線的最大距離，單位 m；越大裂縫越容易跨空連接。
SURFACE_CRACK_MAX_EDGE_LENGTH = 0.040

# 一個 connected component 至少要有幾個點才保留，避免孤立小雜訊。
SURFACE_CRACK_MIN_COMPONENT_POINTS = 3

# 每條裂縫線段用幾個點插值，讓圓柱表面上的線比較順。
SURFACE_CRACK_EDGE_POINTS = 8

# 在 skeleton 主幹之外，額外保留多少比例的短分支；越大分支越多。
SURFACE_CRACK_EXTRA_BRANCH_RATIO = 0.35

# 單一節點最多連幾條線，避免裂縫網路變成過密蜘蛛網。
SURFACE_CRACK_MAX_NODE_DEGREE = 4

# 表面裂縫線的顏色與粗細。
CRACK_LINE_COLOR = "rgba(16,36,145,0.98)"
CRACK_LINE_WIDTH = 6

# ======================== 內部 PCA 破裂面與 network 設定 ========================
# AE 點數少於這個數量時不擬合內部破裂面，避免少量點造成不穩定平面。
INTERNAL_PLANE_MIN_POINTS = 8

# PCA 平面的半尺寸，單位 m；實際顯示時仍會被裁在圓柱內部。
INTERNAL_PLANE_HALF_SIZE = 0.12

# 內部平面網格解析度；越大平面越細，但檔案也會較大。
INTERNAL_PLANE_GRID_STEPS = 48

# 內部破裂面：全部 AE 點先 PCA，保留離初始平面最近的 95% 點，再重新 PCA 畫單一平面。
INTERNAL_PLANE_COVERAGE = 0.95

# 內部破裂面的透明度與顏色。
INTERNAL_PLANE_OPACITY = 0.34
INTERNAL_PLANE_COLOR = "#ef8a24"

# 內部裂縫 network 的 3D graph 參數；單位都是 m。
INTERNAL_CRACK_MIN_POINTS = 8
INTERNAL_CRACK_K_NEIGHBORS = 4
INTERNAL_CRACK_DENSITY_RADIUS = 0.040
INTERNAL_CRACK_MIN_LOCAL_NEIGHBORS = 2
INTERNAL_CRACK_MAX_EDGE_LENGTH = 0.060
INTERNAL_CRACK_MIN_COMPONENT_POINTS = 3
INTERNAL_CRACK_EDGE_POINTS = 4
INTERNAL_CRACK_EXTRA_BRANCH_RATIO = 0.45
INTERNAL_CRACK_MAX_NODE_DEGREE = 4
INTERNAL_CRACK_LINE_COLOR = "rgba(196,58,36,0.96)"
INTERNAL_CRACK_LINE_WIDTH = 5

SCRIPT_DIR = Path(__file__).resolve().parent


def target_test_name_from_results_file(file_name):
    stem = Path(str(file_name)).stem
    if not stem.startswith("results_"):
        return None
    test_name = stem[len("results_"):]
    return test_name or None


RESULTS_PATHS = [
    SCRIPT_DIR / file_name
    for file_name in RESULTS_FILES
]
TARGET_TESTS = tuple(
    test_name
    for file_name in RESULTS_FILES
    for test_name in [target_test_name_from_results_file(file_name)]
    if test_name is not None
)
OUTPUT_DIR = SCRIPT_DIR / "ae_interactive"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_results(path):
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


results = []
sensor_positions = {}
cuboids = {}
specimen = {}
xmin = xmax = ymin = ymax = zmin = zmax = 0.0
cylinder_radius = None
cylinder_center_xy = (0.0, 0.0)
X_RANGE = Y_RANGE = Z_RANGE = [0.0, 1.0]


def padded_axis_range(low, high, padding=SCENE_RANGE_PADDING):
    span = max(abs(high - low), 1e-9)
    pad = span * padding
    return [low - pad, high + pad]


def apply_results_data(data):
    global results, sensor_positions, cuboids, specimen
    global xmin, xmax, ymin, ymax, zmin, zmax
    global cylinder_radius, cylinder_center_xy, X_RANGE, Y_RANGE, Z_RANGE

    results = data["results"]
    sensor_positions = data["sensor_positions"]
    cuboids = data["cuboids"]
    specimen = data["specimen"]
    xmin, xmax = data["xmin"], data["xmax"]
    ymin, ymax = data["ymin"], data["ymax"]
    zmin, zmax = data["zmin"], data["zmax"]
    cylinder_radius = data.get("cylinder_radius")
    cylinder_center_xy = data.get("cylinder_center_xy", (0.0, 0.0))
    if cylinder_radius is None:
        cylinder_radius = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax))

    X_RANGE = padded_axis_range(xmin, xmax)
    Y_RANGE = padded_axis_range(ymin, ymax)
    Z_RANGE = padded_axis_range(zmin, zmax)


def numbered_points(prefix):
    items = []
    for point_id, position in specimen.items():
        if point_id.startswith(prefix) and point_id[len(prefix):].isdigit():
            items.append((int(point_id[len(prefix):]), np.asarray(position, dtype=float)))
    return [position for _, position in sorted(items)]


def specimen_segments():
    if cuboids:
        for vertices in cuboids.values():
            low, high = vertices[:4], vertices[4:]
            if SHOW_SPECIMEN_RING_LINES:
                for ring in (low, high):
                    for index in range(4):
                        yield specimen[ring[index]], specimen[ring[(index + 1) % 4]]
            if SHOW_SPECIMEN_VERTICAL_LINES:
                for index in range(4):
                    yield specimen[low[index]], specimen[high[index]]
            elif SHOW_SPECIMEN_SIDE_VERTICAL_LINES:
                for index in (0, 2):
                    yield specimen[low[index]], specimen[high[index]]
        return

    top = numbered_points("T")
    bottom = numbered_points("B")
    if top and len(top) == len(bottom):
        if SHOW_SPECIMEN_RING_LINES:
            for ring in (top, bottom):
                for index in range(len(ring)):
                    yield ring[index], ring[(index + 1) % len(ring)]
        if SHOW_SPECIMEN_VERTICAL_LINES:
            stride = max(1, len(top) // 18)
            for index in range(0, len(top), stride):
                yield top[index], bottom[index]
        elif SHOW_SPECIMEN_SIDE_VERTICAL_LINES:
            top_array = np.asarray(top, dtype=float)
            side_indexes = sorted({int(np.argmin(top_array[:, 0])), int(np.argmax(top_array[:, 0]))})
            for index in side_indexes:
                yield top[index], bottom[index]


def wireframe_trace():
    xs, ys, zs = [], [], []
    for p1, p2 in specimen_segments():
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        xs += [p1[0], p2[0], None]
        ys += [p1[1], p2[1], None]
        zs += [p1[2], p2[2], None]
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line=dict(color="rgba(120,120,120,0.75)", width=4),
        name="specimen",
        showlegend=False,
        hoverinfo="skip",
    )


def specimen_surface_trace():
    if not SHOW_SPECIMEN_TRANSLUCENT_SURFACE or cuboids:
        return None

    cx, cy = cylinder_center_xy
    theta = np.linspace(0.0, 2.0 * np.pi, SPECIMEN_SURFACE_THETA_STEPS)
    z_values = np.linspace(float(zmin), float(zmax), SPECIMEN_SURFACE_Z_STEPS)
    theta_grid, z_grid = np.meshgrid(theta, z_values)
    x_grid = cx + float(cylinder_radius) * np.cos(theta_grid)
    y_grid = cy + float(cylinder_radius) * np.sin(theta_grid)
    surface_color = np.zeros_like(x_grid)
    return go.Surface(
        x=x_grid,
        y=y_grid,
        z=z_grid,
        surfacecolor=surface_color,
        colorscale=[[0.0, "#d8dee8"], [1.0, "#d8dee8"]],
        opacity=SPECIMEN_SURFACE_OPACITY,
        showscale=False,
        name="specimen surface",
        showlegend=False,
        hoverinfo="skip",
        lighting=dict(ambient=0.9, diffuse=0.35, roughness=0.95, specular=0.03),
    )


def sensor_trace():
    #sensor_positions應該是一個dictionary，sorted(sensor_positions)會把sensor的編號排序，就會變成ids = [1, 2, 3, 4]
    ids = sorted(sensor_positions)
    positions = np.asarray([sensor_positions[sensor_id] for sensor_id in ids], dtype=float)
    #go.Scatter3d(...)是Plotly的 3D 散點圖圖層。
    return go.Scatter3d(
        #positions[:, 0]取所有感測器的 x 座標。positions[:, 1]取所有感測器的 y 座標。positions[:, 2]取所有感測器的 z 座標。
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        #表示這個圖層同時顯示：markers(感測器點)、text(感測器文字標籤)
        mode="markers+text",
        marker=dict(size=SENSOR_SIZE, color="forestgreen", symbol="diamond"),
        #S1, S2, S3, S4
        text=[f"S{sensor_id}" for sensor_id in ids],
        textposition="top center",
        name="sensors",
        #顯示sensor座標
        hovertemplate="Sensor %{text}<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>",
    )


def event_points(result, method):
    xs, ys, zs, times, blocks = [], [], [], [], []
    for event in result.get("events", []):
        if method == "grid":
            x, y, z = result["g0_res"][event]
            t0 = result["g0_t0"][event]
        elif method == "trf":
            x, y, z = result["g1_res"].get(event, result["g0_res"][event])
            t0 = result["g1_t0"].get(event, result["g0_t0"][event])
        else:
            if event in result["hypodd"]:
                x, y, z, t0 = result["hypodd"][event]
            else:
                x, y, z = result["g1_res"].get(event, result["g0_res"][event])
                t0 = result["g1_t0"].get(event, result["g0_t0"][event])
        xs.append(x)
        ys.append(y)
        zs.append(z)
        times.append(t0)
        blocks.append(event)
    return (
        np.asarray(xs, dtype=float),
        np.asarray(ys, dtype=float),
        np.asarray(zs, dtype=float),
        np.asarray(times, dtype=float),
        np.asarray(blocks, dtype=object),
    )


def selected_frame_times(relative_times, max_time_steps=MAX_TIME_STEPS):
    unique_times = np.sort(np.unique(relative_times[np.isfinite(relative_times)]))
    if max_time_steps and unique_times.size > max_time_steps:
        indexes = np.linspace(0, unique_times.size - 1, max_time_steps).astype(int)
        unique_times = unique_times[indexes]
    return unique_times


def inward_depth_cm(xs, ys):
    cx, cy = cylinder_center_xy
    radial_distance = np.hypot(np.asarray(xs, dtype=float) - cx, np.asarray(ys, dtype=float) - cy)
    return np.clip((float(cylinder_radius) - radial_distance) * 100.0, 0.0, float(cylinder_radius) * 100.0)


def point_color_values(xs, ys, zs, relative_times):
    if POINT_COLOR_MODE == "depth":
        # Depth is measured radially inward from the cylinder surface, not by z-height.
        return inward_depth_cm(xs, ys)
    if POINT_COLOR_MODE == "time":
        return np.asarray(relative_times, dtype=float)
    raise ValueError("POINT_COLOR_MODE must be 'depth' or 'time'")


def point_color_range(color_values):
    if POINT_COLOR_MODE == "depth":
        return 0.0, max(float(cylinder_radius) * 100.0, 1e-12)
    finite = np.asarray(color_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return 0.0, max(float(np.nanmax(finite)), 1e-12) if finite.size else 1e-12


def point_colorbar_title():
    if POINT_COLOR_MODE == "depth":
        return "Depth into specimen (cm)"
    return "AE time (s)"


def point_color_filename_suffix():
    return f"_{POINT_COLOR_MODE}"


def event_residual_rmse_values(result, blocks):
    quality_failed = result.get("quality_failed_blocks", {})
    residuals = []
    for block in blocks:
        info = quality_failed.get(str(block), {})
        residual = 0.0
        if isinstance(info, dict):
            try:
                residual = float(info.get("velocity_rmse", 0.0) or 0.0)
            except (TypeError, ValueError):
                residual = 0.0
        residuals.append(residual if np.isfinite(residual) else 0.0)
    return np.asarray(residuals, dtype=float)


def edge_high_residual_mask(result, xs, ys, blocks):
    depths = inward_depth_cm(xs, ys)
    residuals = event_residual_rmse_values(result, blocks)
    mask = (depths <= EDGE_FILTER_DEPTH_CM) & (residuals > EDGE_FILTER_RESIDUAL_RMSE)
    return mask, depths, residuals


def set_trace_visibility(trace, visible):
    trace.visible = visible
    return trace


def ae_trace(
    xs,
    ys,
    zs,
    relative_times,
    blocks,
    visible_mask,
    color_values,
    color_min,
    color_max,
    depth_values,
    residual_values,
    edge_residual_mask,
    name="AE points",
    showlegend=True,
):
    hidden_reason = np.where(edge_residual_mask, "edge + high residual", "")
    custom = np.column_stack([
        blocks[visible_mask],
        relative_times[visible_mask],
        depth_values[visible_mask],
        residual_values[visible_mask],
        hidden_reason[visible_mask],
    ])
    return go.Scatter3d(
        x=xs[visible_mask],
        y=ys[visible_mask],
        z=zs[visible_mask],
        mode="markers",
        marker=dict(
            size=POINT_SIZE,
            color=color_values[visible_mask],
            colorscale="Viridis",
            cmin=color_min,
            cmax=color_max,
            opacity=0.9,
            colorbar=dict(
                title=dict(text=point_colorbar_title(), side="right", font=dict(size=14)),
                x=0.84,
                y=0.45,
                len=0.58,
                thickness=18,
                outlinewidth=0,
                tickfont=dict(size=12),
            ),
        ),
        customdata=custom,
        name=name,
        showlegend=showlegend,
        hovertemplate=(
            "Block %{customdata[0]}<br>"
            "AE time=%{customdata[1]:.6f} s<br>"
            "Depth into specimen=%{customdata[2]:.2f} cm<br>"
            "Velocity RMSE=%{customdata[3]:.1f}<br>"
            "Filter=%{customdata[4]}<br>"
            "x=%{x:.5f}<br>y=%{y:.5f}<br>z=%{z:.5f}<extra></extra>"
        ),
    )


def empty_surface_crack_network_trace():
    return go.Scatter3d(
        x=[],
        y=[],
        z=[],
        mode="lines",
        line=dict(color=CRACK_LINE_COLOR, width=CRACK_LINE_WIDTH),
        name="surface crack network",
        hoverinfo="skip",
        showlegend=SHOW_SURFACE_CRACK_NETWORK,
    )


def fit_crack_plane(points):
    center = points.mean(axis=0)
    shifted = points - center
    _, _, vh = np.linalg.svd(shifted, full_matrices=False)
    normal = vh[-1] / max(np.linalg.norm(vh[-1]), 1e-12)
    return center, normal, vh[0], vh[1]


def point_plane_signed_distances(points, center, normal):
    return (points - center) @ normal


def pca95_fit_crack_plane(points):
    center0, normal0, _, _ = fit_crack_plane(points)
    distance0 = np.abs(point_plane_signed_distances(points, center0, normal0))
    keep_count = int(np.ceil(points.shape[0] * INTERNAL_PLANE_COVERAGE))
    keep_count = max(INTERNAL_PLANE_MIN_POINTS, min(points.shape[0], keep_count))
    keep_indexes = np.argsort(distance0)[:keep_count]
    fit_points = points[keep_indexes]
    center, normal, u, v = fit_crack_plane(fit_points)
    cutoff = float(np.max(distance0[keep_indexes])) if keep_indexes.size else 0.0
    return center, normal, u, v, points.shape[0], fit_points.shape[0], cutoff


def clipped_plane_grid(center, normal, u, v, offset):
    s = np.linspace(-INTERNAL_PLANE_HALF_SIZE, INTERNAL_PLANE_HALF_SIZE, INTERNAL_PLANE_GRID_STEPS)
    t = np.linspace(-INTERNAL_PLANE_HALF_SIZE, INTERNAL_PLANE_HALF_SIZE, INTERNAL_PLANE_GRID_STEPS)
    s_grid, t_grid = np.meshgrid(s, t)
    plane = center + offset * normal + s_grid[..., None] * u + t_grid[..., None] * v
    x_grid = plane[..., 0]
    y_grid = plane[..., 1]
    z_grid = plane[..., 2]

    cx, cy = cylinder_center_xy
    inside = (
        ((x_grid - cx) ** 2 + (y_grid - cy) ** 2 <= float(cylinder_radius) ** 2)
        & (z_grid >= float(zmin))
        & (z_grid <= float(zmax))
    )
    return (
        np.where(inside, x_grid, np.nan),
        np.where(inside, y_grid, np.nan),
        np.where(inside, z_grid, np.nan),
    )


def empty_internal_crack_plane_trace():
    empty = np.full((2, 2), np.nan)
    return go.Surface(
        x=empty,
        y=empty,
        z=empty,
        surfacecolor=np.zeros_like(empty),
        colorscale=[[0.0, INTERNAL_PLANE_COLOR], [1.0, INTERNAL_PLANE_COLOR]],
        opacity=INTERNAL_PLANE_OPACITY,
        showscale=False,
        name="internal 95% PCA plane",
        showlegend=SHOW_INTERNAL_CRACK_PLANE,
        hoverinfo="skip",
    )


def internal_crack_plane_trace(xs, ys, zs, visible_mask):
    if not SHOW_INTERNAL_CRACK_PLANE:
        return empty_internal_crack_plane_trace()

    visible = np.asarray(visible_mask, dtype=bool)
    if np.count_nonzero(visible) < INTERNAL_PLANE_MIN_POINTS:
        return empty_internal_crack_plane_trace()

    points = np.column_stack([xs[visible], ys[visible], zs[visible]])
    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]
    if points.shape[0] < INTERNAL_PLANE_MIN_POINTS:
        return empty_internal_crack_plane_trace()

    center, normal, u, v, total_count, kept_count, cutoff = pca95_fit_crack_plane(points)
    x_grid, y_grid, z_grid = clipped_plane_grid(center, normal, u, v, 0.0)
    if not np.any(np.isfinite(x_grid)):
        return empty_internal_crack_plane_trace()

    surface_color = np.zeros_like(x_grid)
    return go.Surface(
        x=x_grid,
        y=y_grid,
        z=z_grid,
        surfacecolor=surface_color,
        colorscale=[[0.0, INTERNAL_PLANE_COLOR], [1.0, INTERNAL_PLANE_COLOR]],
        opacity=INTERNAL_PLANE_OPACITY,
        showscale=False,
        name="internal 95% PCA plane",
        showlegend=SHOW_INTERNAL_CRACK_PLANE,
        customdata=np.full(x_grid.shape, cutoff * 1000.0),
        hovertemplate=(
            "internal PCA crack plane<br>"
            f"nearest points kept={INTERNAL_PLANE_COVERAGE:.0%}<br>"
            f"points={total_count}, used={kept_count}<br>"
            "initial distance cutoff=%{customdata:.2f} mm<extra></extra>"
        ),
        lighting=dict(ambient=0.82, diffuse=0.45, roughness=0.85, specular=0.05),
    )


def wrapped_angle_delta(theta_to, theta_from):
    return np.arctan2(np.sin(theta_to - theta_from), np.cos(theta_to - theta_from))


def surface_distance_matrix(theta, z_values):
    delta_theta = wrapped_angle_delta(theta[:, None], theta[None, :])
    arc_distance = float(cylinder_radius) * delta_theta
    z_distance = z_values[:, None] - z_values[None, :]
    distance = np.hypot(arc_distance, z_distance)
    np.fill_diagonal(distance, np.inf)
    return distance


def connected_components(node_count, edges):
    adjacency = [[] for _ in range(node_count)]
    for i, j in edges:
        adjacency[i].append(j)
        adjacency[j].append(i)

    components = []
    seen = np.zeros(node_count, dtype=bool)
    for start in range(node_count):
        if seen[start] or not adjacency[start]:
            continue
        stack = [start]
        seen[start] = True
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(neighbor)
        components.append(component)
    return components


def find_root(parent, node):
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


def skeletonize_surface_edges(point_count, candidate_edges, distance, local_neighbors):
    skeleton_edges = set()
    for component in connected_components(point_count, candidate_edges):
        if len(component) < SURFACE_CRACK_MIN_COMPONENT_POINTS:
            continue

        component_nodes = set(component)
        component_edges = sorted(
            [
                edge for edge in candidate_edges
                if edge[0] in component_nodes and edge[1] in component_nodes
            ],
            key=lambda edge: distance[edge],
        )

        parent = {node: node for node in component}
        degree = {node: 0 for node in component}
        tree_edges = set()
        for source, target in component_edges:
            source_root = find_root(parent, source)
            target_root = find_root(parent, target)
            if source_root == target_root:
                continue
            parent[target_root] = source_root
            edge = tuple(sorted((source, target)))
            tree_edges.add(edge)
            degree[source] += 1
            degree[target] += 1

        extra_limit = max(1, int(round(len(component) * SURFACE_CRACK_EXTRA_BRANCH_RATIO)))
        extra_count = 0
        for source, target in component_edges:
            edge = tuple(sorted((source, target)))
            if edge in tree_edges:
                continue
            if distance[edge] > SURFACE_CRACK_DENSITY_RADIUS:
                continue
            if degree[source] >= SURFACE_CRACK_MAX_NODE_DEGREE or degree[target] >= SURFACE_CRACK_MAX_NODE_DEGREE:
                continue
            if min(local_neighbors[source], local_neighbors[target]) < SURFACE_CRACK_MIN_LOCAL_NEIGHBORS:
                continue
            tree_edges.add(edge)
            degree[source] += 1
            degree[target] += 1
            extra_count += 1
            if extra_count >= extra_limit:
                break

        skeleton_edges.update(tree_edges)
    return skeleton_edges


def append_surface_edge(line_xs, line_ys, line_zs, theta_a, z_a, theta_b, z_b):
    cx, cy = cylinder_center_xy
    radius = float(cylinder_radius) * SURFACE_CRACK_OUTSET_RATIO
    progress = np.linspace(0.0, 1.0, SURFACE_CRACK_EDGE_POINTS)
    theta_line = theta_a + wrapped_angle_delta(theta_b, theta_a) * progress
    z_line = z_a + (z_b - z_a) * progress

    line_xs.extend((cx + radius * np.cos(theta_line)).tolist())
    line_ys.extend((cy + radius * np.sin(theta_line)).tolist())
    line_zs.extend(z_line.tolist())
    line_xs.append(None)
    line_ys.append(None)
    line_zs.append(None)


def build_surface_crack_network(xs, ys, zs, visible_mask):
    if not SHOW_SURFACE_CRACK_NETWORK or cylinder_radius is None:
        return [], [], []

    visible = np.asarray(visible_mask, dtype=bool)
    if not np.any(visible):
        return [], [], []

    cx, cy = cylinder_center_xy
    vx = xs[visible] - cx
    vy = ys[visible] - cy
    vz = zs[visible]
    radial = np.hypot(vx, vy)
    #抓靠近表面的AE點
    near_surface = radial >= float(cylinder_radius) * SURFACE_CRACK_MIN_RADIUS_RATIO
    if np.count_nonzero(near_surface) < SURFACE_CRACK_MIN_POINTS:
        return [], [], []

    theta = np.arctan2(vy[near_surface], vx[near_surface])
    z_surface = vz[near_surface]
    finite = np.isfinite(theta) & np.isfinite(z_surface)
    theta = theta[finite]
    z_surface = z_surface[finite]
    point_count = theta.size
    if point_count < SURFACE_CRACK_MIN_POINTS:
        return [], [], []

    #先算每個點彼此距離
    distance = surface_distance_matrix(theta, z_surface)
    #對每個AE點，數一數半徑 SURFACE_CRACK_DENSITY_RADIUS 範圍內有幾個點。
    local_neighbors = np.count_nonzero(distance <= SURFACE_CRACK_DENSITY_RADIUS, axis=1)
    #如果某個AE點附近SURFACE_CRACK_DENSITY_RADIUS內，有SURFACE_CRACK_MIN_LOCAL_NEIGHBORS個AE點，就算是密集區
    active = local_neighbors >= SURFACE_CRACK_MIN_LOCAL_NEIGHBORS
    if np.count_nonzero(active) < SURFACE_CRACK_MIN_POINTS:
        active = local_neighbors >= 1
    if np.count_nonzero(active) < SURFACE_CRACK_MIN_POINTS:
        return [], [], []

    candidate_edges = set()
    for source in np.where(active)[0]:
        added = 0
        for target in np.argsort(distance[source]):
            if not np.isfinite(distance[source, target]) or distance[source, target] > SURFACE_CRACK_MAX_EDGE_LENGTH:
                break
            if not active[target]:
                continue
            edge = tuple(sorted((int(source), int(target))))
            candidate_edges.add(edge)
            added += 1
            if added >= SURFACE_CRACK_K_NEIGHBORS:
                break

    if not candidate_edges:
        return [], [], []

    edges = skeletonize_surface_edges(point_count, candidate_edges, distance, local_neighbors)
    if not edges:
        return [], [], []

    line_xs, line_ys, line_zs = [], [], []
    for source, target in sorted(edges):
        append_surface_edge(
            line_xs,
            line_ys,
            line_zs,
            float(theta[source]),
            float(z_surface[source]),
            float(theta[target]),
            float(z_surface[target]),
        )
    return line_xs, line_ys, line_zs


def surface_crack_network_trace(xs, ys, zs, visible_mask):
    line_xs, line_ys, line_zs = build_surface_crack_network(xs, ys, zs, visible_mask)
    if not line_xs:
        return empty_surface_crack_network_trace()
    return go.Scatter3d(
        x=line_xs,
        y=line_ys,
        z=line_zs,
        mode="lines",
        line=dict(color=CRACK_LINE_COLOR, width=CRACK_LINE_WIDTH),
        name="surface crack network",
        hovertemplate="surface AE crack network<extra></extra>",
    )


def empty_internal_crack_network_trace():
    return go.Scatter3d(
        x=[],
        y=[],
        z=[],
        mode="lines",
        line=dict(color=INTERNAL_CRACK_LINE_COLOR, width=INTERNAL_CRACK_LINE_WIDTH),
        name="internal crack network",
        hoverinfo="skip",
        showlegend=SHOW_INTERNAL_CRACK_NETWORK,
    )


def skeletonize_internal_edges(point_count, candidate_edges, distance, local_neighbors):
    skeleton_edges = set()
    for component in connected_components(point_count, candidate_edges):
        if len(component) < INTERNAL_CRACK_MIN_COMPONENT_POINTS:
            continue

        component_nodes = set(component)
        component_edges = sorted(
            [
                edge for edge in candidate_edges
                if edge[0] in component_nodes and edge[1] in component_nodes
            ],
            key=lambda edge: distance[edge],
        )

        parent = {node: node for node in component}
        degree = {node: 0 for node in component}
        tree_edges = set()
        for source, target in component_edges:
            source_root = find_root(parent, source)
            target_root = find_root(parent, target)
            if source_root == target_root:
                continue
            parent[target_root] = source_root
            edge = tuple(sorted((source, target)))
            tree_edges.add(edge)
            degree[source] += 1
            degree[target] += 1

        extra_limit = max(1, int(round(len(component) * INTERNAL_CRACK_EXTRA_BRANCH_RATIO)))
        extra_count = 0
        for source, target in component_edges:
            edge = tuple(sorted((source, target)))
            if edge in tree_edges:
                continue
            if distance[edge] > INTERNAL_CRACK_DENSITY_RADIUS:
                continue
            if degree[source] >= INTERNAL_CRACK_MAX_NODE_DEGREE or degree[target] >= INTERNAL_CRACK_MAX_NODE_DEGREE:
                continue
            if min(local_neighbors[source], local_neighbors[target]) < INTERNAL_CRACK_MIN_LOCAL_NEIGHBORS:
                continue
            tree_edges.add(edge)
            degree[source] += 1
            degree[target] += 1
            extra_count += 1
            if extra_count >= extra_limit:
                break

        skeleton_edges.update(tree_edges)
    return skeleton_edges


def append_internal_edge(line_xs, line_ys, line_zs, point_a, point_b):
    progress = np.linspace(0.0, 1.0, INTERNAL_CRACK_EDGE_POINTS)
    line = point_a + (point_b - point_a) * progress[:, None]
    line_xs.extend(line[:, 0].tolist())
    line_ys.extend(line[:, 1].tolist())
    line_zs.extend(line[:, 2].tolist())
    line_xs.append(None)
    line_ys.append(None)
    line_zs.append(None)


def build_internal_crack_network(xs, ys, zs, visible_mask):
    if not SHOW_INTERNAL_CRACK_NETWORK:
        return [], [], []

    visible = np.asarray(visible_mask, dtype=bool)
    if np.count_nonzero(visible) < INTERNAL_CRACK_MIN_POINTS:
        return [], [], []

    points = np.column_stack([xs[visible], ys[visible], zs[visible]])
    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]
    point_count = points.shape[0]
    if point_count < INTERNAL_CRACK_MIN_POINTS:
        return [], [], []

    distance = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(distance, np.inf)
    local_neighbors = np.count_nonzero(distance <= INTERNAL_CRACK_DENSITY_RADIUS, axis=1)
    active = local_neighbors >= INTERNAL_CRACK_MIN_LOCAL_NEIGHBORS
    if np.count_nonzero(active) < INTERNAL_CRACK_MIN_POINTS:
        active = local_neighbors >= 1
    if np.count_nonzero(active) < INTERNAL_CRACK_MIN_POINTS:
        return [], [], []

    candidate_edges = set()
    for source in np.where(active)[0]:
        added = 0
        for target in np.argsort(distance[source]):
            if not np.isfinite(distance[source, target]) or distance[source, target] > INTERNAL_CRACK_MAX_EDGE_LENGTH:
                break
            if not active[target]:
                continue
            edge = tuple(sorted((int(source), int(target))))
            candidate_edges.add(edge)
            added += 1
            if added >= INTERNAL_CRACK_K_NEIGHBORS:
                break

    if not candidate_edges:
        return [], [], []

    edges = skeletonize_internal_edges(point_count, candidate_edges, distance, local_neighbors)
    if not edges:
        return [], [], []

    line_xs, line_ys, line_zs = [], [], []
    for source, target in sorted(edges):
        append_internal_edge(line_xs, line_ys, line_zs, points[source], points[target])
    return line_xs, line_ys, line_zs


def internal_crack_network_trace(xs, ys, zs, visible_mask):
    line_xs, line_ys, line_zs = build_internal_crack_network(xs, ys, zs, visible_mask)
    if not line_xs:
        return empty_internal_crack_network_trace()
    return go.Scatter3d(
        x=line_xs,
        y=line_ys,
        z=line_zs,
        mode="lines",
        line=dict(color=INTERNAL_CRACK_LINE_COLOR, width=INTERNAL_CRACK_LINE_WIDTH),
        name="internal crack network",
        hovertemplate="internal AE crack network<extra></extra>",
        showlegend=SHOW_INTERNAL_CRACK_NETWORK,
    )


def scene_layout(title):
    return dict(
        autosize=True,
        title=dict(
            text=title,
            x=0.5,
            y=0.97,
            xanchor="center",
            yanchor="top",
            font=dict(family=FONT_FAMILY, size=22, color="#1f3556"),
        ),
        font=dict(family=FONT_FAMILY, size=13, color="#1f3556"),
        height=VIEW_HEIGHT,
        margin=dict(l=60, r=180, t=90, b=150),
        paper_bgcolor="#f8fafc",
        plot_bgcolor="#f8fafc",
        uirevision="keep-camera",
        legend=dict(
            x=0.78,
            y=0.96,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="rgba(31,53,86,0.18)",
            borderwidth=1,
            font=dict(size=13),
            itemwidth=45,
        ),
        scene=dict(
            domain=dict(x=[0.06, 0.78], y=[0.12, 0.96]),
            xaxis=dict(
                title=dict(text="X (m)", font=dict(size=14)),
                range=X_RANGE,
                showbackground=False,
                gridcolor="rgba(31,53,86,0.12)",
                zerolinecolor="rgba(31,53,86,0.20)",
            ),
            yaxis=dict(
                title=dict(text="Y (m)", font=dict(size=14)),
                range=[Y_RANGE[1], Y_RANGE[0]],
                showbackground=False,
                gridcolor="rgba(31,53,86,0.12)",
                zerolinecolor="rgba(31,53,86,0.20)",
            ),
            zaxis=dict(
                title=dict(text="Z (m)", font=dict(size=14)),
                range=Z_RANGE,
                showbackground=False,
                gridcolor="rgba(31,53,86,0.12)",
                zerolinecolor="rgba(31,53,86,0.20)",
            ),
            aspectmode="data",
            camera=dict(eye=dict(x=1.65, y=1.65, z=1.15)),
            uirevision="keep-camera",
        ),
    )

#fig：Plotly 圖形物件，也就是你做好的 3D 圖。
#output_path：輸出的HTML檔案路徑。
def write_centered_html(fig, output_path):
    plot_html = fig.to_html(
        #不產生完整 HTML，只產生圖的部分
        full_html=False,
        include_plotlyjs=PLOTLY_JS_MODE,
        #responsive:視窗大小改變時，圖會不會調整、scrollZoom:滑鼠滾輪是否可以縮放3D圖、modeBarButtonsToRemove:移除某些工具按鈕
        config={
            "responsive": True,
            "displaylogo": False,
            "scrollZoom": True,
            "modeBarButtonsToRemove": ["toImage", "sendDataToCloud"],
        },
        default_width="100%",
        default_height=f"{VIEW_HEIGHT}px",
    )
# 是在自己包一個完整 HTML 頁面，把剛剛的 Plotly 圖塞進去。意思是Plotly只先產生「圖表本身的 HTML 片段」，但程式後面又自己寫了一個完整的網頁外殼，然後把圖表放進那個網頁裡。   
#margin: 0;瀏覽器預設會讓網頁邊緣有一點空白。
#min-height: 100%;這樣背景色會鋪滿整個頁面，不會只包住內容高度。
#font-family:''；這是在設定整個網頁的字體。
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>AE 3D Viewer</title>
  <style>
    html, body {{
      margin: 0;
      min-height: 100%;
      background: #f8fafc;
      font-family: {FONT_FAMILY};
    }}
    .viewer-shell {{
      width: min(100vw, {VIEW_MAX_WIDTH}px);
      margin: 0 auto;
      background: #f8fafc;
    }}
  </style>
</head>
<body>
  <main class="viewer-shell">
    {plot_html}
  </main>
  <script>
    (function () {{
      const graph = document.querySelector(".plotly-graph-div");
      if (!graph || !window.Plotly) return;

      let rotating = false;
      let rotateTimer = null;
      let filteredMode = false;
      const rotateStep = -0.035;
      const rotateDelayMs = 45;

      function cloneCamera(camera) {{
        return camera ? JSON.parse(JSON.stringify(camera)) : {{}};
      }}

      function currentCamera() {{
        const layoutCamera = graph.layout && graph.layout.scene ? graph.layout.scene.camera : null;
        const fullCamera = graph._fullLayout && graph._fullLayout.scene ? graph._fullLayout.scene.camera : null;
        return cloneCamera(layoutCamera || fullCamera || {{ eye: {{ x: 1.65, y: 1.65, z: 1.15 }} }});
      }}

      function rotateOnce() {{
        if (!rotating) return;
        const camera = currentCamera();
        const eye = camera.eye || {{ x: 1.65, y: 1.65, z: 1.15 }};
        const x = Number.isFinite(eye.x) ? eye.x : 1.65;
        const y = Number.isFinite(eye.y) ? eye.y : 1.65;
        const z = Number.isFinite(eye.z) ? eye.z : 1.15;
        const radius = Math.hypot(x, y) || 2.33;
        const angle = Math.atan2(y, x) + rotateStep;
        camera.eye = {{
          x: radius * Math.cos(angle),
          y: radius * Math.sin(angle),
          z: z,
        }};
        Plotly.relayout(graph, {{ "scene.camera": camera }})
          .then(function () {{
            if (rotating) rotateTimer = window.setTimeout(rotateOnce, rotateDelayMs);
          }});
      }}

      function setRotateButtonActive(active) {{
        const labels = graph.querySelectorAll(".updatemenu-item text");
        labels.forEach(function (label) {{
          if ((label.textContent || "").trim() === "⟳") {{
            label.style.fontWeight = active ? "700" : "";
            label.style.fill = active ? "#0f766e" : "";
          }}
        }});
      }}

      function toggleRotation() {{
        rotating = !rotating;
        if (rotateTimer) {{
          window.clearTimeout(rotateTimer);
          rotateTimer = null;
        }}
        setRotateButtonActive(rotating);
        if (rotating) rotateOnce();
      }}

      function stopRotation() {{
        rotating = false;
        if (rotateTimer) window.clearTimeout(rotateTimer);
        rotateTimer = null;
        setRotateButtonActive(false);
      }}

      function viewerMeta() {{
        return (graph.layout && graph.layout.meta) ||
          (graph._fullLayout && graph._fullLayout.meta) ||
          {{}};
      }}

      function traceVisible(index) {{
        const trace = graph.data && graph.data[index] ? graph.data[index] : null;
        if (!trace || trace.visible === undefined) return true;
        return trace.visible;
      }}

      function switchEdgeFilter(showFiltered) {{
        if (showFiltered === filteredMode) return;
        const meta = viewerMeta();
        const allIndexes = (meta.all_dynamic_indices || []).map(Number);
        const filteredIndexes = (meta.filtered_dynamic_indices || []).map(Number);
        if (!allIndexes.length || allIndexes.length !== filteredIndexes.length) return;

        const sourceIndexes = filteredMode ? filteredIndexes : allIndexes;
        const copiedVisibility = sourceIndexes.map(traceVisible);
        const targetIndexes = allIndexes.concat(filteredIndexes);
        const targetVisibility = [];

        for (let index = 0; index < allIndexes.length; index += 1) {{
          targetVisibility.push(showFiltered ? false : copiedVisibility[index]);
        }}
        for (let index = 0; index < filteredIndexes.length; index += 1) {{
          targetVisibility.push(showFiltered ? copiedVisibility[index] : false);
        }}

        filteredMode = showFiltered;
        Plotly.restyle(graph, {{ visible: targetVisibility }}, targetIndexes);
      }}

      graph.on("plotly_buttonclicked", function (eventData) {{
        const label = eventData && eventData.button ? eventData.button.label : "";
        if (label === "⟳") {{
          toggleRotation();
        }} else if (label === "▶" || label === "■") {{
          stopRotation();
        }} else if (label === "全部") {{
          switchEdgeFilter(false);
        }} else if (label.indexOf("隱藏") === 0) {{
          switchEdgeFilter(true);
        }}
      }});
    }})();
  </script>
</body>
</html>
"""
    Path(output_path).write_text(html, encoding="utf-8")

#專門產生Plotly標題格式。
def title_dict(text):
    return dict(
        text=text,
        x=0.5,
        y=0.97,
        xanchor="center",
        yanchor="top",
        font=dict(family=FONT_FAMILY, size=22, color="#000000"),
    )

#這個函式是在控制「時間滑桿」上要不要顯示文字。
def slider_label(index, current_time, total_steps):
    label_indexes = set(np.linspace(0, total_steps - 1, 4, dtype=int).tolist())
    return f"{current_time:.2f}" if index in label_indexes or index == 0 or index == total_steps - 1 else  ""

#根據AE定位結果result，產生一個可以播放的 3D AE Viewer HTML。
def write_interactive_html(result, method=METHOD):
    xs, ys, zs, t0s, blocks = event_points(result, method)
    #如果某個AE點的座標或時間不是有限數值，就不要畫。
    finite = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs) & np.isfinite(t0s)
    xs, ys, zs, t0s, blocks = xs[finite], ys[finite], zs[finite], t0s[finite], blocks[finite]
    #如果篩完之後沒有任何有效AE點，就跳過這個檔案。
    if t0s.size == 0:
        print(f"Skip {result.get('file', '<unknown>')}: no finite AE points")
        return None
#把時間轉成相對時間
    relative_times = t0s - float(np.nanmin(t0s))
#把AE事件按照發生時間排序    
    order = np.argsort(relative_times)
    xs, ys, zs = xs[order], ys[order], zs[order]
    relative_times, blocks = relative_times[order], blocks[order]
    #frame times顯示「到目前時間為止已經發生的所有AE點」。所以是累積的
    frame_times = selected_frame_times(relative_times, max_time_steps_for_result(result))
    color_values = point_color_values(xs, ys, zs, relative_times)
    color_min, color_max = point_color_range(color_values)
    edge_residual_mask, depth_values, residual_values = edge_high_residual_mask(result, xs, ys, blocks)
    keep_filtered_mask = ~edge_residual_mask
    all_data_mask = np.ones_like(relative_times, dtype=bool)
    total_edge_residual = int(np.count_nonzero(edge_residual_mask))

    def dynamic_traces(time_mask, data_mask, ae_name):
        visible_mask = np.asarray(time_mask, dtype=bool) & np.asarray(data_mask, dtype=bool)
        return [
            internal_crack_plane_trace(xs, ys, zs, visible_mask),
            internal_crack_network_trace(xs, ys, zs, visible_mask),
            ae_trace(
                xs,
                ys,
                zs,
                relative_times,
                blocks,
                visible_mask,
                color_values,
                color_min,
                color_max,
                depth_values,
                residual_values,
                edge_residual_mask,
                name=ae_name,
            ),
            surface_crack_network_trace(xs, ys, zs, visible_mask),
        ]

    # 建立初始畫面；裂縫幾何和 AE 點都會隨時間 frame 更新。
    start_mask = relative_times <= frame_times[0] + 1e-12
    traces = []
    surface_trace = specimen_surface_trace()
    if surface_trace is not None:
        traces.append(surface_trace)
    traces.extend([wireframe_trace(), sensor_trace()])

    all_dynamic_indices = []
    for trace in dynamic_traces(start_mask, all_data_mask, "AE points"):
        all_dynamic_indices.append(len(traces))
        traces.append(set_trace_visibility(trace, True))

    filtered_dynamic_indices = []
    for trace in dynamic_traces(start_mask, keep_filtered_mask, "AE points (filtered)"):
        filtered_dynamic_indices.append(len(traces))
        traces.append(set_trace_visibility(trace, False))

    #這個迴圈會針對每一個時間點建立一個動畫frame。
    frames = []
    steps = []
    for index, current_time in enumerate(frame_times):
        mask = relative_times <= current_time + 1e-12
        frame_title = (
            f"{result['file']} · {method.upper()} AE Viewer | "
            f"t={current_time:.3f} s | points={int(np.count_nonzero(mask))}/{len(xs)} | "
            f"edge+high residual={int(np.count_nonzero(mask & edge_residual_mask))}/{total_edge_residual}"
        )
        #這裡是建立一個Plotly動畫frame，會更新的圖層。
        frames.append(
            go.Frame(
                name=str(index),
                data=(
                    dynamic_traces(mask, all_data_mask, "AE points")
                    + dynamic_traces(mask, keep_filtered_mask, "AE points (filtered)")
                ),
                traces=all_dynamic_indices + filtered_dynamic_indices,
                layout=dict(
                    title=title_dict(frame_title)
                ),
            )
        )
        #建立時間滑桿steps
        steps.append(
            dict(
                method="animate",
                label=slider_label(index, current_time, len(frame_times)),
                args=[
                    [str(index)],
                    dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0)),
                ],
            )
        )

    title = (
        f"{result['file']} · {method.upper()} AE Viewer"
    )
    #這行把初始圖層traces和動畫frames合在一起，變成完整的互動式圖。
    fig = go.Figure(data=traces, frames=frames)
    #設定3D場景，ex.標題、軸範圍、相機角度
    fig.update_layout(**scene_layout(title))
    fig.update_layout(
        meta=dict(
            all_dynamic_indices=all_dynamic_indices,
            filtered_dynamic_indices=filtered_dynamic_indices,
        )
    )
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.20,
                y=0.015,
                xanchor="left",
                yanchor="bottom",
                #按鈕內距，r=10右邊留白10、t=4上方留白4、b=4下方留白4
                pad=dict(r=10, t=4, b=4),
                bgcolor="rgba(255,255,255,0.88)",
                bordercolor="rgba(31,53,86,0.22)",
                borderwidth=1,
                font=dict(size=13, color="#1f3556"),
                buttons=[
                    dict(
                        label="▶",
                        method="animate",
                        #None意思是撥放所有frames，fromcurrent:從目前所在的 frame 開始播放，不會從頭開始。
                        args=[
                            None,
                            dict(
                                fromcurrent=True,
                                mode="immediate",
                                frame=dict(duration=140, redraw=True),
                                #不做平滑轉frames
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(
                        label="■",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                mode="immediate",
                                frame=dict(duration=0, redraw=False),
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(
                        label="⟳",
                        method="skip",
                        args=[],
                    ),
                ],
            ),
            dict(
                type="buttons",
                direction="left",
                x=0.36,
                y=0.015,
                xanchor="left",
                yanchor="bottom",
                pad=dict(r=10, t=4, b=4),
                bgcolor="rgba(255,255,255,0.88)",
                bordercolor="rgba(31,53,86,0.22)",
                borderwidth=1,
                font=dict(size=13, color="#1f3556"),
                active=0,
                buttons=[
                    dict(
                        label="全部",
                        method="skip",
                        args=[],
                    ),
                    dict(
                        label=f"隱藏 ({total_edge_residual})",
                        method="skip",
                        args=[],
                    ),
                ],
            ),
        ],
        sliders=[
            dict(
                active=0,
                x=0.20,
                y=0.06,
                #滑桿長度
                len=0.48,
                currentvalue=dict(
                    prefix="Time: ",
                    suffix=" s",
                    xanchor="right",
                    font=dict(size=15, color="#1f3556"),
                ),
                pad=dict(t=40, b=18),
                ticklen=4,
                font=dict(size=11, color="#1f3556"),
                steps=steps,
            )
        ],
    )

    output_path = OUTPUT_DIR / (
        f"{Path(result['file']).stem}_{method}{point_color_filename_suffix()}_interactive.html"
    )
    write_centered_html(fig, output_path)
    print(f"Interactive 3D viewer saved: {output_path}")
    return output_path


def write_static_overview(result, method=METHOD):
    xs, ys, zs, t0s, blocks = event_points(result, method)
    finite = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs) & np.isfinite(t0s)
    xs, ys, zs, t0s, blocks = xs[finite], ys[finite], zs[finite], t0s[finite], blocks[finite]
    if t0s.size == 0:
        return None
    relative_times = t0s - float(np.nanmin(t0s))
    color_values = point_color_values(xs, ys, zs, relative_times)
    color_min, color_max = point_color_range(color_values)
    edge_residual_mask, depth_values, residual_values = edge_high_residual_mask(result, xs, ys, blocks)
    visible = np.ones_like(relative_times, dtype=bool)
    traces = []
    surface_trace = specimen_surface_trace()
    if surface_trace is not None:
        traces.append(surface_trace)
    traces.extend([
        wireframe_trace(),
        sensor_trace(),
        internal_crack_plane_trace(xs, ys, zs, visible),
        internal_crack_network_trace(xs, ys, zs, visible),
        ae_trace(
            xs,
            ys,
            zs,
            relative_times,
            blocks,
            visible,
            color_values,
            color_min,
            color_max,
            depth_values,
            residual_values,
            edge_residual_mask,
        ),
        surface_crack_network_trace(xs, ys, zs, visible),
    ])
    fig = go.Figure(data=traces)
    fig.update_layout(**scene_layout(f"{result['file']} - {method.upper()} AE static overview"))
    output_path = OUTPUT_DIR / (
        f"{Path(result['file']).stem}_{method}{point_color_filename_suffix()}_static.html"
    )
    write_centered_html(fig, output_path)
    print(f"Static 3D overview saved: {output_path}")
    return output_path


def normalized_test_name(name):
    stem = Path(str(name)).stem.lower().replace("-", "_")
    if stem.startswith("test") and len(stem) > 4 and stem[4].isdigit():
        stem = "test_" + stem[4:]
    return stem


def max_time_steps_for_result(result):
    test_name = normalized_test_name(result.get("file", ""))
    return MAX_TIME_STEPS_BY_TEST.get(test_name, MAX_TIME_STEPS)


def target_results():
    if not TARGET_TESTS:
        return list(results)

    result_by_name = {
        normalized_test_name(item.get("file", "")): item
        for item in results
    }

    selected = []
    missing = []
    for test_name in TARGET_TESTS:
        item = result_by_name.get(normalized_test_name(test_name))
        if item is None:
            missing.append(test_name)
            continue
        selected.append(item)
    if not selected:
        for test_name in missing:
            print(f"Skip target {test_name}: not found in current results file")
    return selected


def export_current_results(results_path):
    global POINT_COLOR_MODE

    if RUN_MODE == "interactive_html":
        for item in target_results():
            test_name = Path(item.get("file", "")).stem
            if item.get("events"):
                print(
                    f"Exporting {test_name} from {Path(results_path).name}: "
                    f"{METHOD}, {', '.join(POINT_COLOR_MODES)}"
                )
                for color_mode in POINT_COLOR_MODES:
                    POINT_COLOR_MODE = color_mode
                    write_interactive_html(item, method=METHOD)
            else:
                print(f"Skip {test_name}: no events")
    elif RUN_MODE == "static_overview":
        for item in target_results():
            test_name = Path(item.get("file", "")).stem
            if item.get("events"):
                print(
                    f"Exporting {test_name} from {Path(results_path).name}: "
                    f"{METHOD}, {', '.join(POINT_COLOR_MODES)}"
                )
                for color_mode in POINT_COLOR_MODES:
                    POINT_COLOR_MODE = color_mode
                    write_static_overview(item, method=METHOD)
            else:
                print(f"Skip {test_name}: no events")
    else:
        raise ValueError("RUN_MODE must be 'interactive_html' or 'static_overview'")


for results_path in RESULTS_PATHS:
    data = load_results(results_path)
    apply_results_data(data)
    export_current_results(results_path)
