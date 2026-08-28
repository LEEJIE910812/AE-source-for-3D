import os
import numpy as np
import matplotlib as mpl
from concrete_common import load_results_pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FFMPEG_PATH = r"c:\Users\LJ\Desktop\ffmpeg-8.1-full_build\bin\ffmpeg.exe"
if os.path.exists(FFMPEG_PATH):
    mpl.rcParams["animation.ffmpeg_path"] = FFMPEG_PATH
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.animation as animation


# ========= 參數 =========
in_path = os.path.join(SCRIPT_DIR, "results_test_2_90.pkl")
OUT_DIR = os.path.dirname(in_path)
LEAD_IN = 1.0          # 片頭前導秒數（只旋轉，不出現事件）
FPS     = 10           # 影格率
N_PER_CSV   = 100      # 每個 CSV 的有效幀數（不含 gap）
GAP_FRAMES  = 0        # 段與段之間插入空白幀數（可保持 0）
LEVELS      = 50       # 色階等級數（離散分段越大，色塊越細）
POINT_COLOR_MODES = ("time", "depth")
EXPERIMENT_NAME = os.path.basename(SCRIPT_DIR).replace(".exp", "")

os.makedirs(OUT_DIR, exist_ok=True)

# ========= 讀資料 =========
data = load_results_pickle(in_path)

all_results      = data['results']
sensor_positions = data['sensor_positions']   # dict: sid -> (x,y,z)
cuboids          = data['cuboids']            # dict: name -> [low4, high4, ...]（鍵對 specimen）
specimen         = data['specimen']           # dict: point_id -> (x,y,z)
xmin, xmax       = data['xmin'], data['xmax']
ymin, ymax       = data['ymin'], data['ymax']
zmin, zmax       = data['zmin'], data['zmax']
cylinder_radius  = data.get('cylinder_radius')
cylinder_center_xy = data.get('cylinder_center_xy', (0.0, 0.0))
if cylinder_radius is None:
    cylinder_radius = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax))

# ========= 前處理：收集三種方法 (x,y,z,t0) 與每 CSV 的 t 範圍 =========
method_data_list = []   # only {'hypodd': (x, y, z, t0)}
file_names       = []
time_arrays      = []   # 每個 CSV 的 (t_min, t_max) —— 合併三方法後

for res in all_results:
    events = res['events']
    if not events:
        print(f"Skip empty CSV segment: {os.path.basename(res['file'])}")
        continue
    xs_h, ys_h, zs_h, t0s_h = [], [], [], []

    for e in events:
        # hypodd
        if e in res['hypodd']:
            x, y, z, t0 = res['hypodd'][e]
        else:
            x, y, z = res['g1_res'].get(e, res['g0_res'][e])
            t0      = res['g1_t0'].get(e, res['g0_t0'][e])
        xs_h.append(x); ys_h.append(y); zs_h.append(z); t0s_h.append(t0)

    md = {
        'hypodd': (np.array(xs_h), np.array(ys_h), np.array(zs_h), np.array(t0s_h, dtype=float)),
    }
    method_data_list.append(md)
    file_names.append(res['file'])

    all_t0 = md['hypodd'][3]
    # 防呆：空或全 NaN
    if all_t0.size == 0 or not np.isfinite(all_t0).any():
        time_arrays.append((0.0, 0.0))
    else:
        time_arrays.append((float(np.nanmin(all_t0)), float(np.nanmax(all_t0))))

n_csv = len(method_data_list)

# ========= 建全局時間軸：每 CSV 減去本段 t_min，接著做累積位移；片頭加 LEAD_IN =========
durations   = [max(0.0, tmax - tmin) for (tmin, tmax) in time_arrays]
offsets_base = np.cumsum([0.0] + durations[:-1])     # 不含 LEAD_IN 的偏移
T_total     = float(np.sum(durations))                # 全片（不含 LEAD_IN）

# ========= 離散色階設定（BoundaryNorm） =========
cmap   = plt.get_cmap('viridis', LEVELS)  # 離散 colormap


def inward_depth_cm(xs, ys):
    center_x, center_y = cylinder_center_xy
    radial_distance = np.hypot(np.asarray(xs, dtype=float) - center_x, np.asarray(ys, dtype=float) - center_y)
    return np.clip((float(cylinder_radius) - radial_distance) * 100.0, 0.0, float(cylinder_radius) * 100.0)


def point_color_values(xs, ys, zs, t0_global, color_mode):
    if color_mode == "depth":
        # Depth is measured radially inward from the cylinder surface, not by z-height.
        return inward_depth_cm(xs, ys)
    if color_mode == "time":
        return np.asarray(t0_global, dtype=float)
    raise ValueError("color_mode must be 'depth' or 'time'")


def point_colorbar_label(color_mode):
    if color_mode == "depth":
        return "Depth into specimen (cm)"
    return "t0 (s)"


def point_color_bounds(color_mode, total_time):
    if color_mode == "depth":
        return np.linspace(0.0, max(float(cylinder_radius) * 100.0, 1e-12), LEVELS + 1)
    # 全域時間範圍 0 ~ (LEAD_IN + T_total) 切成 LEVELS 等級
    return np.linspace(0.0, LEAD_IN + total_time, LEVELS + 1)

# ========= 幾何/旋轉 =========
cx, cy, cz = (xmin+xmax)/2.0, (ymin+ymax)/2.0, (zmin+zmax)/2.0

def rotate_xyz(x, y, z, theta_rad, axis='z', center=(cx, cy, cz)):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float); z = np.asarray(z, dtype=float)
    cx_, cy_, cz_ = center
    X = x - cx_; Y = y - cy_; Z = z - cz_
    c = np.cos(theta_rad); s = np.sin(theta_rad)
    if axis == 'z':
        Xr = c*X - s*Y; Yr = s*X + c*Y; Zr = Z
    elif axis == 'y':
        Xr =  c*X + s*Z; Yr = Y;        Zr = -s*X + c*Z
    elif axis == 'x':
        Xr = X;         Yr = c*Y - s*Z; Zr = s*Y + c*Z
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'.")
    return Xr + cx_, Yr + cy_, Zr + cz_


def numbered_points(prefix):
    items = []
    for point_id, position in specimen.items():
        if point_id.startswith(prefix) and point_id[len(prefix):].isdigit():
            items.append((int(point_id[len(prefix):]), position))
    return [position for _, position in sorted(items)]


def specimen_segments():
    if cuboids:
        for verts in cuboids.values():
            low, high = verts[:4], verts[4:]
            for seq in (low, high):
                for k in range(4):
                    yield specimen[seq[k]], specimen[seq[(k + 1) % 4]]
            for k in range(4):
                yield specimen[low[k]], specimen[high[k]]
        return

    top = numbered_points("T")
    bottom = numbered_points("B")
    if top and len(top) == len(bottom):
        for ring in (top, bottom):
            for k in range(len(ring)):
                yield ring[k], ring[(k + 1) % len(ring)]
        stride = max(1, len(top) // 12)
        for k in range(0, len(top), stride):
            yield top[k], bottom[k]


def draw_specimen_and_sensors(ax, theta, rotate_axis='z'):
    # specimen 線框
    for p1, p2 in specimen_segments():
        x1,y1,z1 = rotate_xyz(*p1, theta, axis=rotate_axis)
        x2,y2,z2 = rotate_xyz(*p2, theta, axis=rotate_axis)
        ax.plot([x1,x2], [y1,y2], [z1,z2], color='gray', alpha=0.8, linewidth=1.0)
    # sensors
    for sid, pos in sensor_positions.items():
        xs,ys,zs = rotate_xyz(*pos, theta, axis=rotate_axis)
        ax.scatter(xs, ys, zs, c='green', marker='^', s=25,alpha=0.7)
        ax.text(xs, ys, zs, f"S{sid}", color='green', fontsize=10, va='bottom', ha='left')

def setup_axes(ax):
    # 固定視角與範圍（正前方）
    ax.set_box_aspect((xmax-xmin, ymax-ymin, zmax-zmin))
    ax.view_init(elev=15, azim=-90)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)   # invert_yaxis 需求
    ax.set_zlim(zmin, zmax)
    ax.invert_yaxis()
    # 關掉網格線與刻度，保留 XYZ 標籤（若也要拿掉，改成 ''）
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    # 移除黑色外框線/面板（多版本相容）
    try:
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_edgecolor((1,1,1,0))
            pane.set_facecolor((1,1,1,0))
    except Exception:
        pass
    for axis in (getattr(ax, 'w_xaxis', ax.xaxis),
                 getattr(ax, 'w_yaxis', ax.yaxis),
                 getattr(ax, 'w_zaxis', ax.zaxis)):
        try:
            axis.line.set_lw(0.0)
        except Exception:
            try:
                axis.line.set_visible(False)
            except Exception:
                pass
        try:
            axis._axinfo["axisline"]["linewidth"] = 0.0
            axis._axinfo["grid"]["linewidth"]     = 0.0
        except Exception:
            pass
    try:
        ax.set_frame_on(False)
    except Exception:
        pass

# ========= 單一方法 MP4（global time + lead-in + 離散 colorbar + 累進顯示） =========
def safe_file_stem(name):
    stem = os.path.splitext(os.path.basename(str(name)))[0]
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in stem)


def render_method_video(method_key, out_name, csv_index,
                        n_frames_per_csv=N_PER_CSV, rotate_axis='z',
                        fps=FPS, color_mode="time"):
    xs, ys, zs, t0s = method_data_list[csv_index][method_key]
    tmin_i, tmax_i = time_arrays[csv_index]
    duration = max(0.0, tmax_i - tmin_i)
    lead_frames = int(round(LEAD_IN * fps))
    total_frames = lead_frames + n_frames_per_csv

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')

    bounds = point_color_bounds(color_mode, duration)
    norm = mpl.colors.BoundaryNorm(bounds, ncolors=cmap.N, clip=False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.05,
                        boundaries=bounds, spacing='proportional')
    cbar.set_label(point_colorbar_label(color_mode))

    title_name = os.path.splitext(os.path.basename(file_names[csv_index]))[0]

    def init():
        ax.clear()
        setup_axes(ax)
        fig.suptitle(title_name, fontsize=16)
        ax.set_title("HypoDD")
        return (ax,)

    def update(frame_idx):
        ax.clear()
        setup_axes(ax)

        theta = 2 * np.pi * (frame_idx / max(1, total_frames - 1))
        draw_specimen_and_sensors(ax, theta, rotate_axis=rotate_axis)
        fig.suptitle(title_name, fontsize=16)
        ax.set_title("HypoDD")

        if frame_idx < lead_frames:
            return (ax,)

        local_idx = min(frame_idx - lead_frames, n_frames_per_csv - 1)
        if n_frames_per_csv <= 1 or tmax_i <= tmin_i:
            current_local_t = tmin_i
        else:
            current_local_t = np.linspace(tmin_i, tmax_i, n_frames_per_csv)[local_idx]

        t0_global = LEAD_IN + (t0s - tmin_i)
        current_global_t = LEAD_IN + (current_local_t - tmin_i)
        mask = np.isfinite(t0_global) & (t0_global <= current_global_t)
        if np.any(mask):
            xsr, ysr, zsr = rotate_xyz(xs[mask], ys[mask], zs[mask], theta, axis=rotate_axis)
            colors = point_color_values(xs[mask], ys[mask], zs[mask], t0_global[mask], color_mode)
            ax.scatter(xsr, ysr, zsr, c=colors, cmap=cmap, norm=norm,
                       s=4, marker='o', alpha=0.7)
        return (ax,)

    anim = animation.FuncAnimation(fig, update, init_func=init,
                                   frames=total_frames, interval=1000 // fps, blit=False)

    FFMpegWriter = animation.FFMpegWriter
    if not FFMpegWriter.isAvailable():
        raise RuntimeError("ffmpeg is not available.")
    writer = FFMpegWriter(fps=FPS, metadata=dict(artist='leejie'), bitrate=6000)
    out_mp4 = os.path.join(OUT_DIR, out_name)
    anim.save(out_mp4, writer=writer)
    plt.close(fig)
    print(f"Saved: {out_mp4}")


for csv_index, file_name in enumerate(file_names):
    test_name = safe_file_stem(file_name)
    for color_mode in POINT_COLOR_MODES:
        suffix = "" if color_mode == "time" else f"-{color_mode}"
        render_method_video(
            'hypodd',
            f'{EXPERIMENT_NAME}_{test_name}_hypodd{suffix}.mp4',
            csv_index=csv_index,
            n_frames_per_csv=N_PER_CSV,
            rotate_axis='z',
            fps=FPS,
            color_mode=color_mode,
        )

print("Done.")

