from __future__ import annotations
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import csv
import os
from typing import Iterable
import h5py
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
import tpc5


# ============================================================
# 使用者設定：通常會改 tpc5、channel、STA/LTA
# ============================================================

TPC5_FILE_PATTERN = "data/test_2_90.tpc5"

# True 會覆蓋 csv/<tpc5檔名>.csv；False 則保留既有檔案。
OVERWRITE_EXISTING_CSV = True

# 一次處理幾個檔案
PARALLEL_WORKERS = 1

# 要做 arrival pick 的 channel。
CHANNELS_TO_PICK = (1, 2, 3, 4, 5, 6)

# STA/LTA trigger。數字越大越不容易觸發，數字越小越敏感。
STA_LTA_TRIGGER_LEVEL = {
    1: 5.0,
    2: 5.0,
    3: 5.0,
    4: 4.0,
    5: 5.0,
    6: 5.0,
}


# ============================================================
# 固定 pick 設定：一般不用改
# ============================================================

OUTPUT_CSV_FOLDER = "csv"
FILTER_BAND_HZ = (2000.0, 20000.0)
IGNORE_FIRST_SECONDS = 0.001
AIC_SEARCH_WINDOW_SECONDS = 0.005

# Use an extra visual-onset correction after STA/LTA + AIC.
# The CSV header remains AIC_Pick_Time_s so the location code can read it.
USE_VISUAL_ONSET_PICK = True
VISUAL_ONSET_SEARCH_BEFORE_SECONDS = 0.0030
VISUAL_ONSET_SEARCH_AFTER_SECONDS = 0.0010
VISUAL_ONSET_NOISE_BEFORE_SECONDS = 0.006
VISUAL_ONSET_NOISE_GAP_SECONDS = 0.0008
VISUAL_ONSET_SMOOTH_SECONDS = 0.000020
VISUAL_ONSET_MIN_RUN_SECONDS = 0.000010
VISUAL_ONSET_PEAK_FRACTION = 0.03
VISUAL_ONSET_NOISE_MAD_FACTOR = 4.0
VISUAL_ONSET_NOISE_PERCENTILE = 99.5
VISUAL_ONSET_PERCENTILE_FACTOR = 1.05
VISUAL_ONSET_MIN_SIGNAL_TO_NOISE = 2.0
VISUAL_ONSET_EVENT_CLUSTER_GAP_SECONDS = 0.0010
VISUAL_ONSET_PREFER_LATEST_EVENT_CLUSTER = True

# 同一個 block 允許偵測多個明顯 onset。先以多 channel 的包絡線共同上升找候選，
# 再檢查各 channel 的峰值振幅與 SNR；至少通過指定 channel 數才寫入 CSV。
MULTI_ONSET_MIN_CHANNELS = 4
MULTI_ONSET_ENVELOPE_SMOOTH_SECONDS = 0.000050
MULTI_ONSET_DETECTION_SNR = 4.5
MULTI_ONSET_MIN_ACTIVE_SECONDS = 0.000020
MULTI_ONSET_MERGE_GAP_SECONDS = 0.0020
MULTI_ONSET_PEAK_SEARCH_SECONDS = 0.0015
MULTI_ONSET_SIGNAL_WINDOW_SECONDS = 0.0020
MULTI_ONSET_NOISE_WINDOW_SECONDS = 0.0030
MULTI_ONSET_NOISE_GAP_SECONDS = 0.0003
MULTI_ONSET_MIN_PEAK_AMPLITUDE_V = 0.00008
MULTI_ONSET_MIN_AMPLITUDE_SNR = 5.0

# ============================================================
# 下面是程式內部流程。一般調參不需要從這裡開始改。
# ============================================================



@dataclass(frozen=True)
class PickConfig:
    """Internal settings used by the picking functions."""

    channels: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    filter_order: int = 3
    lowcut: float = 2000.0
    highcut: float = 20000.0
    cut_seconds: float = 0.001
    aic_window_seconds: float = 0.005
    min_group_seconds: float = 0.0
    min_samples: int = 10
    max_fft_samples: int = 131_072
    eps: float = 1e-6
    ratio_thresholds: dict[int, float] = field(
        default_factory=lambda: STA_LTA_TRIGGER_LEVEL.copy()
    )


def build_pick_config() -> PickConfig:
    """Convert the simple user settings above into the internal config object."""
    lowcut, highcut = FILTER_BAND_HZ
    return PickConfig(
        channels=CHANNELS_TO_PICK,
        lowcut=lowcut,
        highcut=highcut,
        cut_seconds=IGNORE_FIRST_SECONDS,
        aic_window_seconds=AIC_SEARCH_WINDOW_SECONDS,
        ratio_thresholds=STA_LTA_TRIGGER_LEVEL.copy(),
    )

@dataclass
class ChannelDiagnostics:
    block: int
    channel: int
    sample_rate: float
    trigger_sample: float
    trigger_time: float
    time_raw: np.ndarray
    raw: np.ndarray
    time: np.ndarray
    filtered: np.ndarray
    ratio: np.ndarray
    threshold: float
    sta_pick_index: int | None
    aic_pick_index: int | None
    pick_time: float
    peak_frequency: float
    fft_frequency: np.ndarray
    fft_amplitude: np.ndarray
    status: str
    onset_pick_times: list[float] = field(default_factory=list)
    onset_peak_amplitudes: list[float] = field(default_factory=list)
    onset_amplitude_snrs: list[float] = field(default_factory=list)


@dataclass
class BlockOnset:
    onset: int
    center_time: float
    channel_times: dict[int, float]
    peak_amplitudes: dict[int, float]
    amplitude_snrs: dict[int, float]


def discover_valid_blocks(file_handle, reference_channel: int = 1) -> list[int]:
    block_group_name = (
        f"/measurements/00000001/channels/{reference_channel:08d}/blocks"
    )
    try:
        block_group = file_handle[block_group_name]
    except KeyError:
        return []

    blocks: list[int] = []
    for name in block_group.keys():
        try:
            blocks.append(int(name))
        except ValueError:
            continue
    return sorted(blocks)


def _call_tpc5(snake_name: str, camel_name: str, *args):
    func = getattr(tpc5, snake_name, None)
    if func is None:
        func = getattr(tpc5, camel_name)
    return func(*args)


def load_trace(file_handle, channel: int, block: int) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    data = np.asarray(_call_tpc5("get_voltage_data", "getVoltageData", file_handle, channel, block), dtype=float)
    sample_rate = float(_call_tpc5("get_sample_rate", "getSampleRate", file_handle, channel, block))
    trigger_sample = float(_call_tpc5("get_trigger_sample", "getTriggerSample", file_handle, channel, block))
    trigger_time = float(_call_tpc5("get_trigger_time", "getTriggerTime", file_handle, channel, block))
    time = trigger_time + (np.arange(data.size, dtype=float) - trigger_sample) / sample_rate
    return time, data, sample_rate, trigger_sample, trigger_time


def design_bandpass(sample_rate: float, config: PickConfig) -> tuple[np.ndarray, np.ndarray] | None:
    nyquist = 0.5 * sample_rate
    if nyquist <= 0:
        return None
    low = min(max(config.lowcut / nyquist, config.eps), 1.0 - config.eps)
    high = min(max(config.highcut / nyquist, config.eps), 1.0 - config.eps)
    if not low < high:
        return None
    return butter(config.filter_order, [low, high], btype="band")


def filter_trace(data: np.ndarray, sample_rate: float, config: PickConfig) -> np.ndarray:
    coeff = design_bandpass(sample_rate, config)
    if coeff is None:
        return np.asarray(data, dtype=float).copy()
    b, a = coeff
    return filtfilt(b, a, np.asarray(data, dtype=float))


def compute_peak_frequency(
    signal: np.ndarray,
    sample_rate: float,
    max_fft_samples: int = 131_072,
) -> tuple[np.ndarray, np.ndarray, float]:
    source = np.asarray(signal, dtype=float)
    if source.size > max_fft_samples > 0:
        source = source[:max_fft_samples]
    demeaned = source - float(np.mean(source))
    spectrum = np.fft.rfft(demeaned)
    frequency = np.fft.rfftfreq(demeaned.size, d=1.0 / sample_rate)
    amplitude = np.abs(spectrum)
    if amplitude.size:
        amplitude[0] = 0.0
    peak_index = int(np.argmax(amplitude)) if amplitude.size else 0
    peak_frequency = float(frequency[peak_index]) if frequency.size else 0.0
    return frequency, amplitude, peak_frequency


def compute_sta_lta_ratio(signal: np.ndarray, sample_rate: float, peak_frequency: float) -> np.ndarray:
    if peak_frequency <= 0.0:
        sta_samples = 1
        lta_samples = 2
    else:
        sta_samples = max(int((10.0 / peak_frequency) * sample_rate), 1)
        lta_samples = max(int((100.0 / peak_frequency) * sample_rate), sta_samples + 1)

    energy = np.square(signal)
    sta = moving_average_same(energy, sta_samples)
    lta = moving_average_same(energy, lta_samples)
    length = min(signal.size, sta.size, lta.size)
    return sta[:length] / (lta[:length] + 1e-12)


def moving_average_same(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1:
        return np.asarray(values, dtype=float).copy()
    values = np.asarray(values, dtype=float)
    left = window_size // 2
    right = window_size - 1 - left
    padded = np.pad(values, (left, right), mode="constant")
    cumsum = np.cumsum(np.concatenate(([0.0], padded)))
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size


def find_sta_lta_pick(
    ratio: np.ndarray,
    threshold: float,
    sample_rate: float,
    min_group_seconds: float,
) -> int | None:
    above = np.where(ratio >= threshold)[0]
    if above.size == 0:
        return None

    groups = np.split(above, np.where(np.diff(above) != 1)[0] + 1)
    min_group_samples = max(int(min_group_seconds * sample_rate), 1)
    for group in groups:
        if group.size >= min_group_samples:
            return int(group[0])
    return int(above[0])


def refine_pick_with_aic(
    signal: np.ndarray,
    time: np.ndarray,
    pick_index: int,
    sample_rate: float,
    window_seconds: float,
) -> tuple[int | None, float]:
    half_window = int(window_seconds * sample_rate)
    start = max(pick_index - half_window, 1)
    end = min(pick_index + half_window, signal.size - 1)
    if end - start <= 3:
        return None, float("nan")

    envelope = np.abs(hilbert(signal[start:end]))
    local_index = int(np.argmin(compute_aic_curve(envelope)))
    absolute_index = start + local_index
    return absolute_index, float(time[absolute_index])


def compute_aic_curve(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = values.size
    if n == 0:
        return np.asarray([], dtype=float)

    cumsum = np.cumsum(values)
    cumsum2 = np.cumsum(values * values)
    total_sum = cumsum[-1]
    total_sum2 = cumsum2[-1]
    aic = np.full(n, np.inf, dtype=float)

    for index in range(n):
        if index > 1:
            left_sum = cumsum[index - 1]
            left_sum2 = cumsum2[index - 1]
            left_var = max(left_sum2 / index - (left_sum / index) ** 2, 1e-10)
        else:
            left_var = 1e-10

        right_count = n - index
        if right_count > 1:
            right_sum = total_sum - (cumsum[index - 1] if index > 0 else 0.0)
            right_sum2 = total_sum2 - (cumsum2[index - 1] if index > 0 else 0.0)
            right_var = max(right_sum2 / right_count - (right_sum / right_count) ** 2, 1e-10)
        else:
            right_var = 1e-10

        aic[index] = index * np.log(left_var) + (n - index - 1) * np.log(right_var)
    return aic


def diagnose_channel(
    data: np.ndarray,
    sample_rate: float,
    trigger_sample: float,
    trigger_time: float,
    block: int,
    channel: int,
    config: PickConfig,
) -> ChannelDiagnostics:
    raw = np.asarray(data, dtype=float)
    time_raw = trigger_time + (np.arange(raw.size, dtype=float) - trigger_sample) / sample_rate

    threshold = config.ratio_thresholds.get(channel, 3.0)
    empty = np.asarray([], dtype=float)
    default = ChannelDiagnostics(
        block=block,
        channel=channel,
        sample_rate=sample_rate,
        trigger_sample=trigger_sample,
        trigger_time=trigger_time,
        time_raw=time_raw,
        raw=raw,
        time=empty,
        filtered=empty,
        ratio=empty,
        threshold=threshold,
        sta_pick_index=None,
        aic_pick_index=None,
        pick_time=float("nan"),
        peak_frequency=0.0,
        fft_frequency=empty,
        fft_amplitude=empty,
        status="no_data",
    )

    if raw.size < config.min_samples:
        default.status = "too_short"
        return default

    try:
        filtered_raw = filter_trace(raw, sample_rate, config)
    except Exception as exc:
        default.status = f"filter_failed:{type(exc).__name__}"
        return default

    mask = time_raw >= (time_raw[0] + config.cut_seconds)
    time = time_raw[mask]
    filtered = filtered_raw[mask]
    if filtered.size < config.min_samples:
        default.filtered = filtered
        default.time = time
        default.status = "too_short_after_cut"
        return default

    fft_frequency, fft_amplitude, peak_frequency = compute_peak_frequency(
        filtered,
        sample_rate,
        max_fft_samples=config.max_fft_samples,
    )
    ratio = compute_sta_lta_ratio(filtered, sample_rate, peak_frequency)
    length = min(time.size, filtered.size, ratio.size)
    time = time[:length]
    filtered = filtered[:length]
    ratio = ratio[:length]

    sta_pick_index = find_sta_lta_pick(
        ratio=ratio,
        threshold=threshold,
        sample_rate=sample_rate,
        min_group_seconds=config.min_group_seconds,
    )
    if sta_pick_index is None:
        return ChannelDiagnostics(
            block=block,
            channel=channel,
            sample_rate=sample_rate,
            trigger_sample=trigger_sample,
            trigger_time=trigger_time,
            time_raw=time_raw,
            raw=raw,
            time=time,
            filtered=filtered,
            ratio=ratio,
            threshold=threshold,
            sta_pick_index=None,
            aic_pick_index=None,
            pick_time=float("nan"),
            peak_frequency=peak_frequency,
            fft_frequency=fft_frequency,
            fft_amplitude=fft_amplitude,
            status="no_sta_lta_trigger",
        )

    aic_pick_index, pick_time = refine_pick_with_aic(
        signal=filtered,
        time=time,
        pick_index=sta_pick_index,
        sample_rate=sample_rate,
        window_seconds=config.aic_window_seconds,
    )
    return ChannelDiagnostics(
        block=block,
        channel=channel,
        sample_rate=sample_rate,
        trigger_sample=trigger_sample,
        trigger_time=trigger_time,
        time_raw=time_raw,
        raw=raw,
        time=time,
        filtered=filtered,
        ratio=ratio,
        threshold=threshold,
        sta_pick_index=sta_pick_index,
        aic_pick_index=aic_pick_index,
        pick_time=pick_time,
        peak_frequency=peak_frequency,
        fft_frequency=fft_frequency,
        fft_amplitude=fft_amplitude,
        status="ok" if np.isfinite(pick_time) else "aic_failed",
    )


def analyse_block(file_handle, block: int, config: PickConfig) -> dict[int, ChannelDiagnostics]:
    diagnostics: dict[int, ChannelDiagnostics] = {}
    for channel in config.channels:
        try:
            time, data, sample_rate, trigger_sample, trigger_time = load_trace(file_handle, channel, block)
            diagnostics[channel] = diagnose_channel(
                data=data,
                sample_rate=sample_rate,
                trigger_sample=trigger_sample,
                trigger_time=trigger_time,
                block=block,
                channel=channel,
                config=config,
            )
        except (KeyError, OSError, ValueError) as exc:
            diagnostics[channel] = ChannelDiagnostics(
                block=block,
                channel=channel,
                sample_rate=float("nan"),
                trigger_sample=float("nan"),
                trigger_time=float("nan"),
                time_raw=np.asarray([], dtype=float),
                raw=np.asarray([], dtype=float),
                time=np.asarray([], dtype=float),
                filtered=np.asarray([], dtype=float),
                ratio=np.asarray([], dtype=float),
                threshold=config.ratio_thresholds.get(channel, 3.0),
                sta_pick_index=None,
                aic_pick_index=None,
                pick_time=float("nan"),
                peak_frequency=float("nan"),
                fft_frequency=np.asarray([], dtype=float),
                fft_amplitude=np.asarray([], dtype=float),
                status=f"read_failed:{type(exc).__name__}",
            )
    return diagnostics


def _cluster_event_time(times: np.ndarray) -> float | None:
    times = np.asarray([time for time in times if np.isfinite(time)], dtype=float)
    if times.size == 0:
        return None

    sorted_times = np.sort(times)
    clusters: list[np.ndarray] = []
    start = 0
    for index in range(1, sorted_times.size):
        if sorted_times[index] - sorted_times[index - 1] > VISUAL_ONSET_EVENT_CLUSTER_GAP_SECONDS:
            clusters.append(sorted_times[start:index])
            start = index
    clusters.append(sorted_times[start:])

    # 當 AIC pick 被分成早、晚兩群時，先用最多 channel 支持的那一群當共同事件時間。
    # 若兩群數量一樣，晚群通常比較接近真正起振後的共同波包，可避免搜尋窗落在兩群中間。
    best_count = max(cluster.size for cluster in clusters)
    best_clusters = [cluster for cluster in clusters if cluster.size == best_count]
    if VISUAL_ONSET_PREFER_LATEST_EVENT_CLUSTER:
        selected = max(best_clusters, key=lambda cluster: float(np.median(cluster)))
    else:
        global_median = float(np.median(times))
        selected = min(
            best_clusters,
            key=lambda cluster: abs(float(np.median(cluster)) - global_median),
        )

    return float(np.median(selected))


def robust_event_time(diagnostics: dict[int, ChannelDiagnostics]) -> float | None:
    pick_times = np.asarray(
        [
            diag.pick_time
            for diag in diagnostics.values()
            if np.isfinite(diag.pick_time)
        ],
        dtype=float,
    )
    return _cluster_event_time(pick_times)


def find_visual_onset_time(diag: ChannelDiagnostics, event_time: float) -> float | None:
    if (
        not np.isfinite(event_time)
        or diag.time.size < 10
        or diag.filtered.size < 10
        or not np.isfinite(diag.sample_rate)
        or diag.sample_rate <= 0
    ):
        return None

    length = min(diag.time.size, diag.filtered.size)
    time = diag.time[:length]
    signal = diag.filtered[:length]
    sample_rate = float(diag.sample_rate)

    smooth_samples = max(1, int(round(VISUAL_ONSET_SMOOTH_SECONDS * sample_rate)))
    envelope = moving_average_same(np.abs(signal), smooth_samples)

    noise_mask = (
        (time >= event_time - VISUAL_ONSET_NOISE_BEFORE_SECONDS)
        & (time <= event_time - VISUAL_ONSET_NOISE_GAP_SECONDS)
    )
    search_mask = (
        (time >= event_time - VISUAL_ONSET_SEARCH_BEFORE_SECONDS)
        & (time <= event_time + VISUAL_ONSET_SEARCH_AFTER_SECONDS)
    )
    search_indices = np.where(search_mask)[0]
    if search_indices.size < 10:
        return None

    if np.count_nonzero(noise_mask) < 20:
        noise_mask = time < event_time - VISUAL_ONSET_NOISE_GAP_SECONDS
    noise = envelope[noise_mask] if np.any(noise_mask) else envelope[: search_indices[0]]
    if noise.size < 5:
        return None

    noise_floor = float(np.median(noise))
    noise_mad = float(np.median(np.abs(noise - noise_floor)))
    noise_sigma = 1.4826 * noise_mad
    noise_percentile = float(np.percentile(noise, VISUAL_ONSET_NOISE_PERCENTILE))
    search_envelope = envelope[search_indices]
    peak = float(np.max(search_envelope))
    dynamic_range = max(peak - noise_floor, 0.0)
    if peak <= noise_floor * VISUAL_ONSET_MIN_SIGNAL_TO_NOISE and dynamic_range <= noise_sigma:
        return None

    threshold = max(
        noise_floor + VISUAL_ONSET_PEAK_FRACTION * dynamic_range,
        noise_floor + VISUAL_ONSET_NOISE_MAD_FACTOR * noise_sigma,
        noise_percentile * VISUAL_ONSET_PERCENTILE_FACTOR,
    )
    run_samples = max(1, int(round(VISUAL_ONSET_MIN_RUN_SECONDS * sample_rate)))
    above = search_envelope >= threshold
    if not np.any(above):
        return None

    for offset in np.where(above)[0]:
        if offset + run_samples <= above.size and np.all(above[offset : offset + run_samples]):
            return float(time[search_indices[offset]])
    return float(time[search_indices[int(np.argmax(above))]])


def _multi_onset_candidate_times(
    diagnostics: dict[int, ChannelDiagnostics],
) -> list[float]:
    valid = [
        diag
        for diag in diagnostics.values()
        if diag.time.size >= 10
        and diag.filtered.size >= 10
        and np.isfinite(diag.sample_rate)
        and diag.sample_rate > 0
    ]
    if len(valid) < MULTI_ONSET_MIN_CHANNELS:
        return []

    reference = min(valid, key=lambda diag: diag.time.size)
    common_time = reference.time
    sample_rate = float(reference.sample_rate)
    normalized_envelopes: list[np.ndarray] = []

    for diag in valid:
        length = min(diag.time.size, diag.filtered.size)
        smooth_samples = max(
            1,
            int(round(MULTI_ONSET_ENVELOPE_SMOOTH_SECONDS * diag.sample_rate)),
        )
        envelope = moving_average_same(np.abs(diag.filtered[:length]), smooth_samples)
        noise_floor = float(np.median(envelope))
        noise_sigma = 1.4826 * float(np.median(np.abs(envelope - noise_floor)))
        scale = max(noise_sigma, noise_floor * 0.05, np.finfo(float).eps)
        normalized = np.maximum((envelope - noise_floor) / scale, 0.0)
        normalized_envelopes.append(
            np.interp(
                common_time,
                diag.time[:length],
                normalized,
                left=0.0,
                right=0.0,
            )
        )

    normalized_stack = np.asarray(normalized_envelopes, dtype=float)
    # 第 N 大的 SNR 代表至少 N 個 channel 同時支持這個候選。
    common_score = np.sort(normalized_stack, axis=0)[-MULTI_ONSET_MIN_CHANNELS]
    active = common_score >= MULTI_ONSET_DETECTION_SNR
    if not np.any(active):
        return []

    starts = np.where(active & ~np.r_[False, active[:-1]])[0]
    ends = np.where(active & ~np.r_[active[1:], False])[0]
    min_samples = max(1, int(round(MULTI_ONSET_MIN_ACTIVE_SECONDS * sample_rate)))
    runs = [
        [int(start), int(end)]
        for start, end in zip(starts, ends)
        if end - start + 1 >= min_samples
    ]
    if not runs:
        return []

    merge_samples = max(1, int(round(MULTI_ONSET_MERGE_GAP_SECONDS * sample_rate)))
    merged: list[list[int]] = [runs[0]]
    for start, end in runs[1:]:
        if start - merged[-1][1] - 1 <= merge_samples:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    search_samples = max(1, int(round(MULTI_ONSET_PEAK_SEARCH_SECONDS * sample_rate)))
    candidates: list[float] = []
    for start, end in merged:
        peak_end = min(end + 1, start + search_samples)
        peak_index = start + int(np.argmax(common_score[start:peak_end]))
        candidates.append(float(common_time[peak_index]))
    return candidates


def _measure_onset_amplitude(
    diag: ChannelDiagnostics,
    pick_time: float,
) -> tuple[float, float]:
    if not np.isfinite(pick_time) or diag.time.size < 10 or diag.filtered.size < 10:
        return float("nan"), float("nan")

    length = min(diag.time.size, diag.filtered.size)
    time = diag.time[:length]
    signal = diag.filtered[:length]
    signal_mask = (
        (time >= pick_time)
        & (time <= pick_time + MULTI_ONSET_SIGNAL_WINDOW_SECONDS)
    )
    noise_mask = (
        (time >= pick_time - MULTI_ONSET_NOISE_WINDOW_SECONDS)
        & (time <= pick_time - MULTI_ONSET_NOISE_GAP_SECONDS)
    )
    if np.count_nonzero(signal_mask) < 5 or np.count_nonzero(noise_mask) < 5:
        return float("nan"), float("nan")

    peak_amplitude = float(np.max(np.abs(signal[signal_mask])))
    noise = signal[noise_mask]
    noise_center = float(np.median(noise))
    noise_sigma = 1.4826 * float(np.median(np.abs(noise - noise_center)))
    amplitude_snr = peak_amplitude / max(noise_sigma, np.finfo(float).eps)
    return peak_amplitude, float(amplitude_snr)


def apply_visual_onset_pick_times(
    diagnostics: dict[int, ChannelDiagnostics],
) -> list[BlockOnset]:
    if not USE_VISUAL_ONSET_PICK:
        return []

    for diag in diagnostics.values():
        diag.onset_pick_times = []
        diag.onset_peak_amplitudes = []
        diag.onset_amplitude_snrs = []

    accepted: list[BlockOnset] = []
    for candidate_time in _multi_onset_candidate_times(diagnostics):
        channel_times: dict[int, float] = {}
        peak_amplitudes: dict[int, float] = {}
        amplitude_snrs: dict[int, float] = {}

        for channel, diag in diagnostics.items():
            visual_time = find_visual_onset_time(diag, candidate_time)
            if visual_time is None or not np.isfinite(visual_time):
                continue
            peak_amplitude, amplitude_snr = _measure_onset_amplitude(diag, visual_time)
            if (
                not np.isfinite(peak_amplitude)
                or not np.isfinite(amplitude_snr)
                or peak_amplitude < MULTI_ONSET_MIN_PEAK_AMPLITUDE_V
                or amplitude_snr < MULTI_ONSET_MIN_AMPLITUDE_SNR
            ):
                continue

            channel_times[int(channel)] = float(visual_time)
            peak_amplitudes[int(channel)] = float(peak_amplitude)
            amplitude_snrs[int(channel)] = float(amplitude_snr)

        if len(channel_times) < MULTI_ONSET_MIN_CHANNELS:
            continue

        onset_number = len(accepted) + 1
        accepted.append(
            BlockOnset(
                onset=onset_number,
                center_time=float(np.median(list(channel_times.values()))),
                channel_times=channel_times,
                peak_amplitudes=peak_amplitudes,
                amplitude_snrs=amplitude_snrs,
            )
        )

    for onset in accepted:
        for channel, diag in diagnostics.items():
            diag.onset_pick_times.append(onset.channel_times.get(channel, float("nan")))
            diag.onset_peak_amplitudes.append(
                onset.peak_amplitudes.get(channel, float("nan"))
            )
            diag.onset_amplitude_snrs.append(
                onset.amplitude_snrs.get(channel, float("nan"))
            )

    for diag in diagnostics.values():
        finite_times = [time for time in diag.onset_pick_times if np.isfinite(time)]
        diag.pick_time = finite_times[0] if finite_times else float("nan")
        diag.status = f"multi_onset:{len(finite_times)}" if finite_times else "no_qualified_onset"
    return accepted


PICK_CSV_FIELDS = ["Block", "Channel", "AIC_Pick_Time_s"]


def _pick_rows_for_block(
    block: int,
    diagnostics: dict[int, ChannelDiagnostics],
) -> list[dict[str, str]]:
    onset_count = max(
        (len(diag.onset_pick_times) for diag in diagnostics.values()),
        default=0,
    )
    if onset_count == 0:
        onset_count = 1

    rows: list[dict[str, str]] = []
    for onset_index in range(onset_count):
        for channel in sorted(diagnostics):
            diag = diagnostics[channel]
            pick_time = (
                diag.onset_pick_times[onset_index]
                if onset_index < len(diag.onset_pick_times)
                else float("nan")
            )
            rows.append(
                {
                    "Block": f"{block}-{onset_index + 1}",
                    "Channel": str(channel),
                    "AIC_Pick_Time_s": (
                        f"{float(pick_time):.12g}"
                        if np.isfinite(pick_time)
                        else "none"
                    ),
                }
            )
    return rows


def write_pick_csv(output_csv: Path | str, rows: Iterable[dict[str, str]]) -> Path:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=PICK_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def update_pick_csv_for_block(
    output_csv: Path | str,
    block: int,
    diagnostics: dict[int, ChannelDiagnostics],
) -> Path:
    """Update one block in an existing pick CSV without deleting other blocks."""
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    if output_path.exists():
        with output_path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                row_block = str(row.get("Block", "")).strip()
                if not row_block:
                    continue
                base_block = row_block.split("-", 1)[0]
                if base_block != str(block):
                    old_onset = str(row.get("Onset", "")).strip()
                    block_label = (
                        f"{row_block}-{old_onset}"
                        if old_onset and "-" not in row_block
                        else row_block
                    )
                    rows.append(
                        {
                            "Block": block_label,
                            "Channel": row.get("Channel", ""),
                            "AIC_Pick_Time_s": row.get("AIC_Pick_Time_s", ""),
                        }
                    )

    rows.extend(_pick_rows_for_block(block, diagnostics))

    def row_sort_key(row: dict[str, str]) -> tuple[int, int, int]:
        block_text = row["Block"]
        base_text, _, event_text = block_text.partition("-")
        return (
            int(base_text),
            int(event_text or 1),
            int(row["Channel"]),
        )

    rows.sort(
        key=row_sort_key
    )
    with output_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=PICK_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def _thin_for_plot(x: np.ndarray, y: np.ndarray, max_points: int = 8000) -> tuple[np.ndarray, np.ndarray]:
    length = min(x.size, y.size)
    if length <= max_points:
        return x[:length], y[:length]
    indices = np.linspace(0, length - 1, max_points, dtype=int)
    return x[indices], y[indices]


def plot_dynamic_pick_diagnostics(
    diagnostics: dict[int, ChannelDiagnostics],
    title: str,
    max_plot_points: int = 8000,
):
    import matplotlib.pyplot as plt

    channels = sorted(diagnostics)
    fig, axes = plt.subplots(len(channels), 1, figsize=(10, 8), sharex=True)
    if len(channels) == 1:
        axes = [axes]

    for axis, channel in zip(axes, channels):
        diag = diagnostics[channel]
        ratio_axis = axis.twinx()

        if diag.time.size and diag.filtered.size:
            time_plot, filtered_plot = _thin_for_plot(
                diag.time,
                diag.filtered,
                max_points=max_plot_points,
            )
            axis.plot(time_plot, filtered_plot, label="Filtered Signal", alpha=0.6)
        else:
            axis.text(
                0.5,
                0.5,
                diag.status,
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="tab:red",
            )

        if diag.ratio.size and diag.time.size:
            length = min(diag.time.size, diag.ratio.size)
            ratio_time, ratio_plot = _thin_for_plot(
                diag.time[:length],
                diag.ratio[:length],
                max_points=max_plot_points,
            )
            ratio_axis.plot(
                ratio_time,
                ratio_plot,
                "k--",
                linewidth=0.8,
                label="STA/LTA",
            )

        ratio_axis.axhline(diag.threshold, color="gray", linestyle=":", linewidth=0.8)
        ratio_axis.axhline(1.2, color="gray", linestyle="-.", linewidth=0.8)

        if diag.sta_pick_index is not None and diag.sta_pick_index < diag.time.size:
            pick_index = diag.sta_pick_index
            axis.axvline(diag.time[pick_index], color="red", linestyle="--", linewidth=1.0)
            axis.scatter(
                diag.time[pick_index],
                diag.filtered[pick_index],
                s=50,
                c="red",
                edgecolors="k",
                zorder=5,
                label="STA/LTA Pick",
            )

        onset_times = diag.onset_pick_times or (
            [diag.pick_time] if np.isfinite(diag.pick_time) else []
        )
        onset_colors = ("green", "tab:purple", "tab:orange", "tab:brown", "tab:pink")
        for onset_index, onset_time in enumerate(onset_times, start=1):
            if not np.isfinite(onset_time) or not diag.time.size or not diag.filtered.size:
                continue
            color = onset_colors[(onset_index - 1) % len(onset_colors)]
            y_pick = np.interp(onset_time, diag.time, diag.filtered)
            axis.axvline(onset_time, color=color, linestyle="--", linewidth=1.0)
            axis.scatter(
                onset_time,
                y_pick,
                s=50,
                c=color,
                edgecolors="k",
                zorder=5,
                label=f"Block {diag.block}-{onset_index} Pick",
            )
            axis.text(
                onset_time,
                y_pick,
                f"B{diag.block}-{onset_index} {onset_time:.6f}s",
                color=color,
                fontsize=8,
                ha="left",
                va="bottom",
            )

        axis.set_ylabel(f"Ch{channel}\nFiltered (V)")
        ratio_axis.set_ylabel("STA/LTA")
        axis.grid(True)

        handles_1, labels_1 = axis.get_legend_handles_labels()
        handles_2, labels_2 = ratio_axis.get_legend_handles_labels()
        if handles_1 or handles_2:
            axis.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper right", fontsize=6)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, y=0.96)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def save_dynamic_pick_figure(
    diagnostics: dict[int, ChannelDiagnostics],
    output_path: Path | str,
    title: str,
    dpi: int = 150,
    max_plot_points: int = 8000,
) -> Path:
    import matplotlib.pyplot as plt

    resolved_output = Path(output_path)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_dynamic_pick_diagnostics(
        diagnostics,
        title=title,
        max_plot_points=max_plot_points,
    )
    fig.savefig(resolved_output, dpi=dpi)
    plt.close(fig)
    return resolved_output


def export_pick_diagnostic_images_for_file(
    file_path: Path | str,
    output_dir: Path | str,
    config: PickConfig | None = None,
    overwrite: bool = False,
    dpi: int = 150,
    max_plot_points: int = 8000,
) -> list[Path]:
    resolved_config = config or PickConfig()
    source_path = Path(file_path)
    destination_dir = Path(output_dir) / source_path.stem
    destination_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    with h5py.File(str(source_path), "r") as file_handle:
        for block in discover_valid_blocks(file_handle):
            output_path = destination_dir / f"block_{block:06d}_dynamic_sta_lta_aic.png"
            if output_path.exists() and not overwrite:
                written.append(output_path)
                continue

            diagnostics = analyse_block(file_handle, block, resolved_config)
            apply_visual_onset_pick_times(diagnostics)
            title = f"{source_path.stem} - Block {block} - Dynamic STA/LTA & Arrival Picks"
            written.append(
                save_dynamic_pick_figure(
                    diagnostics=diagnostics,
                    output_path=output_path,
                    title=title,
                    dpi=dpi,
                    max_plot_points=max_plot_points,
                )
            )
    return written


def export_pick_diagnostic_images_for_directory(
    input_dir: Path | str,
    output_dir: Path | str,
    # Process TPC5 files that match the shared project pattern.
    pattern: str = "*.tpc5",
    config: PickConfig | None = None,
    #如果PNG已經存在，就不要重做。
    overwrite: bool = False,
    dpi: int = 150,
    max_plot_points: int = 8000,
) -> list[Path]:
    resolved_config = config or PickConfig()
    source_dir = Path(input_dir)
    written: list[Path] = []
    for file_path in sorted(source_dir.glob(pattern)):
        try:
            written.extend(
                export_pick_diagnostic_images_for_file(
                    file_path=file_path,
                    output_dir=output_dir,
                    config=resolved_config,
                    overwrite=overwrite,
                    dpi=dpi,
                    max_plot_points=max_plot_points,
                )
            )
            print(f"OK: {file_path.name}")

        except (OSError, ValueError, KeyError) as exc:
            print(f"Skip bad file: {file_path.name} -> {type(exc).__name__}: {exc}")
            continue


    return written


def process_tpc5_file(
    file_path: Path | str,
    output_csv: Path | str | None = None,
    config: PickConfig | None = None,
) -> list[dict[str, str]]:
    resolved_config = config or PickConfig()
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if not h5py.is_hdf5(str(path)):
        raise ValueError(f"{path} is not a valid HDF5/TPC5 file.")

    picks: list[dict[str, str]] = []
    with h5py.File(str(path), "r") as file_handle:
        for block in discover_valid_blocks(file_handle):
            diagnostics = analyse_block(file_handle, block, resolved_config)
            apply_visual_onset_pick_times(diagnostics)
            picks.extend(_pick_rows_for_block(block, diagnostics))

    if output_csv is not None:
        write_pick_csv(output_csv, picks)
    return picks


def _process_tpc5_file_worker(args) -> Path:
    file_path, output_csv, config = args
    process_tpc5_file(file_path=file_path, output_csv=output_csv, config=config)
    return Path(output_csv)


def process_tpc5_directory(
    input_dir: Path | str,
    output_dir: Path | str,
    pattern: str = "*.tpc5",
    config: PickConfig | None = None,
    overwrite: bool = False,
    workers: int = 1,
) -> list[Path]:
    resolved_config = config or PickConfig()
    source_dir = Path(input_dir)
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    pending: list[tuple[Path, Path, PickConfig]] = []
    written_files: list[Path] = []
    for file_path in sorted(source_dir.glob(pattern)):
        output_csv = destination_dir / f"{file_path.stem}.csv"
        if output_csv.exists() and output_csv.stat().st_size > 0 and not overwrite:
            written_files.append(output_csv)
            continue
        pending.append((file_path, output_csv, resolved_config))

    if workers <= 1 or len(pending) <= 1:
        for file_path, output_csv, config_item in pending:
            try:
                process_tpc5_file(file_path=file_path, output_csv=output_csv, config=config_item)
            except (OSError, ValueError, KeyError, FileNotFoundError) as exc:
                print(f"Skip bad file: {file_path.name} -> {type(exc).__name__}: {exc}")
                continue
            written_files.append(output_csv)
        return written_files

    max_workers = min(workers, len(pending), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_tpc5_file_worker, item): item[0]
            for item in pending
        }
        for future in as_completed(futures):
            file_path = futures[future]
            try:
                written_files.append(future.result())
            except (OSError, ValueError, KeyError, FileNotFoundError) as exc:
                print(f"Skip bad file: {Path(file_path).name} -> {type(exc).__name__}: {exc}")
    return written_files


SCRIPT_DIR = Path(__file__).resolve().parent
TPC5_DIR = SCRIPT_DIR
SAVE_DIR = SCRIPT_DIR / OUTPUT_CSV_FOLDER
TPC5_PATTERN = TPC5_FILE_PATTERN
OVERWRITE = OVERWRITE_EXISTING_CSV
WORKERS = PARALLEL_WORKERS


if __name__ == "__main__":
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print("Step 1/3: Search TPC5 files")
    print(f"  Folder : {TPC5_DIR}")
    print(f"  Pattern: {TPC5_PATTERN}")
    print("Step 2/3: Pick arrival times with Dynamic STA/LTA + AIC + visual onset")
    written_files = process_tpc5_directory(
        input_dir=TPC5_DIR,
        output_dir=SAVE_DIR,
        pattern=TPC5_PATTERN,
        config=build_pick_config(),
        overwrite=OVERWRITE,
        workers=WORKERS,
    )

    if not written_files:
        raise FileNotFoundError(f"No TPC5 files matching {TPC5_PATTERN!r} found in {TPC5_DIR}")

    print("Step 3/3: CSV files ready")
    for csv_path in written_files:
        print(f"  {csv_path}")
