from dataclasses import replace
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from concrete_Arrival_Pick import (
    TPC5_FILE_PATTERN,
    analyse_block,
    apply_visual_onset_pick_times as shared_apply_visual_onset_pick_times,
    build_pick_config,
    discover_valid_blocks,
    plot_dynamic_pick_diagnostics,
    update_pick_csv_for_block,
)


# ============================================================
# User settings
# ============================================================

# "single_plot"         : show one target file/block and optionally update its CSV.
# "export_current_file" : export all blocks in TARGET_TPC5_FILE to PNG and CSV.
# "export_all_files"    : export all matching tpc5 files to PNG and CSV.
WHAT_TO_DO = "export_current_file"

TARGET_TPC5_FILE = "data/test_2_90.tpc5"
TARGET_BLOCK = ""

# Per-event overrides for this Pick_Check script.
# Outer key: tpc5 file stem, without .tpc5.
# Second key: Block number.
# "sta_lta" : override STA/LTA threshold for specific channels.
# "arrival" : manually override arrival time for specific channels and write it to CSV.
#
# Example:
# EVENT_PICK_OVERRIDES = {
#     "test_3": {
#         33: {
#             "sta_lta": {1: 2.5, 3: 4.0},
#             "arrival": {2: 3.1004, 5: 3.1009},
#         },
#     },
# }
EVENT_PICK_OVERRIDES: dict[str, dict[int, dict[str, dict[int, float]]]] = {}

# Default pick and visual-onset parameters come from concrete_Arrival_Pick.py.
# Change them there, and Pick_Check will use the same values.


# ============================================================
# Output and plot settings
# ============================================================

SCRIPT_FOLDER = Path(__file__).resolve().parent
OUTPUT_CSV_FILE = None  # None -> csv/<same tpc5 stem>.csv
EXPORT_IMAGE_FOLDER = "pick_figures"
EXPORT_FILE_PATTERN = TPC5_FILE_PATTERN
SAVE_PICK_TO_CSV = True
SHOW_PYTHON_PLOT = True
MAX_POINTS_PER_CURVE = 8000
PLOT_ZOOM_SECONDS = None
EXPORT_OVERWRITE = True
EXPORT_DPI = 150


def resolve_target_file() -> Path:
    target_file = Path(TARGET_TPC5_FILE)
    if not target_file.is_absolute():
        target_file = SCRIPT_FOLDER / target_file
    return target_file


def resolve_csv_file(target_file: Path) -> Path:
    csv_file = (
        SCRIPT_FOLDER / "csv" / f"{target_file.stem}.csv"
        if OUTPUT_CSV_FILE is None
        else Path(OUTPUT_CSV_FILE)
    )
    if not csv_file.is_absolute():
        csv_file = SCRIPT_FOLDER / csv_file
    return csv_file


def event_override_for(target_file: Path, block: int) -> dict[str, dict[int, float]]:
    return EVENT_PICK_OVERRIDES.get(target_file.stem, {}).get(int(block), {})


def pick_config_for_event(target_file: Path, block: int):
    pick_settings = build_pick_config()
    event_thresholds = event_override_for(target_file, block).get("sta_lta", {})
    if not event_thresholds:
        return pick_settings

    levels = pick_settings.ratio_thresholds.copy()
    levels.update(
        {
            int(channel): float(threshold)
            for channel, threshold in event_thresholds.items()
        }
    )
    return replace(pick_settings, ratio_thresholds=levels)


def print_threshold_override(target_file: Path, block: int) -> None:
    event_thresholds = event_override_for(target_file, block).get("sta_lta", {})
    if event_thresholds:
        print(f"  {target_file.stem} Block {block} STA/LTA override: {event_thresholds}")


def manual_pick_times_for_event(target_file: Path, block: int) -> dict[int, float]:
    manual_picks = event_override_for(target_file, block).get("arrival", {})
    return {
        int(channel): float(time)
        for channel, time in manual_picks.items()
    }


def apply_manual_pick_times(diagnostics: dict, manual_picks: dict[int, float]) -> None:
    for channel, manual_time in manual_picks.items():
        if channel in diagnostics:
            diag = diagnostics[channel]
            diag.pick_time = float(manual_time)
            if diag.onset_pick_times:
                diag.onset_pick_times[0] = float(manual_time)
            else:
                diag.onset_pick_times = [float(manual_time)]
                diag.onset_peak_amplitudes = [float("nan")]
                diag.onset_amplitude_snrs = [float("nan")]
            diag.status = "manual_override"


def save_diagnostic_figure(
    target_file: Path,
    block: int,
    diagnostics: dict,
    block_count: int,
) -> Path:
    image_folder = SCRIPT_FOLDER / EXPORT_IMAGE_FOLDER / target_file.stem
    image_folder.mkdir(parents=True, exist_ok=True)
    output_path = image_folder / f"block_{block:06d}_dynamic_sta_lta_arrival.png"
    if output_path.exists() and not EXPORT_OVERWRITE:
        return output_path

    fig = plot_dynamic_pick_diagnostics(
        diagnostics,
        title=f"{target_file.stem} - Block {block} - Dynamic STA/LTA & Arrival Picks",
        max_plot_points=MAX_POINTS_PER_CURVE,
    )
    fig.savefig(output_path, dpi=EXPORT_DPI)
    plt.close(fig)
    return output_path


def repick_file(target_file: Path, save_figures: bool = True) -> tuple[int, list[Path]]:
    if not target_file.exists():
        raise FileNotFoundError(target_file)

    csv_file = resolve_csv_file(target_file)
    saved_figures: list[Path] = []
    updated_blocks = 0

    with h5py.File(str(target_file), "r") as file_handle:
        valid_blocks = discover_valid_blocks(file_handle)
        for block in valid_blocks:
            pick_settings = pick_config_for_event(target_file, block)
            print_threshold_override(target_file, block)
            diagnostics = analyse_block(file_handle, block, pick_settings)
            detected_onsets = shared_apply_visual_onset_pick_times(diagnostics)
            if detected_onsets:
                counts = [len(onset.channel_times) for onset in detected_onsets]
                labels = [f"{block}-{index}" for index in range(1, len(detected_onsets) + 1)]
                print(f"  Qualified blocks: {labels}, channels={counts}")

            manual_picks = manual_pick_times_for_event(target_file, block)
            if manual_picks:
                print(f"  Manual arrival override: {manual_picks}")
                apply_manual_pick_times(diagnostics, manual_picks)

            if SAVE_PICK_TO_CSV:
                update_pick_csv_for_block(csv_file, block, diagnostics)
                updated_blocks += 1

            if save_figures:
                saved_figures.append(
                    save_diagnostic_figure(
                        target_file=target_file,
                        block=block,
                        diagnostics=diagnostics,
                        block_count=len(valid_blocks),
                    )
                )

    if SAVE_PICK_TO_CSV:
        print(f"  Updated CSV: {csv_file}")
    return updated_blocks, saved_figures


def export_current_file(target_file: Path) -> None:
    print("Export current tpc5 pick figures...")
    updated_blocks, written_paths = repick_file(target_file, save_figures=True)
    print(f"Done. Updated {updated_blocks} blocks.")
    print(f"Done. Saved/kept {len(written_paths)} figures under: {SCRIPT_FOLDER / EXPORT_IMAGE_FOLDER}")


def export_all_files() -> None:
    print("Export all tpc5 pick figures...")
    total_blocks = 0
    written_paths: list[Path] = []
    for target_file in sorted(SCRIPT_FOLDER.glob(EXPORT_FILE_PATTERN)):
        try:
            print(f"Process {target_file.name}")
            updated_blocks, saved_figures = repick_file(target_file, save_figures=True)
        except (OSError, ValueError, KeyError, FileNotFoundError) as exc:
            print(f"Skip bad file: {target_file.name} -> {type(exc).__name__}: {exc}")
            continue
        total_blocks += updated_blocks
        written_paths.extend(saved_figures)
    print(f"Done. Updated {total_blocks} blocks.")
    print(f"Done. Saved/kept {len(written_paths)} figures under: {SCRIPT_FOLDER / EXPORT_IMAGE_FOLDER}")


def show_single_block(target_file: Path) -> None:
    if not target_file.exists():
        raise FileNotFoundError(target_file)

    pick_settings = pick_config_for_event(target_file, TARGET_BLOCK)
    csv_file = resolve_csv_file(target_file)

    print("Read tpc5 file...")
    with h5py.File(str(target_file), "r") as file_handle:
        valid_blocks = discover_valid_blocks(file_handle)
        if TARGET_BLOCK not in valid_blocks:
            raise ValueError(f"Block {TARGET_BLOCK} not found. Valid blocks: {valid_blocks[:10]}...")

        print(f"Found {len(valid_blocks)} blocks")
        print("Run Dynamic STA/LTA + AIC + visual onset pick...")
        print_threshold_override(target_file, TARGET_BLOCK)
        diagnostics = analyse_block(file_handle, TARGET_BLOCK, pick_settings)
        detected_onsets = shared_apply_visual_onset_pick_times(diagnostics)
        if detected_onsets:
            counts = [len(onset.channel_times) for onset in detected_onsets]
            labels = [
                f"{TARGET_BLOCK}-{index}"
                for index in range(1, len(detected_onsets) + 1)
            ]
            print(f"  Qualified blocks: {labels}, channels={counts}")

    manual_picks = manual_pick_times_for_event(target_file, TARGET_BLOCK)
    if manual_picks:
        print(f"  Manual arrival override: {manual_picks}")
        apply_manual_pick_times(diagnostics, manual_picks)

    print("Pick result:")
    for channel in pick_settings.channels:
        diag = diagnostics[channel]
        pick_text = ", ".join(
            f"Block {TARGET_BLOCK}-{index}={time:.9f}s"
            for index, time in enumerate(diag.onset_pick_times, start=1)
            if np.isfinite(time)
        ) or "none"
        print(f"  Ch{channel}: {pick_text} ({diag.status})")

    if SAVE_PICK_TO_CSV:
        update_pick_csv_for_block(csv_file, TARGET_BLOCK, diagnostics)
        print(f"Updated CSV: {csv_file}")

    if SHOW_PYTHON_PLOT:
        fig = plot_dynamic_pick_diagnostics(
            diagnostics,
            title=f"{target_file.stem} - Block {TARGET_BLOCK} - Dynamic STA/LTA & Arrival Picks",
            max_plot_points=MAX_POINTS_PER_CURVE,
        )
        plt.show()


def main() -> None:
    if WHAT_TO_DO not in {"single_plot", "export_current_file", "export_all_files"}:
        raise ValueError("WHAT_TO_DO must be single_plot, export_current_file, or export_all_files")

    target_file = resolve_target_file()

    print("Arrival pick check")
    print(f"  Mode : {WHAT_TO_DO}")
    print(f"  File : {target_file}")
    print(f"  Block: {TARGET_BLOCK}")

    if WHAT_TO_DO == "export_current_file":
        export_current_file(target_file)
        return

    if WHAT_TO_DO == "export_all_files":
        export_all_files()
        return

    show_single_block(target_file)


if __name__ == "__main__":
    main()
