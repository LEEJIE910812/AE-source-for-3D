# AE source for 3D

This repository publishes interactive 3D acoustic emission (AE) source location viewers for concrete experiments.

Website:

https://leejie910812.github.io/AE-source-for-3D/

## What Is Inside

- `index.html`  
  Main website page. It links to each experiment and each color mode.

- `viewers/`  
  Interactive Plotly HTML viewers. Each test usually has two versions:
  - `time`: AE points colored by event time.
  - `depth`: AE points colored by radial depth from the specimen surface.

- `code/t0208/`  
  Example Python source code for one experiment. It shows the full workflow from waveform arrival picking to AE source location and interactive viewer export.

- `requirements.txt`  
  Main Python packages used by the scripts.

## Example Code Workflow

The included example uses the `t0208` experiment. The workflow is:

1. Run `concrete_Arrival_Pick.py`  
   Reads `.tpc5` waveform files and outputs arrival pick CSV files.

2. Run `concrete_Arrival_Pick_Check.py`  
   Checks or manually adjusts arrival picks block by block.

3. Run `concrete_Calculate.py`  
   Reads the adjusted CSV files and calculates AE source locations using grid search, TRF refinement, and HypoDD-style relative relocation.

4. Run `concrete_Figure_Output.py`  
   Reads the calculated `.pkl` result files and exports interactive 3D HTML viewers.

5. Run `concrete_Anime.py` if MP4 animation output is needed.

6. Run `concrete_event_vs_time.py` if event count versus time plots are needed.

## Example Code Files

- `code/t0208/concrete_Arrival_Pick.py`  
  Automatically detects arrival times from `.tpc5` waveform data and writes pick CSV files.

- `code/t0208/concrete_Arrival_Pick_Check.py`  
  Opens block-by-block diagnostic plots for checking and manually adjusting arrival picks.

- `code/t0208/concrete_Calculate.py`  
  Calculates AE source locations from the adjusted CSV files.

- `code/t0208/concrete_Figure_Output.py`  
  Exports the interactive 3D HTML viewers used on this website.

- `code/t0208/concrete_Anime.py`  
  Creates rotating MP4 animations from the calculated location results.

- `code/t0208/concrete_event_vs_time.py`  
  Plots AE event count and event rate versus time.

- `code/t0208/concrete_common.py` and `code/t0208/tpc5.py`  
  Shared helper code used by the other scripts.

## Viewer Controls

The interactive HTML viewers include:

- Time slider.
- Play / stop animation buttons.
- Rotation button.
- Legend toggles for AE points, sensors, surface crack network, internal crack network, and PCA plane.
- A `隱藏` button that hides AE points that are both near the specimen edge and have high velocity residual.

The `隱藏` button does not delete data. It only changes what is displayed in the browser.

## Notes

Raw `.tpc5`, `.csv`, and `.pkl` data files are not included here because they can be large and experiment-specific.

To rerun the example analysis, put the needed `t0208` data files next to the scripts in `code/t0208/`, then adjust the settings near the top of each Python file.
