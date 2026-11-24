# Spleen · Resolution Lab

A portable CT viewer and CPU baseline for **spleen presence per slice**.

Explore how larger voxels change a labelled CT scan.
Compare the original scan with 2× and 4× coarser grids.
View slices, spleen labels, shape changes, and memory use.
Export figures as PNG files and measurements as CSV files.

Runs on your own computer. No cluster, SSH, or GPU required.

## Train the baseline without downloading CT data

The demo makes small synthetic volumes. It checks setup, not scientific accuracy.
Install Python 3.12 first. Clone the repository and enter its root folder.
The prediction workflow needs the optional ML dependencies.

### macOS or Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[ml]"
.venv/bin/python -m ct_poc.cli --data-dir data/demo demo
.venv/bin/python -m ct_poc.cli --data-dir data/demo train --output outputs/demo
.venv/bin/python -m ct_poc.cli --data-dir data/demo view --results-dir outputs/demo
```

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[ml]"
.\.venv\Scripts\python.exe -m ct_poc.cli --data-dir data/demo demo
.\.venv\Scripts\python.exe -m ct_poc.cli --data-dir data/demo train --output outputs/demo
.\.venv\Scripts\python.exe -m ct_poc.cli --data-dir data/demo view --results-dir outputs/demo
```

Open [the viewer](http://127.0.0.1:8501), then expand **Slice prediction baseline**.
That section shows held-out results and each slice's score alongside its target.
The overlays in the main viewer remain the supplied labels, not predictions.
Keep the terminal open; Ctrl+C stops the viewer.

Use a new results folder for each training run. Demo creation also refuses to
overwrite an existing data folder. After the first run, reuse the last command
to reopen the viewer without recreating data or retraining.

The CLI `view` command also works after a non-editable package installation.
No private folder, account name, environment file, SSH key or cluster is needed.

## View one real CT scan

Use Python 3.12 and an internet connection for the first download.
Download or clone this repository, then open its root folder.
These commands fetch one scan, prepare it, and start the viewer.

### macOS or Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m ct_poc.cli fetch --cases spleen_3
.venv/bin/python -m ct_poc.cli prepare --cases spleen_3 --workers 1
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1
```

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m ct_poc.cli fetch --cases spleen_3
.\.venv\Scripts\python.exe -m ct_poc.cli prepare --cases spleen_3 --workers 1
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Open [the viewer](http://127.0.0.1:8501) on the same computer.
Keep the terminal open. Press Ctrl+C to stop the viewer.
The server listens only on your computer's loopback address.

The first scan and mask download is about 11.4 MB.
Preparation writes extra files under the data folder.
Start with one worker; more workers use more RAM.
The project has been checked on Linux with Python 3.12.
Windows and macOS use the same Python app; those platforms are not yet verified.

## Read the viewer

| Panel | Shows |
| --- | --- |
| Original | Scan and supplied spleen labels |
| Coarse | Larger cells and their labels |
| Difference | Lost labels in red; added labels in blue |

Red overlays on the first two panels mark the spleen.
Each panel shows the same place in the body.
Try spleen_3, Axial, slice 26, then change the grid scale.

The numbers above the panels measure the whole 3D spleen.
Dice measures label overlap; 1 means a perfect match.
Cell fractions show how much spleen each block contained.
They are not model confidence.

## Data location and larger runs

Data defaults to the data folder in your current working directory.
Set SPLEEN_DATA_DIR to use another disk or folder.
Use the same setting for fetching, preparation, and viewing.

macOS or Linux:

```bash
export SPLEEN_DATA_DIR="/path/to/spleen-data"
```

Windows PowerShell:

```powershell
$env:SPLEEN_DATA_DIR = "D:\spleen-data"
```

These next commands use an activated virtual environment.
On macOS or Linux, run `source .venv/bin/activate`.
On Windows, run `.\.venv\Scripts\Activate.ps1`.
Or use the full Python path from the quick start.

```bash
python -m ct_poc.cli fetch --cases all
python -m ct_poc.cli prepare --cases all --factors 1 2 4 --workers 1
python -m ct_poc.cli inspect spleen_3
```

All 41 labelled scans and prepared grids used about 2.4 GiB
in the initial run. Other settings can change storage needs.
Increase --workers only when CPU and RAM allow it.

For one CLI command, --data-dir goes before the command:

```bash
python -m ct_poc.cli --data-dir /path/to/spleen-data inspect spleen_3
```

The CLI view command honors --data-dir too.
For a separate streamlit run, set SPLEEN_DATA_DIR to the same folder.

## Train on real CT scans

With the environment active, install the ML extra and fetch labelled scans:

```bash
python -m pip install -e ".[ml]"
python -m ct_poc.cli fetch --cases all
python -m ct_poc.cli train --cases all --output outputs/baseline
python -m ct_poc.cli view --results-dir outputs/baseline
```

Training processes one volume at a time on CPU; no prepared viewer grids are
required. Feature caches and result files are reused independently of the viewer.
Allow several GiB of RAM for the larger scans. A smaller initial real-data run:

```bash
python -m ct_poc.cli fetch --cases spleen_3 spleen_6 spleen_8 spleen_9 spleen_10
python -m ct_poc.cli train --cases spleen_3 spleen_6 spleen_8 spleen_9 spleen_10 --output outputs/small-baseline
```

Five cases suffice to exercise the workflow, not to establish accuracy.
Use all available cases for the exploratory comparison. Splits use case IDs;
patient identity is not independently verified. Labels never enter model inputs.

Run saved-model inference on a CT image without providing a mask:

```bash
python -m ct_poc.cli predict --image data/raw/imagesTr/spleen_3.nii.gz --model outputs/baseline/model.npz --output outputs/spleen-3-predictions.csv
```

Use `--data-dir` for data and `--output` for training results.
The viewer accepts `--results-dir`, or the SPLEEN_RESULTS_DIR environment variable.
Results include a readable report, model, split manifest, metrics and slice scores.
See [the prediction protocol](docs/PREDICTION.md) before interpreting accuracy.

## Checks

With the virtual environment active:

```bash
python -m pip install -e ".[ml,test]"
python -m pytest -q
```

Tests use small made-up scans and need no dataset download.
For the tested Linux/Python 3.12 direct-dependency versions, install with:

```bash
python -m pip install -c requirements-tested.txt -e ".[ml,test]"
```

These constraints record the tested environment, not a cross-platform lockfile.
For the optional real-data check, fetch and prepare both cases:

```bash
python -m ct_poc.cli fetch --cases spleen_3 spleen_19
python -m ct_poc.cli prepare --cases spleen_3 spleen_19 --workers 1
python scripts/smoke.py
```

The smoke check writes results under outputs/.
It briefly uses local port 18501, then stops its server.

## Project map

| Path | Purpose |
| --- | --- |
| app.py | Viewer entry point |
| ct_poc/ | Loading, grids, measurements, and UI |
| ct_poc/prediction/ | CT features, model, experiments, and results panel |
| docs/PREDICTION.md | Split protocol, baseline, metrics, and limitations |
| tests/ | Numerical and viewer checks |
| scripts/smoke.py | Optional real-data check |
| scripts/slurm/ | Generic cluster job examples |
| docs/ARCHITECTURE.md | Data flow and numerical rules |
| docs/REMOTE.md | Optional remote and Slurm setup |
| private/ | Ignored personal notes and launch files |
| data/, outputs/, logs/ | Ignored local data and results |

The private folder is optional and absent from public clones.
Git ignores it, and source packages exclude it.
Keep account names, key paths, and personal job scripts there.
Ignored files are not encrypted. Never force-add private files.

## Data and scope

The data is Medical Segmentation Decathlon Task09 Spleen.
It was provided by Memorial Sloan Kettering Cancer Center.
Dataset licence: CC-BY-SA 4.0.

- [Official dataset project](https://medicaldecathlon.com/)
- [Source archive](https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar)

Keep attribution and licence details with shared data derivatives.
This licence describes the dataset, not the application code.

The resolution viewer compares representations of supplied labels.
The optional baseline predicts spleen presence in axial slices.
Neither establishes clinical accuracy.
Background includes other tissue; it does not mean empty space.

See [the architecture](docs/ARCHITECTURE.md) for how measurements work.
See [remote setup](docs/REMOTE.md) to run on another machine.
