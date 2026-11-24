# Spleen · Resolution Lab

Explore how larger voxels change a labelled CT scan.
Compare the original scan with 2× and 4× coarser grids.
View slices, spleen labels, shape changes, and memory use.
Export figures as PNG files and measurements as CSV files.

Runs on your own computer. No cluster, SSH, or GPU required.

## Quick start

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
Windows and macOS use the same Python app.

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

Data defaults to the data folder beside the source code.
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

That flag does not change the viewer's data folder.
Use SPLEEN_DATA_DIR when both should share the same location.

## Checks

With the virtual environment active:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

Tests use small made-up scans and need no dataset download.
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

This experiment compares representations of the supplied labels.
It does not train a model or establish clinical accuracy.
Background includes other tissue; it does not mean empty space.

See [the architecture](docs/ARCHITECTURE.md) for how measurements work.
See [remote setup](docs/REMOTE.md) to run on another machine.
