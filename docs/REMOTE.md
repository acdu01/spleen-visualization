# Optional remote setup

The app works locally without these steps.
For remote use, install the same Python environment there.
Keep personal commands and job settings in the ignored private folder.
Private keys belong on your own computer, never the remote server.

## A remote machine

Run from the remote project root:

```bash
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

On your own computer, replace USER and HOST:

```bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8501:127.0.0.1:8501 USER@HOST
```

Open [the viewer](http://127.0.0.1:8501) on your own computer.
Keep both terminals open. Ctrl+C stops each process.

## A Slurm cluster

Run preparation and viewing on allocated compute nodes.
Follow your cluster's account, partition, storage, and access rules.
The examples choose neither an account nor a partition.

From the remote project root, replace ACCOUNT and PARTITION:

```bash
mkdir -p logs
sbatch --account=ACCOUNT --partition=PARTITION scripts/slurm/prepare.sbatch
sbatch --account=ACCOUNT --partition=PARTITION scripts/slurm/viewer.sbatch
```

Preparation handles every downloaded scan.
The viewer log prints its compute node and port.
Read logs/viewer-JOB_ID.log using the returned job number.

If a login host is required, replace USER, LOGIN_HOST,
and COMPUTE_NODE in this laptop command:

```bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8501:127.0.0.1:8501 -J USER@LOGIN_HOST USER@COMPUTE_NODE
```

The tunnel must end on the node running the viewer.
Both SSH hops need working access from your computer.
Use your existing SSH config for keys or special steps.
Keep host verification on and the viewer port private.

Stop the viewer job when done:

```bash
scancel JOB_ID
```

The supplied viewer job also stops at its time limit.

## Settings

| Setting | Default |
| --- | --- |
| SPLEEN_DATA_DIR | Project data folder |
| SPLEEN_PYTHON | .venv/bin/python |
| SPLEEN_PORT | 8501 |
| Preparation CPUs | 4 |
| Preparation RAM | 16 GiB |
| Viewer CPUs | 2 |
| Viewer RAM | 8 GiB |

SPLEEN_PYTHON and SPLEEN_PORT apply to the sample job files.
Set them in your shell before submitting a job.
Ensure your scheduler forwards those variables.
If SPLEEN_PORT changes, change both tunnel port numbers too.
Adjust resources and time limits for your cluster.
Keep personal variants under private/ instead of editing shared defaults.
