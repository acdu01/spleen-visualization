"""Exact block aggregation, including smaller blocks at the field edges."""
from dataclasses import dataclass
from pathlib import Path
import json
import os
import tempfile
import numpy as np
from .data import CaseVolume

ALGORITHM_VERSION = 1

@dataclass(frozen=True)
class GridLevel:
    factor: int
    ct: np.ndarray
    fraction: np.ndarray
    hard_mask: np.ndarray
    counts: np.ndarray
    edges: tuple
    voxel_mm3: float
    threshold: float = 0.5

    @property
    def cell_mm3(self):
        return self.counts * self.voxel_mm3

    @property
    def nbytes(self):
        return sum(a.nbytes for a in (self.ct, self.fraction, self.hard_mask, self.counts, *self.edges))

def block_sum(array, factor, dtype=np.float64):
    result = np.asarray(array, dtype=dtype)
    for axis in range(3):
        result = np.add.reduceat(result, np.arange(0, result.shape[axis], factor), axis=axis)
    return result

def make_grid(case: CaseVolume, factor: int, threshold=0.5):
    if not isinstance(factor, int) or isinstance(factor, bool) or factor < 1:
        raise ValueError("Factor must be a positive integer.")
    if not 0 < threshold <= 1:
        raise ValueError("Threshold must be in (0, 1].")
    starts = [np.arange(0, n, factor) for n in case.ct.shape]
    lengths = [np.minimum(factor, n - a) for n, a in zip(case.ct.shape, starts)]
    counts = (lengths[0][:, None, None] * lengths[1][None, :, None] * lengths[2][None, None, :]).astype(np.uint32)
    ct = (block_sum(case.ct, factor) / counts).astype(np.float32)
    fraction = (block_sum(case.mask, factor, np.uint64) / counts).astype(np.float32)
    hard = (fraction >= threshold).astype(np.uint8)
    edges = tuple((np.r_[a, n] - 0.5) * spacing + origin
                  for a, n, spacing, origin in zip(starts, case.ct.shape, case.spacing, case.affine[:3, 3]))
    for array in (ct, fraction, hard, counts, *edges):
        array.setflags(write=False)
    return GridLevel(factor, ct, fraction, hard, counts, edges, case.voxel_mm3, threshold)

def grid_path(root, case_id, factor):
    return Path(root) / "prepared" / case_id / f"scale-{factor}.npz"

def save_grid(path, grid, case):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"algorithm": ALGORITHM_VERSION, "case_id": case.case_id,
                "factor": grid.factor, "threshold": grid.threshold,
                "voxel_mm3": grid.voxel_mm3, "source": case.provenance}
    handle, temporary = tempfile.mkstemp(prefix=".grid-", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez_compressed(stream, ct=grid.ct, fraction=grid.fraction,
                hard_mask=grid.hard_mask, counts=grid.counts,
                edge0=grid.edges[0], edge1=grid.edges[1], edge2=grid.edges[2],
                metadata=json.dumps(metadata))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def load_grid(path, case, factor):
    with np.load(path, allow_pickle=False) as bundle:
        metadata = json.loads(str(bundle["metadata"]))
        source = metadata.get("source", {})
        if (metadata.get("algorithm") != ALGORITHM_VERSION
                or metadata.get("case_id") != case.case_id
                or metadata.get("factor") != factor
                or metadata.get("threshold") != 0.5
                or any(source.get(k) != case.provenance[k] for k in ("ct_sha256", "mask_sha256"))):
            raise ValueError("Prepared grid is stale. Run prepare again.")
        arrays = [bundle[k] for k in ("ct", "fraction", "hard_mask", "counts", "edge0", "edge1", "edge2")]
    expected = tuple((n + factor - 1) // factor for n in case.ct.shape)
    if any(a.shape != expected for a in arrays[:4]) or any(len(e) != n + 1 for e, n in zip(arrays[4:], expected)):
        raise ValueError("Prepared grid has invalid dimensions.")
    for array in arrays:
        array.setflags(write=False)
    return GridLevel(factor, *arrays[:4], tuple(arrays[4:]), metadata["voxel_mm3"])
