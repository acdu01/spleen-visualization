"""Measurements compare representations of the supplied reference labels."""
import numpy as np
from .grids import block_sum

def compare(case, grid):
    positive = block_sum(case.mask, grid.factor, np.uint64)
    truth = int(np.count_nonzero(case.mask))
    predicted = int(np.sum(grid.counts * grid.hard_mask, dtype=np.uint64))
    intersection = int(np.sum(positive * grid.hard_mask, dtype=np.uint64))
    native_ml = truth * case.voxel_mm3 / 1000
    hard_ml = predicted * case.voxel_mm3 / 1000
    fractional_ml = float(np.sum(grid.fraction.astype(np.float64) * grid.cell_mm3) / 1000)
    return {
        "case_id": case.case_id, "factor": grid.factor, "threshold": grid.threshold,
        "reference_ml": native_ml, "fractional_ml": fractional_ml, "hard_ml": hard_ml,
        "volume_change_percent": (hard_ml / native_ml - 1) * 100 if truth else None,
        "dice": 2 * intersection / (truth + predicted) if truth + predicted else 1.0,
        "lost_voxels": truth - intersection, "gained_voxels": predicted - intersection,
        "native_array_mib": (case.ct.nbytes + case.mask.nbytes) / 2**20,
        "grid_array_mib": grid.nbytes / 2**20,
    }
