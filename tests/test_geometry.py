from dataclasses import replace
import numpy as np
import pytest
from ct_poc.grids import make_grid, save_grid, load_grid
from ct_poc.metrics import compare

@pytest.mark.parametrize("factor", [1, 2, 4, 10])
def test_partial_edges_preserve_physical_volume(case, factor):
    grid = make_grid(case, factor)
    assert int(grid.counts.sum()) == case.ct.size
    assert np.sum(grid.cell_mm3) == pytest.approx(case.ct.size * case.voxel_mm3)
    row = compare(case, grid)
    assert row["fractional_ml"] == pytest.approx(case.volume_ml, rel=1e-6)
    for axis, edges in enumerate(grid.edges):
        assert edges[0] == pytest.approx(case.affine[axis, 3] - case.spacing[axis] / 2)
        assert edges[-1] == pytest.approx(case.affine[axis, 3] + (case.ct.shape[axis] - .5) * case.spacing[axis])
    weighted_mean = np.sum(grid.ct.astype(float) * grid.counts) / grid.counts.sum()
    assert weighted_mean == pytest.approx(case.ct.mean(), rel=1e-6)

def test_identity(case):
    grid = make_grid(case, 1)
    np.testing.assert_array_equal(grid.ct, case.ct)
    np.testing.assert_array_equal(grid.hard_mask, case.mask)
    row = compare(case, grid)
    assert row["dice"] == 1
    assert row["volume_change_percent"] == 0
    assert row["lost_voxels"] == row["gained_voxels"] == 0

@pytest.mark.parametrize("factor", [2, 4])
def test_dice_matches_explicit_native_grid(case, factor):
    grid = make_grid(case, factor)
    expanded = grid.hard_mask
    for axis in range(3):
        expanded = np.repeat(expanded, factor, axis=axis)
    expanded = expanded[tuple(slice(0, n) for n in case.mask.shape)]
    intersection = np.count_nonzero(expanded & case.mask)
    expected = 2 * intersection / (np.count_nonzero(expanded) + np.count_nonzero(case.mask))
    assert compare(case, grid)["dice"] == pytest.approx(expected)

def test_empty_reference(case):
    empty = replace(case, mask=np.zeros_like(case.mask))
    row = compare(empty, make_grid(empty, 4))
    assert row["volume_change_percent"] is None
    assert row["dice"] == 1
    assert row["fractional_ml"] == 0

def test_cache_rejects_changed_source(tmp_path, case):
    path = tmp_path / "grid.npz"
    grid = make_grid(case, 4)
    save_grid(path, grid, case)
    loaded = load_grid(path, case, 4)
    np.testing.assert_array_equal(loaded.ct, grid.ct)
    changed = replace(case, provenance={**case.provenance, "ct_sha256": "changed"})
    with pytest.raises(ValueError, match="stale"):
        load_grid(path, changed, 4)

@pytest.mark.parametrize("factor", [0, -1, 1.5, True])
def test_invalid_factor(case, factor):
    with pytest.raises(ValueError):
        make_grid(case, factor)
