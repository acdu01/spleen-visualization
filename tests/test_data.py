import nibabel as nib
import numpy as np
import pytest
from ct_poc.data import load_case

def test_real_nifti_roundtrip(data_root, case):
    loaded = load_case(data_root, "spleen_999")
    np.testing.assert_array_equal(loaded.ct, case.ct)
    np.testing.assert_array_equal(loaded.mask, case.mask)
    assert loaded.volume_ml == pytest.approx(case.volume_ml)
    assert loaded.ct.dtype == np.float32
    assert loaded.mask.dtype == np.uint8
    assert not loaded.ct.flags.writeable

def test_reorient_preserves_alignment(tmp_path, case, pair_writer):
    affine = case.affine.copy()
    affine[0, 0] *= -1
    pair_writer(tmp_path, case.ct, case.mask, affine)
    loaded = load_case(tmp_path, "spleen_999")
    np.testing.assert_array_equal(loaded.ct, case.ct[::-1])
    np.testing.assert_array_equal(loaded.mask, case.mask[::-1])
    assert loaded.spacing[0] > 0
    assert loaded.volume_ml == pytest.approx(case.volume_ml)

def test_misaligned_pair_is_rejected(tmp_path, case, pair_writer):
    shifted = case.affine.copy()
    shifted[0, 3] += 2
    pair_writer(tmp_path, case.ct, case.mask, case.affine, shifted)
    with pytest.raises(ValueError, match="geometry"):
        load_case(tmp_path, "spleen_999")

def test_unknown_label_rejected(tmp_path, case, pair_writer):
    labels = case.mask.copy()
    labels[0, 0, 0] = 2
    pair_writer(tmp_path, case.ct, labels, case.affine)
    with pytest.raises(ValueError, match="only background"):
        load_case(tmp_path, "spleen_999")

def test_conflicting_header_rejected(data_root):
    path = data_root / "raw/imagesTr/spleen_999.nii.gz"
    image = nib.load(path)
    moved = image.affine.copy()
    moved[0, 3] += 10
    image.set_qform(moved, code=1)
    nib.save(image, path)
    with pytest.raises(ValueError, match="qform"):
        load_case(data_root, "spleen_999")
