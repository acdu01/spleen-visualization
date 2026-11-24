import numpy as np
import nibabel as nib
import pytest
from ct_poc.data import CaseVolume

@pytest.fixture
def case():
    shape = (7, 5, 3)
    ct = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[1:6, 1:4, 1:3] = 1
    mask[6, 4, 2] = 1
    affine = np.diag([0.7, 1.2, 5.0, 1.0])
    affine[:3, 3] = [-10, -20, 30]
    return CaseVolume("spleen_999", ct, mask, affine,
                      {"ct_sha256": "ct-fixture", "mask_sha256": "mask-fixture"})

def save_pair(root, ct, mask, affine, mask_affine=None):
    for kind, array, transform in (
        ("imagesTr", ct, affine), ("labelsTr", mask, affine if mask_affine is None else mask_affine)
    ):
        folder = root / "raw" / kind
        folder.mkdir(parents=True, exist_ok=True)
        image = nib.Nifti1Image(array, transform)
        image.header.set_xyzt_units("mm")
        image.set_qform(transform, code=1)
        image.set_sform(transform, code=1)
        nib.save(image, folder / "spleen_999.nii.gz")

@pytest.fixture
def data_root(tmp_path, case):
    save_pair(tmp_path, case.ct, case.mask, case.affine)
    return tmp_path

@pytest.fixture
def pair_writer():
    return save_pair
