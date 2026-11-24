"""Validated native volumes. All spatial quantities use millimeters."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import re
import nibabel as nib
import numpy as np

@dataclass(frozen=True)
class CaseVolume:
    case_id: str
    ct: np.ndarray
    mask: np.ndarray
    affine: np.ndarray
    provenance: dict

    @property
    def spacing(self):
        return np.linalg.norm(self.affine[:3, :3], axis=0)

    @property
    def voxel_mm3(self):
        return float(abs(np.linalg.det(self.affine[:3, :3])))

    @property
    def volume_ml(self):
        return float(np.count_nonzero(self.mask) * self.voxel_mm3 / 1000)

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def case_paths(root, case_id):
    if not re.fullmatch(r"spleen_[0-9]+", case_id):
        raise ValueError("Case IDs must look like spleen_3.")
    raw = Path(root) / "raw"
    return raw / "imagesTr" / (case_id + ".nii.gz"), raw / "labelsTr" / (case_id + ".nii.gz")

def available_cases(root):
    folder = Path(root) / "raw" / "imagesTr"
    return sorted((p.name[:-7] for p in folder.glob("spleen_*.nii.gz")
                   if (Path(root) / "raw" / "labelsTr" / p.name).is_file()),
                  key=lambda name: int(name.split("_")[-1]))

def _check_header(image):
    if len(image.shape) != 3 or any(n < 1 for n in image.shape):
        raise ValueError("Expected a nonempty 3D image.")
    if image.header.get_xyzt_units()[0] != "mm":
        raise ValueError("Spatial units must be millimeters.")
    qform, qc = image.get_qform(coded=True)
    sform, sc = image.get_sform(coded=True)
    if qc and sc and not np.allclose(qform, sform, atol=1e-3, rtol=0):
        raise ValueError("NIfTI qform and sform disagree.")
    if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-9:
        raise ValueError("Invalid image coordinate transform.")

def load_case(root, case_id):
    image_path, mask_path = case_paths(root, case_id)
    if not image_path.is_file() or not mask_path.is_file():
        raise FileNotFoundError(f"Missing scan/mask pair for {case_id}. Run the fetch command.")
    image, labels = nib.load(image_path), nib.load(mask_path)
    _check_header(image)
    _check_header(labels)
    if image.shape != labels.shape or not np.allclose(image.affine, labels.affine, atol=1e-4, rtol=0):
        raise ValueError("Scan and mask geometry do not match; refusing to overlay them.")
    image, labels = nib.as_closest_canonical(image), nib.as_closest_canonical(labels)
    spacing = np.linalg.norm(image.affine[:3, :3], axis=0)
    if not np.allclose(image.affine[:3, :3], np.diag(spacing), atol=1e-4, rtol=0):
        raise ValueError("Tilted or sheared grids are not supported by this viewer.")
    ct = image.get_fdata(dtype=np.float32, caching="unchanged")
    raw_mask = np.asarray(labels.dataobj)
    if not np.isin(raw_mask, [0, 1]).all():
        raise ValueError("Expected only background=0 and spleen=1 labels.")
    if not np.isfinite(ct).all():
        raise ValueError("CT contains non-finite values.")
    mask = raw_mask.astype(np.uint8, copy=False)
    ct.setflags(write=False)
    mask.setflags(write=False)
    return CaseVolume(case_id, ct, mask, image.affine.copy(), {
        "ct_sha256": sha256(image_path), "mask_sha256": sha256(mask_path),
        "ct_path": str(image_path), "mask_path": str(mask_path),
        "orientation": "RAS", "units": "mm",
    })
