"""Small, explicitly synthetic volumes for a portable, offline smoke run."""
from pathlib import Path
import nibabel as nib
import numpy as np
from .acquire import write_json

def create_demo(root, cases=12, seed=42):
    root = Path(root)
    if cases < 5:
        raise ValueError("The demo needs at least five cases.")
    if root.exists() and any(root.iterdir()):
        raise ValueError("Demo destination must be empty. Use a new --data-dir.")
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    shape = (48, 48, 24)
    x, y, z = np.indices(shape, dtype=np.float32)
    names = []
    for index in range(cases):
        name = f"spleen_{1000 + index}"
        names.append(name)
        body = ((x - 24) / 20) ** 2 + ((y - 24) / 17) ** 2 < 1
        cx, cy, cz = 34 + rng.uniform(-2, 2), 25 + rng.uniform(-2, 2), 12 + rng.uniform(-2, 2)
        mask = (((x - cx) / 5) ** 2 + ((y - cy) / 6) ** 2 +
                ((z - cz) / rng.uniform(4, 7)) ** 2 < 1)
        ct = np.where(body, 30.0, -950.0) + rng.normal(0, 10, shape)
        ct[mask] += 95 + rng.uniform(-10, 10)
        affine = np.diag([1.5, 1.5, 3.0, 1.0])
        for kind, array in (("imagesTr", ct.astype(np.float32)), ("labelsTr", mask.astype(np.uint8))):
            path = root / "raw" / kind / f"{name}.nii.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            image = nib.Nifti1Image(array, affine)
            image.header.set_xyzt_units("mm")
            image.set_qform(affine, code=1)
            image.set_sform(affine, code=1)
            nib.save(image, path)
    write_json(root / "synthetic.json", {"synthetic": True, "seed": seed, "cases": names,
        "note": "Made-up shapes for software checks. Not human scans or research evidence."})
    return names
