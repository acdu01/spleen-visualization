"""CT-only features. Masks are used solely to construct reference targets."""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import os
import tempfile
import numpy as np
from ..data import case_paths, load_case, sha256
from ..grids import block_sum

FEATURE_VERSION = 1

@dataclass(frozen=True)
class FeatureConfig:
    tiles: int = 12
    bins: int = 16
    hu_low: float = -200.0
    hu_high: float = 300.0

    def __post_init__(self):
        if not 1 <= self.tiles <= 32 or not 2 <= self.bins <= 64:
            raise ValueError("Feature tiles must be 1..32 and bins 2..64.")
        if not np.isfinite([self.hu_low, self.hu_high]).all() or self.hu_low >= self.hu_high:
            raise ValueError("Feature window must be finite and increasing.")

    @property
    def size(self):
        return 2 * self.tiles ** 2 + self.bins + 2

def validate_factor(factor):
    if not isinstance(factor, int) or isinstance(factor, bool) or factor < 1:
        raise ValueError("Resolution factor must be a positive integer.")

def coarse_ct(ct, factor):
    validate_factor(factor)
    if factor == 1:
        return ct
    lengths = [np.minimum(factor, n - np.arange(0, n, factor)) for n in ct.shape]
    counts = lengths[0][:, None, None] * lengths[1][None, :, None] * lengths[2][None, None, :]
    return (block_sum(ct, factor) / counts).astype(np.float32)

def slice_features(image, config):
    """Fixed field-of-view tiles, histogram and gradients; no label/position input."""
    image = np.clip((image - config.hu_low) / (config.hu_high - config.hu_low), 0, 1)
    # Tiny test images are repeated, without interpolation or padding.
    repeats = [max(1, int(np.ceil(config.tiles / n))) for n in image.shape]
    if repeats != [1, 1]:
        image = np.repeat(np.repeat(image, repeats[0], axis=0), repeats[1], axis=1)
    edges = [np.linspace(0, n, config.tiles + 1, dtype=int) for n in image.shape]
    features = []
    for a, b in zip(edges[0][:-1], edges[0][1:]):
        for c, d in zip(edges[1][:-1], edges[1][1:]):
            tile = image[a:b, c:d]
            features.extend((float(tile.mean()), float(tile.std())))
    hist = np.histogram(image, bins=config.bins, range=(0, 1))[0] / image.size
    gradients = [float(np.abs(np.diff(image, axis=axis)).mean()) if image.shape[axis] > 1 else 0.
                 for axis in (0, 1)]
    return np.asarray([*features, *hist, *gradients], dtype=np.float32)

def volume_features(ct, factor=1, config=None):
    config = config or FeatureConfig()
    validate_factor(factor)
    if ct.ndim != 3 or any(n < 1 for n in ct.shape) or not np.isfinite(ct).all():
        raise ValueError("Expected a finite nonempty 3D CT array.")
    coarse = coarse_ct(ct, factor)
    rows = []
    # Repeated native slices share one coarse slab; compute its features once.
    for z in range(coarse.shape[2]):
        image = np.repeat(np.repeat(coarse[:, :, z], factor, axis=0), factor, axis=1)
        image = image[:ct.shape[0], :ct.shape[1]]
        row = slice_features(image, config)
        rows.extend([row] * min(factor, ct.shape[2] - z * factor))
    return np.stack(rows)

def slice_targets(mask):
    return np.any(mask != 0, axis=(0, 1)).astype(np.uint8)

def save_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".features-", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez_compressed(stream, **values)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def case_features(root, case_id, factors, config):
    """Return validated, hash-keyed caches; changed data rebuilds only this case."""
    factors = sorted(set(factors))
    for factor in factors:
        validate_factor(factor)
    image_path, mask_path = case_paths(root, case_id)
    source = {"ct_sha256": sha256(image_path), "mask_sha256": sha256(mask_path)}
    base = {"version": FEATURE_VERSION, "case_id": case_id, "source": source,
            "config": asdict(config)}
    found, missing = {}, []
    for factor in factors:
        path = Path(root) / "features-v1" / case_id / f"scale-{factor}.npz"
        expected = {**base, "factor": factor}
        try:
            with np.load(path, allow_pickle=False) as cache:
                x, y = cache["x"], cache["y"]
                valid = (json.loads(str(cache["metadata"])) == expected
                         and x.ndim == 2 and x.shape == (len(y), config.size)
                         and y.ndim == 1 and np.isin(y, [0, 1]).all()
                         and np.isfinite(x).all())
                if not valid:
                    raise ValueError("Stale feature cache.")
                found[factor] = (x, y)
        except (OSError, ValueError, KeyError, EOFError):
            missing.append((factor, path, expected))
    if missing:
        case = load_case(root, case_id)
        y = slice_targets(case.mask)
        for factor, path, metadata in missing:
            x = volume_features(case.ct, factor, config)
            save_npz(path, x=x, y=y, metadata=json.dumps(metadata))
            found[factor] = (x, y)
    return found, source
