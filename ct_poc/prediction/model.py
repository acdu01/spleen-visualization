"""A logistic baseline saved as plain arrays, without executable pickle files."""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import warnings
import numpy as np
from .features import FeatureConfig, FEATURE_VERSION, save_npz

MODEL_VERSION = 1

@dataclass(frozen=True)
class SliceModel:
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float
    metadata: dict

    def score(self, x):
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError("Prediction features have the wrong shape or non-finite values.")
        logits = np.clip(((x - self.mean) / self.scale) @ self.coefficient + self.intercept, -50, 50)
        return 1.0 / (1.0 + np.exp(-logits))

    def save(self, path):
        save_npz(path, mean=self.mean, scale=self.scale, coefficient=self.coefficient,
                 intercept=np.asarray(self.intercept), metadata=json.dumps(self.metadata))

def load_model(path):
    try:
        with np.load(Path(path), allow_pickle=False) as values:
            metadata = json.loads(str(values["metadata"]))
            mean, scale, coefficient = [values[key].copy() for key in ("mean", "scale", "coefficient")]
            intercept = float(values["intercept"])
        config = FeatureConfig(**metadata["feature_config"])
        if (metadata.get("model_version") != MODEL_VERSION
                or metadata.get("feature_version") != FEATURE_VERSION):
            raise ValueError("Unsupported model version.")
        if any(a.shape != (config.size,) or not np.isfinite(a).all()
               for a in (mean, scale, coefficient)) or np.any(scale <= 0) or not np.isfinite(intercept):
            raise ValueError("Invalid model arrays.")
        return SliceModel(mean, scale, coefficient, intercept, metadata)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f"Cannot load baseline model: {error}") from error

def require_training():
    try:
        import sklearn
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from threadpoolctl import threadpool_limits
    except ImportError as error:
        raise ValueError('Training needs the ML extra: python -m pip install -e ".[ml]"') from error
    return sklearn, ConvergenceWarning, LogisticRegression, StandardScaler, threadpool_limits

def fit_model(x, y, config, seed=42):
    sklearn, ConvergenceWarning, LogisticRegression, StandardScaler, threadpool_limits = require_training()
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Training cases must contain both spleen and non-spleen slices.")
    with threadpool_limits(limits=1):
        scaler = StandardScaler().fit(x)
        classifier = LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs",
                                        max_iter=2000, random_state=seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(scaler.transform(x), y)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            raise ValueError("Baseline did not converge; no model was accepted.")
    metadata = {"model_version": MODEL_VERSION, "feature_version": FEATURE_VERSION,
                "feature_config": asdict(config), "seed": seed, "threshold": 0.5,
                "algorithm": "StandardScaler + balanced LogisticRegression",
                "C": 1.0, "max_iter": 2000, "iterations": int(classifier.n_iter_[0]),
                "numpy_version": np.__version__, "sklearn_version": sklearn.__version__}
    return SliceModel(scaler.mean_, scaler.scale_, classifier.coef_[0],
                      float(classifier.intercept_[0]), metadata)
