"""Reproducible case splits and a fixed-model resolution stress test."""
from dataclasses import asdict
from pathlib import Path
import csv
import json
import platform
import numpy as np
from ..acquire import write_json
from ..data import available_cases
from .features import FeatureConfig, case_features, validate_factor
from .model import fit_model, require_training

def split_cases(cases, seed=42):
    cases = sorted(set(cases))
    if len(cases) < 5:
        raise ValueError("Use at least five complete cases for train/validation/test. Try the demo.")
    order = np.random.default_rng(seed).permutation(cases).tolist()
    held = max(1, int(round(len(order) * 0.2)))
    return {"train": sorted(order[2 * held:]), "validation": sorted(order[held:2 * held]),
            "test": sorted(order[:held])}

def binary_metrics(y, scores, threshold=0.5):
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    if (y.ndim != 1 or scores.shape != y.shape or not len(y)
            or not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all()
            or np.any((scores < 0) | (scores > 1))):
        raise ValueError("Metrics need nonempty binary labels and finite scores in [0, 1].")
    y = y.astype(np.uint8)
    prediction = scores >= threshold
    tp = int(np.count_nonzero(prediction & (y == 1)))
    tn = int(np.count_nonzero(~prediction & (y == 0)))
    fp = int(np.count_nonzero(prediction & (y == 0)))
    fn = int(np.count_nonzero(~prediction & (y == 1)))
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {"slices": len(y), "positive_slices": int(y.sum()),
            "true_positive": tp, "true_negative": tn, "false_positive": fp, "false_negative": fn,
            "accuracy": (tp + tn) / len(y), "sensitivity": sensitivity,
            "specificity": specificity,
            "balanced_accuracy": (sensitivity + specificity) / 2
                if sensitivity is not None and specificity is not None else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "brier": float(np.mean((scores - y) ** 2))}

def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("No result rows to write.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def run_experiment(root, output, cases=None, factors=(1, 2, 4), seed=42,
                   train_factor=1, config=None):
    require_training()
    root, output = Path(root), Path(output)
    config = config or FeatureConfig()
    cases = available_cases(root) if cases is None else list(cases)
    if len(cases) != len(set(cases)):
        raise ValueError("Case IDs must be unique.")
    missing = set(cases) - set(available_cases(root))
    if missing:
        raise ValueError("Missing complete cases: " + ", ".join(sorted(missing)))
    split = split_cases(cases, seed)
    factors = sorted(set([*factors, train_factor]))
    for factor in factors:
        validate_factor(factor)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Run folder is not empty. Choose a new --output to preserve past results.")
    output.mkdir(parents=True, exist_ok=True)
    features, sources = {}, {}
    for case_id in sorted(cases):
        print(f"Features: {case_id}", flush=True)
        features[case_id], sources[case_id] = case_features(root, case_id, factors, config)
    x = np.concatenate([features[c][train_factor][0] for c in split["train"]])
    y = np.concatenate([features[c][train_factor][1] for c in split["train"]])
    model = fit_model(x, y, config, seed)
    model.metadata.update({"train_factor": train_factor, "training_cases": split["train"],
                           "training_sources": {c: sources[c] for c in split["train"]}})
    model.save(output / "model.npz")
    majority = int(np.mean(y) >= 0.5)
    predictions, per_case, summaries = [], [], []
    for subset, members in split.items():
        for factor in factors:
            labels, scores, case_metrics = [], [], []
            for case_id in members:
                x_case, y_case = features[case_id][factor]
                score = model.score(x_case)
                measured = binary_metrics(y_case, score)
                per_case.append({"split": subset, "factor": factor, "case_id": case_id, **measured})
                case_metrics.append(measured)
                labels.extend(y_case.tolist())
                scores.extend(score.tolist())
                for index, (target, value) in enumerate(zip(y_case, score)):
                    predictions.append({"split": subset, "case_id": case_id,
                        "slice_index": index, "factor": factor, "target": int(target),
                        "score": float(value), "prediction": int(value >= 0.5)})
            measured = binary_metrics(labels, scores)
            valid = [r["balanced_accuracy"] for r in case_metrics if r["balanced_accuracy"] is not None]
            summaries.append({"split": subset, "factor": factor, "model": "logistic",
                **measured, "macro_case_balanced_accuracy": float(np.mean(valid)) if valid else None,
                "cases": len(members)})
            summaries.append({"split": subset, "factor": factor, "model": "majority",
                **binary_metrics(labels, np.full(len(labels), majority)),
                "macro_case_balanced_accuracy": None, "cases": len(members)})
    manifest = {"schema": 1, "task": "axial-slice spleen presence",
        "synthetic": (root / "synthetic.json").is_file(), "seed": seed, "splits": split,
        "sources": sources, "feature_config": asdict(config), "model": model.metadata,
        "factors": factors, "train_factor": train_factor, "threshold": 0.5,
        "target": "Any native reference-mask voxel in this axial slice is spleen.",
        "resolution": "3D block mean CT; mapped back to native slice indices; native targets unchanged.",
        "protocol": "Train once; fixed model and threshold across scales. No test-set selection.",
        "score_note": "Class-balanced logistic score; not calibrated clinical confidence.",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "sklearn": model.metadata["sklearn_version"]}}
    write_json(output / "manifest.json", manifest)
    write_csv(output / "predictions.csv", predictions)
    write_csv(output / "per-case.csv", per_case)
    write_csv(output / "metrics.csv", summaries)
    write_json(output / "metrics.json", summaries)
    test = [r for r in summaries if r["split"] == "test" and r["model"] == "logistic"]
    lines = ["# Slice-presence baseline", "",
             "**SYNTHETIC DEMO: not research evidence.**" if manifest["synthetic"] else
             "Exploratory held-out case evaluation; not clinical validation.", "",
             f"Cases: {len(cases)}. Train / validation / test: " +
             " / ".join(str(len(split[k])) for k in ("train", "validation", "test")) + ".",
             f"Training scale: {train_factor}. Fixed threshold: 0.5.", "",
             "| Scale | Sensitivity | Specificity | Balanced accuracy |",
             "| --- | --- | --- | --- |"]
    for row in test:
        values = ["undefined" if row[key] is None else f"{row[key]:.3f}"
                  for key in ("sensitivity", "specificity", "balanced_accuracy")]
        lines.append("| " + " | ".join([str(row["factor"]), *values]) + " |")
    lines.extend(["", "See metrics.csv for the train-only majority comparator and all splits.",
                  "Scores are not calibrated confidence. Slice counts are correlated within cases.",
                  "Case IDs define the split; unique patient identity is not independently verified.",
                  "Coarse slabs mix neighboring slices. Targets remain defined on native slices.",
                  "Do not tune this model against test results. Routing is not implemented.", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved baseline and report: {output}", flush=True)
    return manifest, summaries
