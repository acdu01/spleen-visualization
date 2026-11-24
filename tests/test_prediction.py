"""Scientific and portability checks for the CPU baseline."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import nibabel as nib
import pytest
from ct_poc.demo import create_demo
from ct_poc.data import load_case, load_ct, case_paths
from ct_poc.prediction.features import FeatureConfig, volume_features, slice_targets, case_features
from ct_poc.prediction.model import fit_model, load_model
from ct_poc.prediction.experiment import split_cases, binary_metrics, run_experiment

def test_case_splits_are_disjoint_and_order_independent():
    cases = [f"spleen_{i}" for i in range(41)]
    a = split_cases(cases, 12)
    assert a == split_cases(list(reversed(cases)), 12)
    assert {key: len(value) for key, value in a.items()} == {"train": 25, "validation": 8, "test": 8}
    assert len({c for values in a.values() for c in values}) == len(cases)
    assert set(a["train"]).isdisjoint(a["validation"])
    assert set(a["train"]).isdisjoint(a["test"])
    with pytest.raises(ValueError, match="five"):
        split_cases(cases[:4])

def test_inputs_do_not_depend_on_masks(case):
    other = replace(case, mask=np.zeros_like(case.mask))
    config = FeatureConfig(tiles=2)
    assert np.array_equal(volume_features(case.ct, 4, config), volume_features(other.ct, 4, config))
    assert not np.array_equal(slice_targets(case.mask), slice_targets(other.mask))

def test_coarse_slabs_keep_native_targets_and_partial_edges():
    ct = np.broadcast_to(np.arange(5, dtype=np.float32), (6, 6, 5)).copy()
    config = FeatureConfig(tiles=2)
    x = volume_features(ct, 2, config)
    assert x.shape == (5, config.size)
    np.testing.assert_array_equal(x[0], x[1])
    np.testing.assert_array_equal(x[2], x[3])
    np.testing.assert_allclose(x[4], volume_features(ct[:, :, 4:], 1, config)[0])
    mask = np.zeros_like(ct)
    mask[0, 0, 1] = 1
    assert slice_targets(mask).tolist() == [0, 1, 0, 0, 0]

def test_known_confusion_metrics_and_absent_classes():
    metrics = binary_metrics([1, 1, 0, 0], [.9, .2, .8, .1])
    assert [metrics[k] for k in ["true_positive", "true_negative", "false_positive", "false_negative"]] == [1] * 4
    assert metrics["balanced_accuracy"] == .5
    empty_class = binary_metrics([0, 0], [.1, .9])
    assert empty_class["sensitivity"] is None
    assert empty_class["balanced_accuracy"] is None

def test_saved_model_roundtrip_and_training_only_scaler(tmp_path):
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(3)
    config = FeatureConfig(tiles=2)
    x = rng.normal(size=(20, config.size))
    y = np.array([0, 1] * 10)
    x[:, 0] += y * 2
    model = fit_model(x, y, config)
    np.testing.assert_allclose(model.mean, x.mean(axis=0))
    original_mean = model.mean.copy()
    model.score(x + 100)
    np.testing.assert_array_equal(model.mean, original_mean)
    path = tmp_path / "model.npz"
    model.save(path)
    np.testing.assert_allclose(load_model(path).score(x), model.score(x))
    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[key].dtype != object for key in archive.files)

def test_features_cache_invalidates_labels_without_changing_inputs(tmp_path):
    create_demo(tmp_path, cases=5)
    config = FeatureConfig(tiles=2)
    a, source_a = case_features(tmp_path, "spleen_1000", [1], config)
    _, label_path = case_paths(tmp_path, "spleen_1000")
    label = nib.load(label_path)
    changed = nib.Nifti1Image(np.zeros(label.shape, dtype=np.uint8), label.affine, label.header)
    nib.save(changed, label_path)
    b, source_b = case_features(tmp_path, "spleen_1000", [1], config)
    assert source_a["mask_sha256"] != source_b["mask_sha256"]
    np.testing.assert_array_equal(a[1][0], b[1][0])
    assert a[1][1].any() and not b[1][1].any()

def test_ct_only_loader_needs_no_mask(tmp_path):
    create_demo(tmp_path, cases=5)
    image, label = case_paths(tmp_path, "spleen_1000")
    expected = load_case(tmp_path, "spleen_1000").ct
    label.unlink()
    np.testing.assert_array_equal(load_ct(image).ct, expected)

def test_synthetic_run_artifacts_and_no_test_label_leakage(tmp_path):
    pytest.importorskip("sklearn")
    root, output = tmp_path / "data", tmp_path / "run"
    create_demo(root, cases=5)
    config = FeatureConfig(tiles=2)
    manifest, metrics = run_experiment(root, output, config=config)
    assert manifest["synthetic"]
    assert set(manifest["splits"]) == {"train", "validation", "test"}
    assert {r["factor"] for r in metrics} == {1, 2, 4}
    assert {r["model"] for r in metrics} == {"logistic", "majority"}
    assert (output / "report.md").read_text().startswith("# Slice-presence baseline")
    original = load_model(output / "model.npz")
    test_case = manifest["splits"]["test"][0]
    _, path = case_paths(root, test_case)
    image = nib.load(path)
    nib.save(nib.Nifti1Image(np.zeros(image.shape, dtype=np.uint8), image.affine, image.header), path)
    changed = tmp_path / "changed"
    run_experiment(root, changed, config=config)
    after = load_model(changed / "model.npz")
    np.testing.assert_array_equal(original.coefficient, after.coefficient)
    np.testing.assert_array_equal(original.mean, after.mean)
    with pytest.raises(ValueError, match="not empty"):
        run_experiment(root, output, config=config)

def test_demo_cannot_mix_with_existing_data(tmp_path):
    (tmp_path / "existing.txt").write_text("keep")
    with pytest.raises(ValueError, match="empty"):
        create_demo(tmp_path)

def test_cli_reports_missing_cases_and_supports_wheel_paths(tmp_path, monkeypatch, capsys):
    from ct_poc.cli import main, default_root
    monkeypatch.delenv("SPLEEN_DATA_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    assert default_root() == tmp_path / "data"
    with pytest.raises(SystemExit) as error:
        main(["--data-dir", str(tmp_path), "train", "--output", str(tmp_path / "run")])
    assert error.value.code == 2
    assert "five" in capsys.readouterr().err

def test_results_panel_reads_saved_run(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    from streamlit.testing.v1 import AppTest
    root, output = tmp_path / "data", tmp_path / "outputs" / "demo"
    create_demo(root, cases=5)
    run_experiment(root, output, config=FeatureConfig(tiles=2))
    monkeypatch.setenv("SPLEEN_DATA_DIR", str(root))
    monkeypatch.setenv("SPLEEN_RESULTS_DIR", str(output))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert any("Synthetic demo" in warning.value for warning in app.warning)
    assert any(box.label == "Prediction case" for box in app.selectbox)
    next(box for box in app.selectbox if box.label == "Prediction scale").select(4).run()
    assert not app.exception

@pytest.mark.parametrize("labels", [[0.7, 1], [256, 0], [-1, 1]])
def test_metrics_reject_nonbinary_labels_before_casting(labels):
    with pytest.raises(ValueError, match="binary"):
        binary_metrics(labels, [0.2, 0.8])

def test_missing_ml_dependency_fails_before_data_or_output_work(tmp_path, monkeypatch):
    import ct_poc.prediction.experiment as experiment
    def unavailable():
        raise ValueError("Training needs the ML extra")
    monkeypatch.setattr(experiment, "require_training", unavailable)
    with pytest.raises(ValueError, match="ML extra"):
        experiment.run_experiment(tmp_path / "missing", tmp_path / "output")
    assert not (tmp_path / "output").exists()
