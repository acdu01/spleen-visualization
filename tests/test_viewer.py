from pathlib import Path
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest
from ct_poc.grids import make_grid
from ct_poc.views import figure, png_bytes, PLANES

@pytest.mark.parametrize("plane", list(PLANES))
def test_render_planes_with_partial_cells(case, plane):
    grid = make_grid(case, 4)
    fig = figure(case, grid, plane, index=case.ct.shape[PLANES[plane]] - 1)
    assert png_bytes(fig).startswith(b"\x89PNG\r\n\x1a\n")
    for ax in fig.axes:
        assert ax.get_aspect() == 1.0

def test_viewer_controls_reuse_loaded_objects(data_root, monkeypatch):
    monkeypatch.setenv("SPLEEN_DATA_DIR", str(data_root))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30)
    app.run()
    assert not app.exception
    original = app.session_state["case"]
    grid = app.session_state["grids"][2]
    slider = next(s for s in app.slider if s.label == "Slice")
    slider.set_value(0).run()
    assert not app.exception
    assert app.session_state["case"] is original
    assert app.session_state["grids"][2] is grid
    app.selectbox[1].select("Sagittal").run()
    app.select_slider[0].set_value(4).run()
    app.checkbox[1].check().run()
    assert not app.exception
    assert app.session_state["grids"][4].factor == 4
    assert len(app.get("download_button")) == 2

def test_missing_data_has_clear_state(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLEEN_DATA_DIR", str(tmp_path))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    app.run()
    assert not app.exception
    assert "Fetch a labelled scan" in app.info[0].value
