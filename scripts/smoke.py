"""Exercise the real sample and the Streamlit app after preparation."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import requests
import numpy as np
from streamlit.testing.v1 import AppTest
from ct_poc.cli import default_root
from ct_poc.data import load_case, available_cases
from ct_poc.grids import load_grid, grid_path
from ct_poc.metrics import compare
from ct_poc.views import figure, png_bytes

project = Path(__file__).resolve().parents[1]
root = default_root()
out = project / "outputs"
out.mkdir(exist_ok=True)
case = load_case(root, "spleen_3")
assert abs(case.volume_ml - 158.137924748657) < 1e-5
grid = load_grid(grid_path(root, case.case_id, 4), case, 4)
row = compare(case, grid)
assert abs(row["fractional_ml"] - case.volume_ml) < 1e-4
fig = figure(case, grid, "Axial", 26)
(out / "spleen_3_axial_scale-4.png").write_bytes(png_bytes(fig))
app = AppTest.from_file(str(project / "app.py"), default_timeout=90).run()
assert not app.exception, app.exception
app.select_slider[0].set_value(4).run()
assert not app.exception, app.exception
for plane in ["Coronal", "Sagittal", "Axial"]:
    app.selectbox[1].select(plane).run()
    assert not app.exception, app.exception
app.selectbox[0].select("spleen_19").run()
assert not app.exception, app.exception
assert app.session_state["case"].case_id == "spleen_19"
summary = {"downloaded_cases": len(available_cases(root)), "sample": row,
           "ui": "case switch, all planes, scale switch, exports: passed"}
# Check the actual web server on this allocated node, then stop it.
with (out / "server-smoke.log").open("w") as log:
    process = subprocess.Popen([sys.executable, "-m", "streamlit",
        "run", str(project / "app.py"), "--server.address", "127.0.0.1",
        "--server.port", "18501", "--server.headless", "true"], stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(30):
            if process.poll() is not None:
                raise RuntimeError("Viewer server exited; see outputs/server-smoke.log")
            try:
                response = requests.get("http://127.0.0.1:18501/_stcore/health", timeout=2)
                if response.ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        else:
            raise RuntimeError("Viewer server did not become healthy.")
        page = requests.get("http://127.0.0.1:18501", timeout=5)
        page.raise_for_status()
        assert "<html" in page.text.lower()
        summary["http"] = {"health": response.status_code, "page": page.status_code}
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
(out / "smoke.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
