"""Interactive viewer; each browser session owns one active case."""
from pathlib import Path
import csv
import io
import json
import os
import numpy as np
import streamlit as st
from .cli import default_root
from .data import available_cases, case_paths, load_case
from .grids import grid_path, load_grid, make_grid
from .metrics import compare
from .views import PLANES, figure, png_bytes

def csv_bytes(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()

def main():
    st.set_page_config(page_title="Spleen · Resolution Lab", page_icon="◈", layout="wide")
    st.caption("CT  /  DATA EXPLORATION")
    st.title("Spleen · Resolution Lab")
    st.write("See what changes when a CT scan uses fewer voxels.")
    root = default_root()
    if (root / "synthetic.json").is_file():
        st.warning("Synthetic demo: these are made-up shapes, not human CT scans.")
    cases = available_cases(root)
    if not cases:
        st.info("Fetch a labelled scan to begin. The first sample is about 11 MB.")
        st.code("python -m ct_poc.cli fetch --cases spleen_3\npython -m ct_poc.cli prepare --cases spleen_3")
        st.stop()
    with st.sidebar:
        st.header("Explore")
        selected = st.selectbox("Case", cases, index=cases.index("spleen_3") if "spleen_3" in cases else 0)
        plane = st.selectbox("Plane", list(PLANES))
        factor = st.select_slider("Grid scale", options=[1, 2, 4], value=2,
                                  format_func=lambda f: "Original" if f == 1 else f"{f}× coarser")
        overlay = st.checkbox("Show spleen labels", value=True)
        opacity = st.slider("Label opacity", 0.0, 1.0, 0.45, 0.05)
        fractions = st.checkbox("Show cell fractions", value=False)
        center = st.slider("CT window center", -1000, 1000, 50, 10)
        width = st.slider("CT window width", 50, 3000, 400, 50)
    paths = case_paths(root, selected)
    key = (str(root.resolve()), selected, *(p.stat().st_mtime_ns for p in paths))
    try:
        if st.session_state.get("case_key") != key:
            # Drop references before loading the next full volume.
            st.session_state.pop("case", None)
            st.session_state["grids"] = {}
            st.session_state["case"] = load_case(root, selected)
            st.session_state["case_key"] = key
        case = st.session_state["case"]
        levels = st.session_state["grids"]
        if factor not in levels:
            prepared = grid_path(root, selected, factor)
            if prepared.is_file():
                levels[factor] = load_grid(prepared, case, factor)
            else:
                with st.spinner("Building this grid once…"):
                    levels[factor] = make_grid(case, factor)
        grid = levels[factor]
    except (OSError, ValueError, KeyError) as error:
        st.error(str(error))
        st.stop()
    axis = PLANES[plane]
    counts = np.count_nonzero(case.mask, axis=tuple(i for i in range(3) if i != axis))
    initial = int(np.argmax(counts)) if counts.any() else case.ct.shape[axis] // 2
    if case.ct.shape[axis] == 1:
        index = 0
        st.caption("Single slice")
    else:
        index = st.slider("Slice", 0, case.ct.shape[axis] - 1, initial, key=f"slice_{selected}_{plane}")
    measurements = compare(case, grid)
    a, b, c, d = st.columns(4)
    a.metric("Reference spleen", f"{measurements['reference_ml']:.2f} mL")
    change = measurements["volume_change_percent"]
    b.metric("Coarse hard volume", f"{measurements['hard_ml']:.2f} mL",
             f"{change:+.2f}%" if change is not None else None, delta_color="off")
    c.metric("Mask overlap · Dice", f"{measurements['dice']:.4f}")
    d.metric("Grid arrays", f"{measurements['grid_array_mib']:.2f} MiB")
    fig = figure(case, grid, plane, index, (center - width / 2, center + width / 2),
                 overlay, opacity, fractions)
    rendered = png_bytes(fig)
    st.image(rendered, width="stretch")
    st.caption("All panels share physical coordinates. Red = lost spleen; blue = gained spleen.")
    if fractions:
        st.caption("Fractions show how much of each cell was labelled spleen. They are not confidence scores.")
    details, export = st.columns([2, 1])
    with details:
        st.subheader("What the grid preserves")
        st.write(f"Fractional volume: **{measurements['fractional_ml']:.2f} mL**. "
                 "Hard labels use a fixed 50% threshold.")
        native_spacing = " × ".join(f"{s:.3f}" for s in case.spacing)
        coarse_spacing = " × ".join(f"{s * factor:.3f}" for s in case.spacing)
        st.caption(f"Original cells: {native_spacing} mm · Full coarse cells: {coarse_spacing} mm")
        st.caption("Edge cells may be smaller. Background includes other organs and tissue.")
    with export:
        st.subheader("Keep this comparison")
        stem = f"{selected}_{plane.lower()}_{index}_scale-{factor}"
        st.download_button("Download figure", rendered, stem + ".png", "image/png")
        row = {**measurements, "plane": plane, "slice_index": index,
               "position_mm": float(case.affine[axis, 3] + index * case.spacing[axis]),
               "window_center": center, "window_width": width,
               "overlay": overlay, "opacity": opacity, "fractions": fractions,
               **{k: case.provenance[k] for k in ("ct_sha256", "mask_sha256")}}
        st.download_button("Download measurements", csv_bytes([row]), stem + ".csv", "text/csv")
    with st.expander("Across prepared cases"):
        rows = []
        for path in sorted((root / "prepared").glob("spleen_*/report.json")):
            try:
                rows.extend(json.loads(path.read_text())["results"])
            except (OSError, ValueError, KeyError):
                st.warning(f"Could not read the report for {path.parent.name}.")
        if rows:
            import pandas as pd
            frame = pd.DataFrame(rows)
            st.dataframe(frame[["case_id", "factor", "hard_ml", "volume_change_percent", "dice"]],
                         hide_index=True, width="stretch")
            chart = frame.pivot(index="case_id", columns="factor", values="dice")
            st.bar_chart(chart)
            st.download_button("Download cohort measurements", csv_bytes(rows), "cohort.csv", "text/csv")
        else:
            st.write("Run preparation to compare cases here.")
    with st.expander("Slice prediction baseline"):
        from .prediction.ui import results_panel
        results_panel()
    with st.expander("About this experiment"):
        st.write("These are supplied reference labels, not model predictions. "
                 "This tool measures representation changes and is for research exploration.")
        st.markdown("[Medical Segmentation Decathlon](https://medicaldecathlon.com/) · "
                    "Spleen dataset · CC-BY-SA 4.0")
        st.json({"case": selected, "shape": list(case.ct.shape),
                 "spacing_mm": case.spacing.tolist(), "orientation": "RAS",
                 "source_hashes": {k: case.provenance[k] for k in ("ct_sha256", "mask_sha256")}})
