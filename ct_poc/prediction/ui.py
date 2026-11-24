"""Read saved experiment results without training or requiring scikit-learn."""
from pathlib import Path
import json
import os
import pandas as pd
import streamlit as st

def results_panel():
    root = Path(os.environ.get("SPLEEN_RESULTS_DIR", Path.cwd() / "outputs"))
    st.subheader("Slice prediction experiments")
    st.caption("Separate from the supplied spleen overlays above. Training runs in the CLI.")
    candidates = [root / "manifest.json"] if (root / "manifest.json").is_file() else sorted(root.glob("*/manifest.json"))
    if not candidates:
        st.info("No saved experiments yet. Try the synthetic demo in README.md.")
        return
    chosen = st.selectbox("Experiment", candidates, format_func=lambda path: path.parent.name)
    try:
        manifest = json.loads(chosen.read_text())
        metrics = pd.read_csv(chosen.parent / "metrics.csv")
        predictions = pd.read_csv(chosen.parent / "predictions.csv")
        if manifest.get("schema") != 1:
            raise ValueError("Unsupported results version.")
        if manifest.get("synthetic"):
            st.warning("Synthetic demo: made-up shapes, not evidence about real CT scans.")
        st.write("Does each axial slice contain any labelled spleen?")
        st.caption("Cases are split before fitting. Scores are not calibrated confidence.")
        held = metrics[metrics["split"] == "test"]
        st.dataframe(held[["factor", "model", "sensitivity", "specificity", "balanced_accuracy"]],
                     hide_index=True, width="stretch")
        st.caption("Sensitivity: share of spleen slices found. Specificity: share of non-spleen slices rejected.")
        case_id = st.selectbox("Prediction case", sorted(predictions["case_id"].unique()))
        factor = st.selectbox("Prediction scale", manifest["factors"])
        rows = predictions[(predictions["case_id"] == case_id) & (predictions["factor"] == factor)].sort_values("slice_index")
        st.caption(f"Case split: {rows['split'].iloc[0]}. Target 1 means supplied spleen label present.")
        chart = rows.set_index("slice_index")[["target", "score"]].copy()
        chart["threshold"] = manifest["threshold"]
        st.line_chart(chart)
        st.download_button("Download experiment metrics", (chosen.parent / "metrics.csv").read_bytes(),
                           file_name="slice-metrics.csv", mime="text/csv")
        with st.expander("Experiment settings"):
            st.json(manifest)
    except (OSError, ValueError, KeyError, IndexError) as error:
        st.error(f"Cannot read this experiment: {error}")
