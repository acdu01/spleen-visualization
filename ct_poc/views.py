"""Physical-space slice rendering with no web dependencies."""
import io
import numpy as np
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

PLANES = {"Axial": 2, "Coronal": 1, "Sagittal": 0}
AXIS_LABELS = ("Left → Right (mm)", "Posterior → Anterior (mm)", "Inferior → Superior (mm)")

def plane_slice(array, axis, index):
    return np.take(array, index, axis=axis).T

def figure(case, grid, plane="Axial", index=0, window=(-150, 250),
           overlay=True, opacity=0.4, fractions=False):
    axis = PLANES[plane]
    if not 0 <= index < case.ct.shape[axis]:
        raise ValueError("Slice index out of range.")
    if window[1] <= window[0]:
        raise ValueError("Window upper bound must exceed the lower bound.")
    spatial = [i for i in range(3) if i != axis]
    native_edges = tuple((np.arange(n + 1) - 0.5) * s + o
                        for n, s, o in zip(case.ct.shape, case.spacing, case.affine[:3, 3]))
    location = case.affine[axis, 3] + index * case.spacing[axis]
    coarse_index = int(np.clip(np.searchsorted(grid.edges[axis], location, side="right") - 1,
                               0, grid.ct.shape[axis] - 1))
    native_ct = plane_slice(case.ct, axis, index)
    native_mask = plane_slice(case.mask, axis, index)
    coarse_ct = plane_slice(grid.ct, axis, coarse_index)
    coarse_mask = plane_slice(grid.fraction if fractions else grid.hard_mask, axis, coarse_index)
    coarse_hard = plane_slice(grid.hard_mask, axis, coarse_index)
    first, second = spatial
    expanded = np.repeat(np.repeat(coarse_hard, grid.factor, axis=0), grid.factor, axis=1)
    expanded = expanded[:native_mask.shape[0], :native_mask.shape[1]]
    difference = np.zeros_like(native_mask, dtype=np.uint8)
    difference[(native_mask == 1) & (expanded == 0)] = 1
    difference[(native_mask == 0) & (expanded == 1)] = 2
    fig = Figure(figsize=(12, 4.4), layout="constrained", facecolor="#0b1220")
    axes = fig.subplots(1, 3, sharex=True, sharey=True)
    for ax in axes:
        ax.set_facecolor("#0b1220")
        ax.tick_params(colors="#b6c4d6", labelsize=8)
        ax.set_aspect("equal")
        ax.set_xlabel(AXIS_LABELS[first], color="#b6c4d6", fontsize=9)
        for spine in ax.spines.values():
            spine.set_color("#334155")
    axes[0].set_ylabel(AXIS_LABELS[second], color="#b6c4d6", fontsize=9)
    for ax, ct, mask, edges in zip(axes[:2], (native_ct, coarse_ct), (native_mask, coarse_mask),
                                  (native_edges, grid.edges)):
        ax.pcolormesh(edges[first], edges[second], ct, cmap="gray",
                      vmin=window[0], vmax=window[1], shading="flat", rasterized=True)
        if overlay:
            ax.pcolormesh(edges[first], edges[second], np.ma.masked_where(mask <= 0, mask),
                          cmap="Reds", vmin=0, vmax=1, alpha=opacity,
                          shading="flat", rasterized=True)
    axes[2].pcolormesh(native_edges[first], native_edges[second], native_ct,
                      cmap="gray", vmin=window[0], vmax=window[1], shading="flat", rasterized=True)
    axes[2].pcolormesh(native_edges[first], native_edges[second],
                      np.ma.masked_where(difference == 0, difference),
                      cmap=ListedColormap(["#fb7185", "#38bdf8"]), vmin=0.5, vmax=2.5,
                      alpha=0.85, shading="flat", rasterized=True)
    axes[2].legend(handles=[Patch(color="#fb7185", label="Lost"), Patch(color="#38bdf8", label="Gained")],
                   loc="lower left", fontsize=8, facecolor="#0b1220", labelcolor="white")
    names = ["Original", f"{grid.factor}× grid" + (" · fractions" if fractions else ""), "Hard-label difference"]
    for ax, title in zip(axes, names):
        ax.set_title(title, color="#f1f5f9", fontsize=12)
    fig.suptitle(f"{case.case_id} · {plane.lower()} · slice {index} · {location:.2f} mm",
                 color="#f1f5f9", fontsize=12)
    return fig

def png_bytes(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140, facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.15)
    return buffer.getvalue()
