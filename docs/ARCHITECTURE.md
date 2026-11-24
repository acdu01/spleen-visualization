# Architecture

The CLI prepares data independently from the viewer.
The viewer keeps one active case per session.

```text
Archive -> acquire -> data -> grids -> metrics -> views -> app
```

| Module | Role |
| --- | --- |
| ct_poc.acquire | Fetch scan/mask pairs and record hashes |
| ct_poc.data | Validate shape, spacing, labels, and orientation |
| ct_poc.grids | Average CT blocks and keep spleen fractions |
| ct_poc.metrics | Compare coarse and reference labels |
| ct_poc.views | Render slices with physical spacing |
| ct_poc.app | Own controls, session state, and exports |
| ct_poc.cli | Expose fetch, prepare, and inspect |

Changing slices reuses the loaded case and grids.
Missing levels are built once per session.
Switching cases drops the previous case from session state.

## Numerical rules

- NIfTI image and mask shapes and affines must match. Units must be mm.
- Arrays are placed in RAS order; tilted/sheared grids are rejected.
- CT intensities remain float32. Labels are uint8, background=0 and spleen=1.
- Block CT values are means; spleen fractions are labelled voxel counts / valid counts.
- Hard labels use fraction >= 0.5. Fractions describe geometry, not confidence.
- Partial edge blocks retain their actual size and volume. No padded zeros.
- Fractional volume preserves reference volume within floating-point tolerance.
- Dice compares coarse hard labels expanded over their native cells to the reference mask.
- Volume change is undefined for an empty reference; Dice of two empty masks is 1.
- Brightness changes only display; calculations always use original values.
- Grid memory includes means, fractions, hard labels, counts, and edge coordinates.
- Prepared grids include source hashes and algorithm version; stale grids are rejected.


## Reference example

spleen_3 has shape 512×512×40.
Its spacing is about 0.738×0.738×5 mm.
Its supplied label volume is about 158.14 mL.
A 4× cell spans 20 mm along the slice axis.

Fractional volume preserves the supplied label volume.
Hard volume can change as the boundary becomes blocky.
Dice measures this representation change, not model accuracy.
