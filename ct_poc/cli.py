"""Separate acquisition, preparation, and reporting commands."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import csv
import json
import os
from .acquire import fetch_cases, write_json
from .data import available_cases, load_case
from .grids import grid_path, make_grid, save_grid
from .metrics import compare

def default_root():
    return Path(os.environ.get("SPLEEN_DATA_DIR", Path.cwd() / "data"))

def prepare_one(task):
    root, case_id, factors = task
    case = load_case(root, case_id)
    rows = []
    for factor in factors:
        grid = make_grid(case, factor)
        save_grid(grid_path(root, case_id, factor), grid, case)
        rows.append(compare(case, grid))
    report = {"case_id": case_id, "source": case.provenance,
              "shape": list(case.ct.shape), "spacing_mm": case.spacing.tolist(),
              "results": rows}
    write_json(Path(root) / "prepared" / case_id / "report.json", report)
    return rows

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_root())
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="Fetch selected labelled cases.")
    fetch.add_argument("--cases", nargs="+", default=["spleen_3"], help="Case IDs or all")
    prepare = commands.add_parser("prepare", help="Build grids and reports on CPU.")
    prepare.add_argument("--cases", nargs="+", default=["spleen_3"], help="Case IDs or all")
    prepare.add_argument("--factors", nargs="+", type=int, default=[1, 2, 4])
    prepare.add_argument("--workers", type=int, default=1)
    inspect = commands.add_parser("inspect", help="Read one case's geometry and labels.")
    inspect.add_argument("case_id")
    from .prediction.commands import add_commands, dispatch
    add_commands(commands)
    args = parser.parse_args(argv)
    if dispatch(args, parser):
        return
    if args.command == "fetch":
        fetch_cases(args.data_dir, args.cases)
        return
    if args.command == "inspect":
        case = load_case(args.data_dir, args.case_id)
        print(json.dumps({"case_id": case.case_id, "shape": case.ct.shape,
                          "spacing_mm": case.spacing.tolist(), "volume_ml": case.volume_ml,
                          "labels": [int(v) for v in __import__("numpy").unique(case.mask)],
                          "source": case.provenance}, indent=2))
        return
    if args.workers < 1 or any(f < 1 for f in args.factors):
        parser.error("Workers and factors must be positive.")
    allocation = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    if args.workers > allocation:
        parser.error(f"Requested {args.workers} workers but only {allocation} CPUs are allocated.")
    cases = available_cases(args.data_dir) if args.cases == ["all"] else args.cases
    if not cases:
        parser.error("No complete cases found. Run fetch first.")
    tasks = [(str(args.data_dir), case, sorted(set(args.factors))) for case in cases]
    if args.workers == 1:
        results = list(map(prepare_one, tasks))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(prepare_one, tasks))
    rows = [row for result in results for row in result]
    # This report describes this invocation; per-case reports retain earlier runs.
    report_path = args.data_dir / "prepared" / "last-run.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Prepared {len(cases)} cases; report: {report_path}", flush=True)

if __name__ == "__main__":
    main()
