"""Portable commands; training is never started by the viewer."""
from pathlib import Path
import numpy as np
from .features import FeatureConfig, volume_features
from .model import load_model

def add_commands(commands):
    demo = commands.add_parser("demo", help="Create small synthetic cases without downloading data.")
    demo.add_argument("--cases", type=int, default=12)
    demo.add_argument("--seed", type=int, default=42)
    train = commands.add_parser("train", help="Train a CPU slice baseline and evaluate fixed case splits.")
    train.add_argument("--output", type=Path, required=True, help="New empty results folder.")
    train.add_argument("--cases", nargs="+", default=["all"])
    train.add_argument("--factors", nargs="+", type=int, default=[1, 2, 4])
    train.add_argument("--train-factor", type=int, default=1)
    train.add_argument("--seed", type=int, default=42)
    predict = commands.add_parser("predict", help="Score an unlabelled CT volume using a saved model.")
    predict.add_argument("--image", type=Path, required=True)
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True, help="New CSV file.")
    predict.add_argument("--factor", type=int, default=1)
    view = commands.add_parser("view", help="Start the local viewer; also works after wheel installation.")
    view.add_argument("--results-dir", type=Path, default=None)
    view.add_argument("--port", type=int, default=8501)

def dispatch(args, parser):
    if args.command not in {"demo", "train", "predict", "view"}:
        return False
    try:
        if args.command == "demo":
            from ..demo import create_demo
            cases = create_demo(args.data_dir, args.cases, args.seed)
            print(f"Created {len(cases)} SYNTHETIC cases in {args.data_dir}. Not research data.")
        elif args.command == "train":
            from .experiment import run_experiment
            run_experiment(args.data_dir, args.output,
                None if args.cases == ["all"] else args.cases,
                args.factors, args.seed, args.train_factor)
        elif args.command == "predict":
            from ..data import load_ct, sha256
            from .experiment import write_csv
            if args.output.exists():
                raise ValueError("Prediction output exists. Choose a new --output.")
            model = load_model(args.model)
            case = load_ct(args.image)
            config = FeatureConfig(**model.metadata["feature_config"])
            scores = model.score(volume_features(case.ct, args.factor, config))
            image_hash, model_hash = sha256(args.image), sha256(args.model)
            rows = [{"ct_sha256": image_hash, "model_sha256": model_hash, "slice_index": i, "position_mm": float(case.affine[2, 3] + i * case.spacing[2]),
                     "factor": args.factor, "score": float(p), "prediction": int(p >= 0.5)}
                    for i, p in enumerate(scores)]
            write_csv(args.output, rows)
            print(f"Scored {len(rows)} slices; saved {args.output}")
        else:
            import os
            import sys
            from streamlit.web import cli
            if not 1 <= args.port <= 65535:
                raise ValueError("Port must be 1..65535.")
            os.environ["SPLEEN_DATA_DIR"] = str(args.data_dir.resolve())
            if args.results_dir is not None:
                os.environ["SPLEEN_RESULTS_DIR"] = str(args.results_dir.resolve())
            sys.argv = ["streamlit", "run", str(Path(__file__).parents[1] / "viewer.py"),
                        "--server.address", "127.0.0.1", "--server.port", str(args.port),
                        "--server.headless", "true", "--browser.gatherUsageStats", "false"]
            cli.main()
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return True
