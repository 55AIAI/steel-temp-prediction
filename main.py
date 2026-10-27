"""Train, evaluate and use the steel-temperature regression workflow."""
import argparse
from pathlib import Path

from steel_model import predict_file, train_model

BASE = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Prepare private raw process CSVs")
    prepare.add_argument("--raw-dir", type=Path, required=True)
    prepare.add_argument("--output", type=Path, default=BASE / "data/prepared.csv")
    train = commands.add_parser("train", help="Compare models and evaluate one chronological holdout")
    source = train.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true", help="Use included synthetic data")
    source.add_argument("--csv", type=Path, help="Prepared batch CSV matching the documented schema")
    train.add_argument("--output-dir", type=Path, default=BASE / "output")
    predict = commands.add_parser("predict", help="Predict with a trusted model artifact")
    predict.add_argument("--model", type=Path, required=True, help="Trusted joblib file created by train")
    predict.add_argument("--csv", type=Path, required=True)
    predict.add_argument("--output", type=Path, default=BASE / "output/predictions.csv")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            from prepare_data import prepare_raw
            frame, audit = prepare_raw(args.raw_dir, args.output)
            print(f"Prepared {len(frame)} batches at {args.output}")
            print("Excluded batches:", audit["dropped_batches"])
        elif args.command == "train":
            path = BASE / "demo/batches.csv" if args.demo else args.csv
            report = train_model(path, args.output_dir, synthetic=args.demo)
            print(f"Selected: {report['selected_model']}; holdout MAE: {report['holdout']['mae']:.3f} C")
            print(f"Report: {args.output_dir / 'report.md'}")
            if args.demo:
                print("Synthetic example: business threshold is not assessed.")
        else:
            result = predict_file(args.model, args.csv, args.output)
            print(f"Predicted {len(result)} batches at {args.output}")
    except (ValueError, TypeError, KeyError, OSError, ImportError) as exc:
        parser.exit(1, f"Steel workflow failed: {exc}\n")


if __name__ == "__main__":
    main()
