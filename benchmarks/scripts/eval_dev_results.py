import argparse

from benchmark import Benchmark


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str, required=True,
        help="The directory where the generation results are saved."
    )
    parser.add_argument(
        "--data_dir",
        type=str, 
        default=None
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing metrics and recompute from scratch"
    )
    parser.add_argument(
        "--task_types",
        type=str,
        nargs='+',
        default=None,
        help="Task name list (e.g., item_understand rec_reason). If not specified, all tasks will be evaluated."
    )
    return parser.parse_args()


def main():
    args = get_args()
    eval_results_path = f"{args.output_dir}/eval_results.json"
    Benchmark.evaluate_dev(
        generation_results_dir=args.output_dir,
        output_path=eval_results_path,
        data_dir=args.data_dir,
        overwrite=args.overwrite,
        task_types=args.task_types
    )


if __name__ == "__main__":
    main()
