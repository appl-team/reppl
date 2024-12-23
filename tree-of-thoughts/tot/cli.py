"""Command line interface for Tree of Thoughts solver."""

from pathlib import Path

from appl import get_parser, update_appl_configs
from loguru import logger

from .game24 import Game24Task
from .solver import solve


def main():
    """Main entry point for CLI."""
    data_path = Path(__file__).parent.parent / "data" / "24.csv"
    parser = get_parser()
    parser.add_argument("--data-path", type=str, default=str(data_path))
    parser.add_argument(
        "--task-start-index",
        type=int,
        default=900,
        help="Start index of tasks, included",
    )
    parser.add_argument(
        "--task-end-index",
        type=int,
        default=901,
        help="End index of tasks, not included",
    )
    parser.add_argument(
        "--n-propose-sample",
        type=int,
        default=8,
        help="Number of proposals to generate",
    )
    parser.add_argument(
        "--n-evaluate-sample",
        type=int,
        default=2,
        help="Number of evaluations to generate",
    )  # samples = 3 in the paper
    parser.add_argument(
        "--n-select-sample",
        type=int,
        default=5,
        help="Number of proposals to select",
    )
    args = parser.parse_args()
    update_appl_configs(args.appl)

    task = Game24Task(args.data_path)

    num_solved = 0
    for idx in range(args.task_start_index, args.task_end_index):
        input_numbers = task.get_input(idx)
        candidates = solve(
            input_numbers,
            n_propose_sample=args.n_propose_sample,
            n_evaluate_sample=args.n_evaluate_sample,
            n_select_sample=args.n_select_sample,
        )

        if len(candidates) > 0:
            test_result = task.test_output(idx, candidates[0])
            result_msg = "correct" if test_result else "incorrect"
            logger.info(
                f"Proposed solution {candidates[0]} for {input_numbers} is {result_msg}"
            )
            num_solved += int(test_result)
        else:
            logger.info(f"No solution found for {input_numbers}")

        logger.info(
            f"solved {num_solved} out of {idx - args.task_start_index + 1} tasks, solved rate: {num_solved / (idx - args.task_start_index + 1)}"
        )

    total_tasks = args.task_end_index - args.task_start_index
    logger.info(
        f"[Summary] Solved {num_solved} out of {total_tasks} tasks, solved rate: {num_solved / total_tasks}"
    )


if __name__ == "__main__":
    main()
