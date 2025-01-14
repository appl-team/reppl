"""
Modified based on cursor generated.

Prompt:
based on @humaneval-rs-hardest50.jsonl, write a script that reads prompt and tests from the data, complete the function using LLMs, and then test that and give a final score.
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List

from appl import Generation, SystemMessage, gen, grow, ppl
from appl.compositor import Tagged

parser = argparse.ArgumentParser()
parser.add_argument("--num-problems", type=int, default=None)
args = parser.parse_args()


@dataclass
class RustProblem:
    name: str
    prompt: str
    test: str
    entry_point: str


# Gets 33/50 (66%) correct on humaneval-rs-hardest50 with gpt-4o-mini
# Gets 40/50 (80%) correct on humaneval-rs-hardest50 with gpt-4o-2024-11-20
@ppl
def solve_rust_problem(problem: RustProblem) -> Generation:
    SystemMessage(
        """
        You are an expert Rust programmer. You will be given a programming problem and need to implement a solution in Rust.
        Your solution should be correct and handle all test cases. Focus on writing clean, efficient code that follows Rust best practices.
        """
    )
    grow("Here is the problem description and function signature:")
    grow("```rust")
    grow(problem.prompt)
    grow("```")

    grow("\nPlease implement a solution that passes all test cases.")
    grow(
        f"Your solution should be a self-contained function for {problem.entry_point}. You do not need to include the `main` function in your solution."
    )
    grow(
        "Think before writing the solution and no more explanation is required after the solution."
    )
    grow("wrap your solution in a code block with the tag `solution`, e.g.")
    with Tagged("solution"):
        grow("```rust")
        grow("<code for solution>")
        grow("```")

    return gen()


def read_problems(file_path: str) -> List[RustProblem]:
    problems = []
    with open(file_path) as f:
        for line in f:
            data = json.loads(line)
            problems.append(
                RustProblem(
                    name=data["name"],
                    prompt=data["prompt"],
                    test=data["test"],
                    entry_point=data["entry_point"],
                )
            )
    return problems


def test_solution(problem: RustProblem, solution: str) -> bool:
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create the source file
        src_path = os.path.join(tmpdir, "solution.rs")
        with open(src_path, "w") as f:
            # Write the complete Rust program
            f.write(solution)
            f.write("\n")
            f.write(problem.test)

        # Try to compile and run
        try:
            # Compile
            compile_result = subprocess.run(
                ["rustc", src_path], cwd=tmpdir, capture_output=True, text=True
            )
            if compile_result.returncode != 0:
                print(f"Compilation failed for {problem.name}:")
                print(compile_result.stderr)
                return False

            # Run
            run_result = subprocess.run(
                ["./solution"], cwd=tmpdir, capture_output=True, text=True, timeout=3
            )
            if run_result.returncode != 0:
                print(f"Test failed for {problem.name}:")
                print(run_result.stderr)
                return False

            return True

        except Exception as e:
            print(f"Error testing {problem.name}: {str(e)}")
            return False


def main():
    # Read problems from the JSONL file
    problems = read_problems("./data/humaneval-rs-hardest50.jsonl")

    if args.num_problems:
        problems = problems[: args.num_problems]

    total = len(problems)
    passed = 0

    # get solutions in parallel
    responses = [solve_rust_problem(problem) for problem in problems]

    for problem, response in zip(problems, responses):
        print(f"\nevaluating {problem.name}...")

        code_block_pattern = r"<solution>\n```.*?\n(.*?)\n```\n</solution>"
        try:
            solution = re.search(code_block_pattern, str(response), re.DOTALL).group(1)

            # Test the solution
            success = test_solution(problem, solution)
        except Exception as e:
            success = False
            print(f"Error testing solution for {problem.name}: {str(e)}")

        if success:
            passed += 1
            print(f"✓ {problem.name} passed!")
        else:
            print(f"✗ {problem.name} failed")

    # Calculate and print final score
    score = (passed / total) * 100
    print(f"\nFinal Score: {score:.2f}% ({passed}/{total} problems solved)")


if __name__ == "__main__":
    main()
