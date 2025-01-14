import argparse
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional

from appl import Generation, SystemMessage, gen, grow, ppl
from appl.compositor import Tagged
from loguru import logger

parser = argparse.ArgumentParser()
parser.add_argument("--num-reflections", type=int, default=1)
parser.add_argument("--num-tests", type=int, default=6)
parser.add_argument("--start-problem", type=int, default=0)
parser.add_argument("--num-problems", type=int, default=None)
args = parser.parse_args()


@dataclass
class RustProblem:
    name: str
    prompt: str
    test: str
    entry_point: str


@dataclass
class TestResult:
    passed: bool
    message: Optional[str] = None

    def display(self) -> str:
        s = "Result: "
        if self.passed:
            s += "Passed"
        else:
            s += "Failed"
            if self.message:
                s += f"\nMessage:\n{self.message}"
        return s


@dataclass
class ProblemState:
    problem: RustProblem
    tests_code: Optional[str] = None
    solutions: List[str] = field(default_factory=list)
    reflections: List[Optional[str]] = field(default_factory=list)
    internal_test_results: List[TestResult] = field(default_factory=list)
    final_test_results: List[TestResult] = field(default_factory=list)
    pass_internal_tests: bool = False
    pass_final_tests: bool = False


def extract_code_block(response: str, tag: Optional[str] = None) -> str:
    pattern = raw_pattern = r"```.*?\n(.*?)\n```"
    if tag:
        pattern = rf"<{tag}>\n{raw_pattern}\s*\n</{tag}>"
    try:
        return re.search(pattern, response, re.DOTALL).group(1)
    except Exception:
        try:
            return re.search(raw_pattern, response, re.DOTALL).group(1)
        except Exception:
            logger.error(f"Error extracting code block: {response}")
            return ""


@ppl
def generate_test_cases(problem: RustProblem) -> Generation:
    SystemMessage(
        """
        You are an expert Rust programmer. Generate comprehensive test cases for the given function 
        using Rust's testing framework. Include edge cases and typical usage scenarios.
        """
    )
    grow("Here is the function signature and description:")
    grow("```rust")
    grow(problem.prompt)
    grow("```")

    grow(
        "Think before writing the test cases and no more explanation is required after the test cases."
        "Your test cases should strictly follow the description of the function. Pay attention to the constraints and edge cases."
    )
    grow(f"You can generate at most {args.num_tests} test cases.")
    grow("wrap your test cases in a code block with the tag `tests`, e.g.")
    with Tagged("tests"):
        grow("```rust")
        grow("<code for tests>")
        grow("```")

    return gen()  # stream=True


@ppl
def generate_reflection(
    problem: RustProblem, solution: str, test_result: TestResult
) -> Generation:
    SystemMessage(
        """You are an expert Rust programmer. Analyze the failed test cases and provide insights 
        on why the solution failed and how it could be improved. Be specific about the issues found.
        """
    )
    grow("Original problem:")
    with Tagged("problem"):
        grow("```rust")
        grow(problem.prompt)
        grow("```")

    grow("\nAttempted solution:")
    with Tagged("attempted_solution"):
        grow("```rust")
        grow(solution)
        grow("```")

    grow("\nTest results:")
    with Tagged("test_results"):
        grow(test_result.display())

    grow(
        "Think before writing the reflection and no more explanation is required after the reflection."
    )
    grow("You should not suggest changes to the name of the function.")
    grow(
        "generate the reflection wrapped in a code block with the tag `reflection`, e.g."
    )
    with Tagged("reflection"):
        grow("```markdown")
        grow("<your reflections>")
        grow("```")

    return gen()  # stream=True


@ppl
def generate_solution(
    problem: RustProblem,
    previous_impl: Optional[str] = None,
    test_messages: Optional[str] = None,
    reflection: Optional[str] = None,
) -> Generation:
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

    if previous_impl:
        grow("\nPrevious attempt implementation:")
        with Tagged("previous_impl"):
            grow(previous_impl)

    if test_messages:
        grow("\nTest messages for previous attempt:")
        with Tagged("test_messages"):
            grow(test_messages)

    if reflection:
        grow("\nReflection on previous attempt:")
        with Tagged("reflection"):
            grow(reflection)

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

    return gen()  # stream=True


def create_cargo_project(tmpdir: str, name: str) -> str:
    """Create a new Cargo project and return the src path"""
    subprocess.run(["cargo", "new", name], cwd=tmpdir, capture_output=True)
    return os.path.join(tmpdir, name, "src")


def test_solution(
    problem: RustProblem,
    solution: str,
    test_code: Optional[str] = None,
    self_generated_test: bool = False,
) -> TestResult:
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = create_cargo_project(tmpdir, "solution")

        # Write solution to lib.rs
        with open(os.path.join(src_path, "main.rs"), "w") as f:
            f.write(solution)
            if test_code:
                f.write("\n\n")
                f.write(test_code)
            else:
                f.write("\n")
                f.write(problem.test)

        # print(solution + "\n" + test_code)

        try:
            # Run cargo test
            if self_generated_test:
                # test-threads=1 to make the tests run sequentially (so their order is likely deterministic)
                test_result = subprocess.run(
                    ["cargo", "test", "--", "--test-threads=1"],
                    cwd=os.path.join(tmpdir, "solution"),
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
            else:
                test_result = subprocess.run(
                    ["cargo", "run", "./main.rs"],
                    cwd=src_path,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )

            if test_result.returncode != 0:
                # remove information including the tmp dir, so that the cache can be reused
                # however, the order of the tests could still be random.
                if "error: could not compile" in test_result.stderr:
                    pattern = re.compile(r"\s*Compiling.*?\n", re.DOTALL)
                    return TestResult(
                        passed=False,
                        message=pattern.sub("", test_result.stderr)
                        + "\n"
                        + test_result.stdout,
                    )
                elif "error: test failed" in test_result.stderr:
                    return TestResult(
                        passed=False,
                        message="Error: test failed" + "\n" + test_result.stdout,
                    )
                else:
                    if self_generated_test:
                        logger.warning("unknown error")
                    return TestResult(
                        passed=False,
                        message=test_result.stderr + "\n" + test_result.stdout,
                    )

            return TestResult(passed=True)

        except Exception as e:
            return TestResult(passed=False, error_message=str(e))


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


def main():
    problems = read_problems("./data/humaneval-rs-hardest50.jsonl")
    if args.start_problem:
        problems = problems[args.start_problem :]
    if args.num_problems:
        problems = problems[: args.num_problems]
    total = len(problems)

    # generate solutions and test cases in parallel
    solutions = [generate_solution(p) for p in problems]
    test_cases = [generate_test_cases(p) for p in problems]
    problem_states = [
        ProblemState(
            problem=p,
            solutions=[extract_code_block(str(solutions[i]), tag="solution")],
            tests_code=extract_code_block(str(test_cases[i]), tag="tests"),
        )
        for i, p in enumerate(problems)
    ]

    scores = []
    for iteration in range(args.num_reflections + 1):
        logger.info(f"\n=== Iteration {iteration} ===")

        reflections = [None] * len(problem_states)
        for i, ps in enumerate(problem_states):
            if not ps.pass_internal_tests:
                test_result = test_solution(
                    ps.problem,
                    ps.solutions[-1],
                    ps.tests_code,
                    self_generated_test=True,
                )
                ps.internal_test_results.append(test_result)
                if test_result.passed or iteration == args.num_reflections:
                    ps.pass_internal_tests = test_result.passed
                else:
                    reflections[i] = generate_reflection(
                        ps.problem, ps.solutions[-1], test_result
                    )

                test_result = test_solution(
                    ps.problem, ps.solutions[-1], ps.problem.test
                )
                ps.final_test_results.append(test_result)
                ps.pass_final_tests = test_result.passed

        if iteration < args.num_reflections:
            for i, ps in enumerate(problem_states):
                if reflections[i]:
                    ps.reflections.append(
                        extract_code_block(
                            str(reflections[i]),
                            tag="reflection",
                        )
                    )

        # Calculate and store score for this iteration
        internal_passed = sum(1 for p in problem_states if p.pass_internal_tests)
        internal_score = (internal_passed / total) * 100
        final_passed = sum(1 for p in problem_states if p.pass_final_tests)
        final_score = (final_passed / total) * 100
        scores.append((final_score, internal_score))
        logger.info(
            f"\nIteration {iteration} Score for internal tests: {internal_score:.2f}% ({internal_passed}/{total} problems solved)"
        )
        logger.info(
            f"\nIteration {iteration} Score for final tests: {final_score:.2f}% ({final_passed}/{total} problems solved)"
        )

        # Generate solutions for next iteration
        if iteration == args.num_reflections:
            break

        solutions_for_problems = [
            None
            if ps.pass_internal_tests
            else generate_solution(
                ps.problem,
                ps.solutions[-1],
                ps.internal_test_results[-1].message,
                ps.reflections[-1],
            )
            for ps in problem_states
        ]
        for ps, solution in zip(problem_states, solutions_for_problems):
            if solution:
                ps.solutions.append(extract_code_block(str(solution), tag="solution"))

    for i, ps in enumerate(problem_states):
        logger.info(f"\n=== Problem {i} ===")
        logger.info(f"Prompt:\n{ps.problem.prompt}")
        logger.info(f"Internal Tests:\n{ps.tests_code}")
        logger.info(f"Internal Test Passed: {ps.pass_internal_tests}")
        logger.info(f"Final Test Passed: {ps.pass_final_tests}")
        for j, solution in enumerate(ps.solutions):
            logger.info(f"Solution {j}: {solution}")
            logger.info(
                f"Internal Test Results:\n{ps.internal_test_results[j].display()}"
            )
            if j < len(ps.reflections):
                logger.info(f"Reflections: {ps.reflections[j]}")
            if j < len(ps.final_test_results):
                logger.info(
                    f"Final Test Results:\n{ps.final_test_results[j].display()}"
                )

    # Print final results
    logger.info("\n=== Final Results ===")
    for i, score in enumerate(scores):
        logger.info(f"Iteration {i} for internal tests: {score[1]:.2f}%")
        logger.info(f"Iteration {i} for final tests: {score[0]:.2f}%")


if __name__ == "__main__":
    main()
