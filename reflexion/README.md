# Reflexion Re-implementation

This is a re-implementation of the [Reflexion](https://arxiv.org/pdf/2303.11366) algorithm ([original repo](https://github.com/noahshinn/reflexion)) using [APPL](https://github.com/appl-team/appl). Reflexion is a framework that enables language models to self-reflect on their mistakes and gradually improve their performance through an iterative refinement process.

## Features

The implementation leverages APPL's capabilities for:

- Seamless prompt integration into code
- Parallelized LLM calls for efficiency
- Caching and tracing for efficient debugging

## Results

Using `gpt-4o-mini` to solve hardest 50 problems in `HumanEval-RS`, the implementation achieves:

**Final Test Pass Rates:**
- No reflection: 68%
- After 1 reflection: 74% 
- After 2 reflections: 76%
- After 3 reflections: 76%

**Internal Test Pass Rates:**
- No reflection: 26%
- After 1 reflection: 34%
- After 2 reflections: 38% 
- After 3 reflections: 40%

These results demonstrate how reflection helps improve both internal test coverage and final test performance.

## Usage

```bash
# Run with default settings (1 reflection, 6 tests)
python reflexion.py

# Run with custom settings
python reflexion.py --num-reflections 3 --num-tests 10 --start-problem 0
```

Arguments:
- `--num-reflections`: Number of reflection rounds (default: 1)
- `--num-tests`: Number of test cases to run (default: 6) 
- `--start-problem`: Starting problem index (default: 0)
- `--num-problems`: Number of problems to solve (default: None)

## Implementation Details

The implementation follows the Reflexion framework with four main steps:

1. **Initial Solution**: The model attempts to solve the programming task
2. **Test & Feedback**: The solution is tested and test messages are collected as feedback
3. **Reflection**: The model reflects on test failures and refines its solution
4. **Refinement**: The model refines its solution based on the reflection. Proceed to the step 2 (test) again.

Each step is implemented using APPL's prompting system:

```python
@ppl
def generate_test_cases(problem: RustProblem) -> Generation:
    SystemMessage("You are an expert Rust programmer.")
    "Here is the function signature and description:"
    "```rust"
    problem.prompt
    "```"

    # ... omitted for clarity ...

    f"You can generate at most {args.num_tests} test cases."
    "wrap your test cases in a code block with the tag `tests`, e.g."
    with Tagged("tests"):
        "```rust"
        "<code for tests>"
        "```"

    return gen()

@ppl
def generate_reflection(
    problem: RustProblem, solution: str, test_result: TestResult
) -> Generation:
    SystemMessage("You are an expert Rust programmer."
)
    "Original problem:"
    with Tagged("problem"):
        "```rust"
        problem.prompt
        "```"

    "Attempted solution:"
    with Tagged("attempted_solution"):
        "```rust"
        solution
        "```"

    "Test results:"
    with Tagged("test_results"):
        test_result.display()

    # ... omitted for clarity ...

    "generate the reflection wrapped in a code block with the tag `reflection`, e.g."
    with Tagged("reflection"):
        "```markdown"
        "<your reflections>"
        "```"

    return gen()

@ppl
def generate_solution(
    problem: RustProblem,
    previous_impl: Optional[str] = None,
    test_messages: Optional[str] = None,
    reflection: Optional[str] = None,
) -> Generation:
    SystemMessage("You are an expert Rust programmer.")
    "Here is the problem description and function signature:"
    "```rust"
    problem.prompt
    "```"

    if previous_impl:
        "\nPrevious attempt implementation:"
        with Tagged("previous_impl"):
            previous_impl

    if test_messages:
        "\nTest messages for previous attempt:"
        with Tagged("test_messages"):
            test_messages

    if reflection:
        "\nReflection on previous attempt:"
        with Tagged("reflection"):
            reflection

    # ... omitted for clarity ...

    "wrap your solution in a code block with the tag `solution`, e.g."
    with Tagged("solution"):
        "```rust"
        "<code for solution>"
        "```"

    return gen()
```

The reflection process helps the model understand its mistakes and generate improved solutions in subsequent iterations.

## Note

The implementation is based on the [official code](https://github.com/noahshinn/reflexion), but not strictly following the original implementation (e.g. some prompts are slightly modified).

The main purpose of this implementation is to provide a clear overview of the Reflexion algorithm and demonstrate how it can be implemented using APPL.

## License

MIT
