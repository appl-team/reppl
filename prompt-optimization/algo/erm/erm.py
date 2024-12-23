import argparse
import json
import random
import re
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from appl import AIRole, gen, grow, ppl, traceable
from appl.compositor import Tagged
from loguru import logger
from pydantic import BaseModel


@dataclass
class Feedback:
    content: str
    score: float = 0.8  # Initial score


@dataclass
class Exemplar:
    text: str
    label: str
    solution: str
    score: float = 0.8  # Initial score


@dataclass
class MemoryBase:
    """Base class for memory components with shared functionality"""

    temperature: float = 0.5  # Controls randomness in selection
    threshold: float = 0.3  # Minimum score threshold
    beta: float = 0.2  # Learning rate for score updates

    def _update_score(
        self, score: float, improved: bool, performance_gain: float
    ) -> float:
        """Update score based on performance

        Args:
            score: Current score
            improved: Whether performance improved
            performance_gain: Amount of improvement/degradation
        Returns:
            Updated score
        """
        if improved:
            # Increase score based on performance gain
            gain = max(0.0, min(1.0, performance_gain))
            score = (1 - self.beta) * score + self.beta * (1.0 + gain)
        else:
            # Decrease score more if performance dropped significantly
            penalty = max(0.0, min(1.0, abs(performance_gain)))
            score = (1 - self.beta) * score - self.beta * penalty

        # Ensure score stays between 0 and 1
        return max(0.0, min(1.0, score))

    def _get_selection_probs(self, scores: np.ndarray) -> np.ndarray:
        """Convert scores to selection probabilities using softmax

        Args:
            scores: Array of scores
        Returns:
            Array of selection probabilities
        """
        probs = np.exp(scores / self.temperature)
        return probs / np.sum(probs)


@dataclass
class FeedbackMemory(MemoryBase):
    feedbacks: List[Feedback] = field(default_factory=list)

    def add_feedback(self, feedback: str):
        """Add new feedback to memory"""
        self.feedbacks.append(Feedback(content=feedback))

    def retrieve_feedbacks(self, n: int = 3) -> List[str]:
        """Retrieve top n feedbacks based on scores"""
        if not self.feedbacks:
            return []

        scores = np.array([f.score for f in self.feedbacks])
        probs = self._get_selection_probs(scores)

        indices = np.random.choice(
            len(self.feedbacks),
            size=min(n, len(self.feedbacks)),
            p=probs,
            replace=False,
        )
        return [self.feedbacks[i].content for i in indices]

    def update_scores(
        self, feedbacks: List[str], improved: bool, performance_gain: float = 0.0
    ):
        """Update feedback scores based on performance"""
        for fb in self.feedbacks:
            if fb.content in feedbacks:
                fb.score = self._update_score(fb.score, improved, performance_gain)

        # Remove low scoring feedbacks (selective forgetting)
        self.feedbacks = [f for f in self.feedbacks if f.score >= self.threshold]


@dataclass
class ExemplarFactory(MemoryBase):
    exemplars: List[Exemplar] = field(default_factory=list)

    def add_exemplar(self, text: str, label: str, solution: str):
        """Add new exemplar to factory"""
        self.exemplars.append(Exemplar(text=text, label=label, solution=solution))

    def retrieve_exemplars(self, n: int = 5) -> List[Tuple[str, str, str]]:
        """Retrieve top n exemplars based on scores"""
        if not self.exemplars:
            return []

        scores = np.array([e.score for e in self.exemplars])
        probs = self._get_selection_probs(scores)

        indices = np.random.choice(
            len(self.exemplars),
            size=min(n, len(self.exemplars)),
            p=probs,
            replace=False,
        )
        return [
            (
                self.exemplars[i].text,
                self.exemplars[i].label,
                self.exemplars[i].solution,
            )
            for i in indices
        ]

    def update_scores(
        self,
        exemplars: List[Tuple[str, str, str]],
        improved: bool,
        performance_gain: float = 0.0,
    ):
        """Update exemplar scores based on performance"""
        for e in self.exemplars:
            if (e.text, e.label, e.solution) in exemplars:
                e.score = self._update_score(e.score, improved, performance_gain)

        # Remove low scoring exemplars (selective forgetting)
        self.exemplars = [e for e in self.exemplars if e.score >= self.threshold]


class ExemplarResponse(BaseModel):
    text: str
    label: str
    solution: str


class FeedbackResponse(BaseModel):
    feedback: str


@ppl
def get_exemplars_and_feedback(
    prompt: str,
    error_samples: List[Dict],
    num_exemplars: int = 4,
    num_feedbacks: int = 3,
) -> Tuple[List[ExemplarResponse], List[FeedbackResponse]]:
    """Get exemplars and feedback using the instructive meta-prompt"""

    grow(
        "I'm trying to write and complete a zero-shot classifier prompt from difficult or erroneous examples, "
        "'text' field means model input, 'label' field means true label."
    )
    grow(f"The current prompt is:\n{prompt}")
    grow("But this prompt gets the following examples wrong:")
    for sample in error_samples:
        grow(f"text: {sample['input']}")
        grow(f"label: {sample['target']}")

    grow(
        f"To improve my understanding and performance, I would like to identify {num_exemplars} "
        "typical examples from the above cases where the current prompt fails."
    )
    grow("These examples should be diverse to cover a range of different issues.")
    grow(
        "For each example, provide the following format in JSON and wrap each example with "
        "<key_example> and </key_example>:"
    )
    with Tagged("key_example"):
        grow("{")
        grow('"text": "{{input}}",')
        grow('"label": "{{label}}",')
        grow(
            '"solution": "How to solve this problem step-by-step to get a more accurate answer."'
        )
        grow("}")

    grow(
        f"After identifying these {num_exemplars} typical examples, please provide {num_feedbacks} "
        "reasons why the prompt could have gotten these examples wrong. Wrap each reason with "
        "<feedback> and </feedback>."
    )

    response = gen("large")

    # Parse exemplars and feedback from response
    exemplars = []
    feedbacks = []

    # Extract exemplars between <key_example> tags
    import re

    exemplar_matches = re.finditer(
        r"<key_example>(.*?)</key_example>", str(response), re.DOTALL
    )
    for match in exemplar_matches:
        try:
            exemplar_dict = json.loads(match.group(1))
            exemplars.append(ExemplarResponse(**exemplar_dict))
        except Exception:
            continue

    # Extract feedback between <feedback> tags
    feedback_matches = re.finditer(
        r"<feedback>(.*?)</feedback>", str(response), re.DOTALL
    )
    for match in feedback_matches:
        feedbacks.append(FeedbackResponse(feedback=match.group(1).strip()))

    return exemplars, feedbacks


@ppl
def optimize_prompt(prompt: str, error_samples: List[Dict], feedbacks: List[str]):
    """Generate optimized prompt using feedback"""

    grow(
        "I'm trying to write and complete a zero-shot classifier prompt from difficult or erroneous examples, "
        "'text' field means model input, 'label' field means true label."
    )
    grow(f"The current prompt is: {prompt}")
    grow("But this prompt gets the following examples wrong:")
    for sample in error_samples:
        grow(f"Text: {sample['input']}")
        grow(f"Label: {sample['target']}")

    grow("Based on these examples the problem with this prompt is that:")
    for fb in feedbacks:
        grow(fb)

    grow(
        "Based on the above information, I refine the prompt to make the model predict correctly."
    )
    grow(
        "The refined prompt is wrapped with <prompt> and </prompt>, less that 512 words:"
    )

    response = None
    for i in range(5):
        try:
            response = str(gen("large"))
            with AIRole():
                grow(response.format(input="placeholder"))

            # Extract prompt between <prompt> tags

            match = re.search(r"<prompt>(.*?)</prompt>", response, re.DOTALL)
            if match:
                return match.group(1).strip()
            elif i == 0:
                grow("Please follow the format exactly.")
            else:
                grow("you have failed on the <prompt> syntax, please try again.")

        except KeyError as e:
            logger.warning(f"Attempt to produce prompt with extra variables: {e}")
            if i == 0:
                grow("The prompt can only contain `{input}` as the only variable.")
            else:
                grow("you have failed on including extra variables, please try again.")

    return prompt


def load_data(data_path: str) -> List[Dict]:
    """Load data from jsonl file"""
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


class ERM:
    def __init__(
        self,
        feedback_memory: Optional[FeedbackMemory] = None,
        exemplar_factory: Optional[ExemplarFactory] = None,
    ):
        self.feedback_memory = feedback_memory or FeedbackMemory()
        self.exemplar_factory = exemplar_factory or ExemplarFactory()

    def optimize(
        self,
        initial_prompt: str,
        train_data: List[Dict[str, str]],
        test_data: List[Dict[str, str]],
        num_steps: int = 10,
    ) -> Tuple[float, str]:
        """Main optimization loop"""

        best_prompt = current_prompt = initial_prompt
        train_results = self.evaluate(best_prompt, train_data)
        best_score = sum(train_results) / len(train_results)
        logger.info(f"Initial score: {best_score}")

        for step in range(num_steps):
            # Get error samples
            error_samples = self.get_error_samples(train_data, train_results)
            if not error_samples:
                logger.info("No errors found, optimization complete")
                break

            # Get exemplars and feedback
            exemplars, feedbacks = get_exemplars_and_feedback(
                current_prompt, error_samples
            )

            # Store exemplars and feedback
            for ex in exemplars:
                self.exemplar_factory.add_exemplar(ex.text, ex.label, ex.solution)
            for fb in feedbacks:
                self.feedback_memory.add_feedback(fb.feedback)

            # Retrieve stored feedback
            stored_feedbacks = self.feedback_memory.retrieve_feedbacks()

            # Generate new prompt
            current_prompt = optimize_prompt(
                current_prompt, error_samples, stored_feedbacks
            )

            # Evaluate periodically
            test_results = self.evaluate(current_prompt, test_data)
            logger.info(
                f"Step {step + 1}, Score: {sum(test_results) / len(test_results)}"
            )

            # Calculate performance gain/loss
            performance_gain = (
                sum(test_results) / len(test_results) - best_score
            ) / max(1e-6, best_score)

            # Update memory scores with performance gain
            improved = sum(test_results) / len(test_results) > best_score
            self.feedback_memory.update_scores(
                stored_feedbacks, improved, performance_gain
            )
            self.exemplar_factory.update_scores(exemplars, improved, performance_gain)

            if improved:
                best_score = sum(test_results) / len(test_results)
                best_prompt = current_prompt

        return best_score, best_prompt

    def get_error_samples(
        self, data: List[Dict[str, str]], results: List[bool], max_samples: int = 5
    ) -> List[Dict[str, str]]:
        """Get samples where current prompt produces incorrect answers"""
        error_samples = [
            sample for sample, is_correct in zip(data, results) if not is_correct
        ]
        return error_samples[:max_samples]

    @ppl
    def evaluate_prompt(self, prompt: str, data: str) -> float:
        grow(prompt.format(input=data))
        return gen("small")

    @traceable
    def evaluate(self, prompt: str, data: List[Dict[str, str]]) -> List[bool]:
        """Evaluate current prompt on a list of samples"""

        def is_target(response, target):
            # TODO: use better metric
            return target.lower() in str(response).split("\n")[-1].lower()

        results = [self.evaluate_prompt(prompt, sample["input"]) for sample in data]
        is_correct = []
        for r, d in zip(results, data):
            is_correct.append(is_target(r, d["target"]))

        return is_correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="bbh/navigate")
    parser.add_argument("--data-path", type=str, default="./data")
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--num-train-samples", type=int, default=None)
    parser.add_argument("--num-val-samples", type=int, default=None)
    parser.add_argument("--num-test-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    # Need to set the seed to use the trace for reproducibility
    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load data
    train_data = load_data(f"{args.data_path}/{args.dataset}/train.jsonl")
    val_data = load_data(f"{args.data_path}/{args.dataset}/eval.jsonl")
    test_data = load_data(f"{args.data_path}/{args.dataset}/test.jsonl")
    if args.num_train_samples:
        train_data = train_data[: args.num_train_samples]
    if args.num_val_samples:
        val_data = val_data[: args.num_val_samples]
    if args.num_test_samples:
        test_data = test_data[: args.num_test_samples]

    # Initial prompt from paper
    initial_prompt = textwrap.dedent(
        """
        ## Task
        If you follow these instructions, do you return to the starting point?
        ## Output format
        Answer Yes or No as labels.
        ## Prediction
        Text: {input}\n
        Label:
        """
    ).strip()

    # Initialize ERM
    erm = ERM()

    # Run optimization
    final_score, final_prompt = erm.optimize(
        initial_prompt,
        train_data,
        val_data,
        num_steps=args.num_steps,
    )
    logger.info(f"Final score on validation set: {final_score}")
    logger.info(f"Final prompt:\n{final_prompt}")

    init_test_results = erm.evaluate(initial_prompt, test_data)
    init_test_score = sum(init_test_results) / len(init_test_results)
    logger.info(f"Initial score on test set: {init_test_score}")

    final_test_results = erm.evaluate(final_prompt, test_data)
    final_test_score = sum(final_test_results) / len(final_test_results)
    logger.info(f"Final score on test set: {final_test_score}")


if __name__ == "__main__":
    main()
