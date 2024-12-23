"""Example prompts for Game of 24 task."""

# Examples for proposing next steps
PROPOSE_EXAMPLES = [
    {
        "Input": "2 8 8 14",
        "Possible next steps": [
            "2 + 8 = 10 (left: 8 10 14)",
            "8 / 2 = 4 (left: 4 8 14)",
            "14 + 2 = 16 (left: 8 8 16)",
            "2 * 8 = 16 (left: 8 14 16)",
            "8 - 2 = 6 (left: 6 8 14)",
            "14 - 8 = 6 (left: 2 6 8)",
            "14 / 2 = 7 (left: 7 8 8)",
            "14 - 2 = 12 (left: 8 8 12)",
        ],
    }
]

# Examples for evaluating states
VALUE_EXAMPLES = [
    {"Input": "10 14", "Thoughts": ["10 + 14 = 24"], "Evaluation": "sure"},
    {
        "Input": "11 12",
        "Thoughts": ["11 + 12 = 23", "12 - 11 = 1", "11 * 12 = 132", "11 / 12 = 0.91"],
        "Evaluation": "impossible",
    },
    {
        "Input": "4 4 10",
        "Thoughts": ["4 + 4 + 10 = 18", "4 * 10 - 4 = 36", "(10 - 4) * 4 = 24"],
        "Evaluation": "sure",
    },
    {"Input": "4 9 11", "Thoughts": ["9 + 11 + 4 = 24"], "Evaluation": "sure"},
    {
        "Input": "5 7 8",
        "Thoughts": [
            "5 + 7 + 8 = 20",
            "(8 - 5) * 7 = 21",
            "I cannot obtain 24 now, but numbers are within a reasonable range",
        ],
        "Evaluation": "likely",
    },
    {
        "Input": "5 6 6",
        "Thoughts": [
            "5 + 6 + 6 = 17",
            "(6 - 5) * 6 = 6",
            "I cannot obtain 24 now, but numbers are within a reasonable range",
        ],
        "Evaluation": "likely",
    },
    {
        "Input": "10 10 11",
        "Thoughts": [
            "10 + 10 + 11 = 31",
            "(11 - 10) * 10 = 10",
            "10 10 10 are all too big",
        ],
        "Evaluation": "impossible",
    },
    {
        "Input": "1 3 3",
        "Thoughts": ["1 * 3 * 3 = 9", "(1 + 3) * 3 = 12", "1 3 3 are all too small"],
        "Evaluation": "impossible",
    },
]

# 5-shot Examples for chain-of-thought reasoning
COT_EXAMPLES = [
    {
        "Input": "4 4 6 8",
        "Steps": [
            "4 + 8 = 12 (left: 4 6 12)",
            "6 - 4 = 2 (left: 2 12)",
            "2 * 12 = 24 (left: 24)",
        ],
        "Answer": "(6 - 4) * (4 + 8) = 24",
    },
    {
        "Input": "2 9 10 12",
        "Steps": [
            "12 * 2 = 24 (left: 9 10 24)",
            "10 - 9 = 1 (left: 1 24)",
            "24 * 1 = 24 (left: 24)",
        ],
        "Answer": "(12 * 2) * (10 - 9) = 24",
    },
    {
        "Input": "4 9 10 13",
        "Steps": [
            "13 - 10 = 3 (left: 3 4 9)",
            "9 - 3 = 6 (left: 4 6)",
            "4 * 6 = 24 (left: 24)",
        ],
        "Answer": "4 * (9 - (13 - 10)) = 24",
    },
    {
        "Input": "1 4 8 8",
        "Steps": [
            "8 / 4 = 2 (left: 1 2 8)",
            "1 + 2 = 3 (left: 3 8)",
            "3 * 8 = 24 (left: 24)",
        ],
        "Answer": "(1 + 8 / 4) * 8 = 24",
    },
    {
        "Input": "5 5 5 9",
        "Steps": [
            "5 + 5 = 10 (left: 5 9 10)",
            "10 + 5 = 15 (left: 9 15)",
            "15 + 9 = 24 (left: 24)",
        ],
        "Answer": "((5 + 5) + 5) + 9 = 24",
    },
]

# Examples for judging solutions
JUDGE_EXAMPLES = [
    {
        "Input": "4 4 6 8",
        "Answer": "(4 + 8) * (6 - 4) = 24",
        "Judge": "sure",
    },
    {
        "Input": "4 9 10 13",
        "Answer": "(13 - 9) * (10 - 4) = 24",
        "Judge": "sure",
    },
    {
        "Input": "4 4 6 8",
        "Answer": "(4 + 8) * (6 - 4) + 1 = 25",
        "Judge": "impossible",
    },
    {
        "Input": "2 9 10 12",
        "Answer": "2 * (12 - 10) = 24",
        "Judge": "impossible",
    },
    {
        "Input": "4 9 10 13",
        "Answer": "(13 - 4) * (10 - 9) = 24",
        "Judge": "impossible",
    },
    {
        "Input": "2 9 10 12",
        "Answer": "2 * 12 * (10 - 9) = 24",
        "Judge": "sure",
    },
]
