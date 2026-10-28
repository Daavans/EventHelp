"""Run all eval test cases against the agent and print a score per category.

Usage: python -m evals.run_evals [--delay SECONDS]
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from src.agent import AgentResult, run_agent

EVALS_DIR = Path(__file__).parent
TESTCASES_PATH = EVALS_DIR / "testcases.json"
RESULTS_PATH = EVALS_DIR / "results.json"


def check(case: dict, result: AgentResult) -> list[str]:
    """Return a list of failed checks for one test case (empty list = passed)."""
    failures = []
    answer = result.answer.lower()
    called_tools = [call["name"] for call in result.tool_calls]

    if "must_contain_any" in case:
        if not any(text.lower() in answer for text in case["must_contain_any"]):
            failures.append(f"answer contains none of {case['must_contain_any']}")

    for text in case.get("must_not_contain", []):
        if text.lower() in answer:
            failures.append(f"answer contains forbidden text '{text}'")

    if case.get("no_tool_calls") and called_tools:
        failures.append(f"expected no tool calls, got {called_tools}")

    # The model may try a tool, as long as our validation stops it (e.g. missing name/email).
    if "tool_must_not_succeed" in case:
        for call in result.tool_calls:
            if call["name"] == case["tool_must_not_succeed"] and "error" not in call["result"]:
                failures.append(f"'{call['name']}' succeeded with {call['args']}")

    if "expected_tool" in case:
        calls = [call for call in result.tool_calls if call["name"] == case["expected_tool"]]
        if not calls:
            failures.append(f"expected tool '{case['expected_tool']}', got {called_tools or 'none'}")
        elif "expected_args" in case:
            args = {key: str(value).lower() for key, value in calls[0]["args"].items()}
            for key, expected in case["expected_args"].items():
                if args.get(key) != str(expected).lower():
                    failures.append(f"expected {key}={expected}, got {args.get(key)}")

    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    # The free tier allows only a few requests per minute, so wait between test cases.
    parser.add_argument("--delay", type=float, default=4.0, help="seconds to wait between test cases")
    delay = parser.parse_args().delay

    testcases = json.loads(TESTCASES_PATH.read_text(encoding="utf-8"))
    scores = defaultdict(lambda: [0, 0])  # category -> [passed, total]
    results = []

    for i, case in enumerate(testcases):
        if i > 0:
            time.sleep(delay)
        try:
            result = run_agent(case["question"])
            failures = check(case, result)
            answer, tool_calls = result.answer, result.tool_calls
        except Exception as e:  # an error counts as a failed test, not a crashed eval run
            failures = [f"error: {e}"]
            answer, tool_calls = "", []

        passed = not failures
        scores[case["category"]][0] += passed
        scores[case["category"]][1] += 1
        results.append(
            {"id": case["id"], "passed": passed, "failures": failures, "answer": answer, "tool_calls": tool_calls}
        )

        print(f"{'PASS' if passed else 'FAIL'}  {case['id']}")
        for failure in failures:
            print(f"      - {failure}")
            print(f"      answer: {answer[:200]!r}")

    print("\nScore per category:")
    for category, (passed, total) in scores.items():
        print(f"  {category:<8} {passed}/{total}")
    total_passed = sum(passed for passed, _ in scores.values())
    print(f"  {'total':<8} {total_passed}/{len(testcases)}")

    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDetailed results written to {RESULTS_PATH.name}")


if __name__ == "__main__":
    main()
