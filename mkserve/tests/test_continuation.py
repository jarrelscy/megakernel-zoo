"""python3 -m unittest mkserve/tests/test_continuation.py (from the repo root): continuing a reasoning a thinking budget
cut (mkserve/continuation.py). Standard library only: no engine, no tokenizer, no MLX."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mkserve.continuation import reasoning_after, split_continuation  # noqa: E402

END = 7   # the end-of-message token that closes a reasoning


def encode(text: str) -> list[int]:
    return [100 + ord(c) for c in text]


PROMPT = [{"role": "system", "content": "You write reports."}, {"role": "user", "content": "Comparison made to prior CT."}]


class SplitContinuation(unittest.TestCase):
    def test_the_conversation_before_the_reasoning_and_its_token_ids(self):
        messages = PROMPT + [{"role": "assistant", "content": "", "reasoning_content": "We need", "mk_reasoning_ids": [11, 12, 13]}]
        before, ids = split_continuation(messages, encode, END)
        self.assertEqual(before, PROMPT)
        self.assertEqual(ids, [11, 12, 13])

    def test_the_ids_are_preferred_to_the_text(self):
        messages = PROMPT + [{"role": "assistant", "content": None, "reasoning_content": "ab", "mk_reasoning_ids": [5]}]
        self.assertEqual(split_continuation(messages, encode, END)[1], [5])

    def test_without_ids_the_text_is_encoded(self):
        messages = PROMPT + [{"role": "assistant", "content": "", "reasoning_content": "ab"}]
        self.assertEqual(split_continuation(messages, encode, END)[1], encode("ab"))

    def test_refused_when_the_last_message_is_not_reasoning_alone(self):
        cases = [
            PROMPT,                                                                       # no assistant message last
            PROMPT + [{"role": "assistant", "content": "text", "mk_reasoning_ids": [1]}],  # an answer, not reasoning
            PROMPT + [{"role": "assistant", "content": "", "tool_calls": [{"id": "x"}], "mk_reasoning_ids": [1]}],
            PROMPT + [{"role": "assistant", "content": ""}],                               # nothing to continue
            PROMPT + [{"role": "assistant", "content": "", "mk_reasoning_ids": "1,2"}],
            PROMPT + [{"role": "assistant", "content": "", "mk_reasoning_ids": [1, END, 2]}],  # already closed
            [{"role": "assistant", "content": "", "mk_reasoning_ids": [1]}],               # no prompt before it
        ]
        for messages in cases:
            with self.subTest(messages=messages[-1]):
                with self.assertRaises(ValueError):
                    split_continuation(messages, encode, END)


class ReasoningAfter(unittest.TestCase):
    def test_a_reasoning_the_budget_cut(self):
        self.assertEqual(reasoning_after([], [1, 2, 3], END), ([1, 2, 3], True))

    def test_a_reasoning_the_model_closed(self):
        self.assertEqual(reasoning_after([], [1, 2, END], END), ([1, 2], False))

    def test_a_continued_reasoning_is_the_whole_reasoning(self):
        self.assertEqual(reasoning_after([1, 2], [3, 4], END), ([1, 2, 3, 4], True))
        self.assertEqual(reasoning_after([1, 2], [3, END], END), ([1, 2, 3], False))

    def test_the_continued_prompt_is_the_first_runs_prompt_and_reasoning(self):
        """What chat_completions prefills to continue is the first request's prompt, the reasoning opener and the
        reasoning's tokens: a prefix of what the first request ran, so the prefix cache holds it all, and greedy
        decoding goes on as one run with the two budgets together would have."""
        prompt, opener = [50, 51, 52], [1, 2]
        first, cut = reasoning_after([], [60, 61, 62, 63], END)          # budget 4, cut
        first_run = prompt + opener + first + [END]                       # what the first answer continued from
        self.assertTrue(cut)
        messages = PROMPT + [{"role": "assistant", "content": "", "mk_reasoning_ids": first}]
        _before, prior = split_continuation(messages, encode, END)
        continued = prompt + opener + prior
        self.assertEqual(first_run[:len(continued)], continued)
        whole, cut_again = reasoning_after(prior, [64, END], END)
        self.assertEqual((whole, cut_again), ([60, 61, 62, 63, 64], False))


if __name__ == "__main__":
    unittest.main()
