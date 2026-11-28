"""Continuing a reasoning a thinking budget cut (chat completions, `"mk_continue_reasoning": true`).

A request with a thinking budget reasons for at most that many tokens; mkserve closes the reasoning there and the model
answers (protocols.chat_completions). Its answer carries the reasoning's own token ids (`mk_reasoning_ids`) and whether
the budget cut it (`mk_reasoning_cut`). A client that wants more reasoning sends the same request again with one more
message last, the assistant's reasoning so far:

    {"role": "assistant", "content": "", "reasoning_content": "<the reasoning text>", "mk_reasoning_ids": [...]}

and `"mk_continue_reasoning": true`. mkserve renders the conversation before that message exactly as the first request
rendered it, opens the reasoning, appends the given tokens and lets the model go on reasoning for at most
`thinking_budget` NEW tokens, then closes it and the model answers. The prompt and the reasoning are the first request's
token for token, so the engine's prefix cache holds all of it, and under greedy decoding the reasoning goes on exactly as
a request with the larger budget would have.

The token ids are what make it exact: the reasoning text in the answer is decoded with the special tokens left out and
trimmed, and a model may write a word in tokens that encoding the text would not reproduce. Without
`mk_reasoning_ids` the text is encoded, which may differ from what the model wrote (the cache then holds less of it).

No imports beyond the standard library: unit tested without the engine (mkserve/tests/test_continuation.py).
"""

from __future__ import annotations

from typing import Callable


def split_continuation(messages: list, encode: Callable[[str], list[int]], end_id: int) -> tuple[list, list[int]]:
    """(the conversation before the final assistant message, the token ids of that message's reasoning) for a request
    that continues a reasoning. Raises ValueError for a request that does not end with such a message."""
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("mk_continue_reasoning: the conversation must end with the assistant's reasoning, after the prompt")
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "assistant":
        raise ValueError("mk_continue_reasoning: the last message must be the assistant's reasoning")
    if last.get("content") or last.get("tool_calls"):
        raise ValueError("mk_continue_reasoning: the last message holds only reasoning (no content, no tool calls)")
    given = last.get("mk_reasoning_ids")
    if given is not None:
        if not isinstance(given, list) or not all(isinstance(t, int) and not isinstance(t, bool) for t in given):
            raise ValueError("mk_continue_reasoning: mk_reasoning_ids must be a list of token ids")
        ids = list(given)
    else:
        text = last.get("reasoning_content") or last.get("reasoning") or ""
        if not isinstance(text, str):
            raise ValueError("mk_continue_reasoning: reasoning_content must be text")
        ids = encode(text) if text else []
    if not ids:
        raise ValueError("mk_continue_reasoning: the last message has no reasoning to continue")
    if end_id in ids:
        raise ValueError("mk_continue_reasoning: the reasoning to continue is already closed")
    return messages[:-1], ids


def reasoning_after(prior: list[int], generated: list[int], end_id: int) -> tuple[list[int], bool]:
    """(the whole reasoning's token ids, whether the budget cut it) after a budgeted reasoning run that started after
    `prior` and generated `generated` (its stop token, the end of the reasoning, included when the model wrote it)."""
    closed = end_id in generated
    new = generated[:generated.index(end_id)] if closed else list(generated)
    return list(prior) + new, not closed
