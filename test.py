#!/usr/bin/env python3
import os
from dataclasses import dataclass
from typing import List, Tuple

try:
    from openai import OpenAI
except ImportError as e:
    raise SystemExit("Missing dependency. Install with: pip install openai") from e


MODEL = "deepseek-chat"
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/beta")


@dataclass(frozen=True)
class SentencePair:
    name: str
    prefix: str
    continuation: str


PAIRS: List[SentencePair] = [
    SentencePair(
        name="Forward",
        prefix="他午饭吃了很多，所以",
        continuation="他不想吃晚饭",
    ),
    SentencePair(
        name="Backward",
        prefix="他不想吃晚饭，因为",
        continuation="他午饭吃了很多",
    ),
]


def get_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Set DEEPSEEK_API_KEY in your environment.")
    return OpenAI(api_key=api_key, base_url=BASE_URL)


def _field(obj, name):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def fetch_prompt_logprobs(client: OpenAI, prompt: str):
    # Use echo=True + max_tokens=0 to score prompt tokens without generating.
    response = client.completions.create(
        model=MODEL,
        prompt=prompt,
        max_tokens=0,
        temperature=0,
        logprobs=1,
        echo=True,
    )
    choice = response.choices[0]
    logprobs = _field(choice, "logprobs")
    if logprobs is None:
        raise RuntimeError("No logprobs returned. Check API support for logprobs/echo.")
    return logprobs


def _continuation_indices(
    prefix: str, prompt: str, tokens: List[str], offsets: List[int]
) -> List[int]:
    if not tokens:
        raise RuntimeError("No tokens returned in logprobs.")

    # Prefer offsets if they align; otherwise fall back to reconstruction.
    if offsets:
        full_chars = len(prompt)
        full_bytes = len(prompt.encode("utf-8"))
        last_off = offsets[-1]
        last_tok = tokens[-1]

        if last_off + len(last_tok) == full_chars:
            boundary = len(prefix)
            tok_len = lambda t: len(t)
        elif last_off + len(last_tok.encode("utf-8")) == full_bytes:
            boundary = len(prefix.encode("utf-8"))
            tok_len = lambda t: len(t.encode("utf-8"))
        else:
            offsets = None

    if offsets:
        idxs = [i for i, off in enumerate(offsets) if off >= boundary]
        if idxs:
            return idxs

        # Handle rare case where a token spans the boundary.
        for i, off in enumerate(offsets):
            if off < boundary < off + tok_len(tokens[i]):
                return list(range(i, len(tokens)))

    # Fallback: reconstruct by concatenating tokens (character-based).
    boundary = len(prefix)
    idxs = []
    pos = 0
    for i, tok in enumerate(tokens):
        start = pos
        end = start + len(tok)
        pos = end
        if end <= boundary:
            continue
        idxs.append(i)
    return idxs


def average_logprob_for_continuation(
    client: OpenAI, prefix: str, continuation: str
) -> Tuple[float, int]:
    prompt = prefix + continuation
    logprobs = fetch_prompt_logprobs(client, prompt)

    tokens = _field(logprobs, "tokens") or []
    token_logprobs = _field(logprobs, "token_logprobs") or []
    offsets = _field(logprobs, "text_offset") or []

    if len(tokens) != len(token_logprobs):
        raise RuntimeError("Token/logprob length mismatch.")

    idxs = _continuation_indices(prefix, prompt, tokens, offsets)
    cont_lps = [token_logprobs[i] for i in idxs]

    if any(lp is None for lp in cont_lps):
        raise RuntimeError("Missing token logprobs for continuation tokens.")

    avg = sum(cont_lps) / len(cont_lps)
    return avg, len(cont_lps)


def main() -> None:
    client = get_client()
    results = {}

    for pair in PAIRS:
        avg, _ = average_logprob_for_continuation(client, pair.prefix, pair.continuation)
        results[pair.name] = avg

    forward = results["Forward"]
    backward = results["Backward"]
    diff = forward - backward

    print(f"Forward average log-prob: {forward:.6f}")
    print(f"Backward average log-prob: {backward:.6f}")
    print(f"Difference (Forward - Backward): {diff:.6f}")


if __name__ == "__main__":
    main()
```
