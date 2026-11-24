"""Twin prompts must line up token by token (CLAUDE.md, rule 4): checked, never assumed."""


class TwinError(ValueError):
    """Twins that do not line up as the experiment needs."""


def check(clean: list[int], corrupt: list[int], length: int) -> None:
    """Refuse twins that are not BOS + a copy + (the copy again | fresh tokens), aligned."""
    n = 1 + 2 * length
    if len(clean) != n or len(corrupt) != n:
        raise TwinError(f"twins have {len(clean)} and {len(corrupt)} tokens, expected {n}")
    if clean[: 1 + length] != corrupt[: 1 + length]:
        raise TwinError("twins differ before the second copy")
    first = clean[1 : 1 + length]
    if len(set(first)) != length:
        raise TwinError("the first copy repeats a token, so induction's answer is ambiguous")
    if clean[1 + length :] != first:
        raise TwinError("the clean prompt's second copy is not the first copy")
    if set(corrupt[1 + length :]) & set(first) or len(set(corrupt[1 + length :])) != length:
        raise TwinError("the corrupted second copy is not fresh, distinct tokens")
