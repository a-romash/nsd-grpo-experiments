"""MATH answer extraction and verifiable binary rewards.

Uses math-verify for symbolic equality when available. If unavailable, only a
conservative string/numeric fallback is used; it is NOT a substitute for
math-verify when reporting benchmark results.
"""
from __future__ import annotations

import re
from fractions import Fraction


def extract_last_boxed(text: str) -> str | None:
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    i = start + len(marker)
    depth = 1
    output = []
    while i < len(text) and depth:
        char = text[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                break
        output.append(char)
        i += 1
    return "".join(output).strip() if depth == 0 else None


def extract_answer(text: str) -> str | None:
    """Only use the final answer, never an arbitrary intermediate calculation."""
    if not isinstance(text, str) or not text.strip():
        return None
    boxed = extract_last_boxed(text)
    if boxed:
        return boxed
    # If generation was truncated without a final answer, this yields None.
    found = re.findall(r"(?im)^\s*(?:final\s+)?answer\s*:\s*(.+?)\s*$", text)
    return found[-1].strip().rstrip(".") if found else None


def _numeric(text: str) -> Fraction | None:
    try:
        return Fraction(text.replace(" ", "").replace(",", ""))
    except (ValueError, ZeroDivisionError):
        return None


def _simple_equal(a: str, b: str) -> bool:
    aa = a.strip().replace(" ", "").replace("$", "")
    bb = b.strip().replace(" ", "").replace("$", "")
    if aa == bb:
        return True
    fa, fb = _numeric(aa), _numeric(bb)
    return fa is not None and fa == fb


def is_correct(completion: str, gold_answer: str) -> bool:
    predicted = extract_answer(completion)
    if predicted is None:
        return False
    gold = extract_last_boxed(gold_answer) or gold_answer
    if _simple_equal(predicted, gold):
        return True
    try:
        from math_verify import LatexExtractionConfig, parse, verify
        # Wrapping forces mathematical answer interpretation for common symbolic forms.
        options = [LatexExtractionConfig()]
        gold_parsed = parse(r"\boxed{" + gold + "}", extraction_config=options)
        pred_parsed = parse(r"\boxed{" + predicted + "}", extraction_config=options)
        return bool(gold_parsed and pred_parsed and verify(gold_parsed, pred_parsed))
    except Exception:
        return False


def accuracy_reward(completions: list[str], answer: list[str], **_: object) -> list[float]:
    return [float(is_correct(c, a)) for c, a in zip(completions, answer)]
