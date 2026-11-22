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
    labels = list(re.finditer(r"(?im)^[ \t]*(?:final[ \t]+)?answer[ \t]*:", text))
    if not labels:
        return extract_last_boxed(text) or None
    final = text[labels[-1].end():].strip()
    if not final:
        return None
    first_line = final.splitlines()[0].strip()
    if r"\boxed{" in first_line or first_line in {r"\[", "$$"}:
        return extract_last_boxed(first_line) or extract_last_boxed(final) or None
    return first_line.rstrip(".") or None


def _numeric(text: str) -> Fraction | None:
    try:
        if "," in text:
            if not re.fullmatch(r"[+-]?[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?", text):
                return None
            text = text.replace(",", "")
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None


def _simple_equal(a: str, b: str) -> bool:
    aa = a.strip().strip("$").strip()
    bb = b.strip().strip("$").strip()
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
