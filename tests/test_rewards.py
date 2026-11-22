from src.rewards import extract_last_boxed, extract_answer, is_correct


def test_nested_latex_boxed():
    assert extract_last_boxed(r"we find \boxed{\frac{1}{2}}") == r"\frac{1}{2}"


def test_final_answer_and_no_random_number_matching():
    assert extract_answer("2 + 2 = 4\nAnswer: 4") == "4"
    assert extract_answer("I tried 4, maybe 5") is None
    assert is_correct("Answer: 1/2", "0.5")
    assert not is_correct("Answer: 3", "4")
