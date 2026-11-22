from src.rewards import extract_last_boxed, extract_answer, is_correct


def test_nested_latex_boxed():
    assert extract_last_boxed(r"we find \boxed{\frac{1}{2}}") == r"\frac{1}{2}"


def test_final_answer_and_no_random_number_matching():
    assert extract_answer("2 + 2 = 4\nAnswer: 4") == "4"
    assert extract_answer("I tried 4, maybe 5") is None
    assert is_correct("Answer: 1/2", "0.5")
    assert not is_correct("Answer: 3", "4")


def test_final_label_overrides_earlier_box():
    completion = "Answer: \\boxed{4}\nFinal Answer: 5"
    assert is_correct(completion, "5")
    assert not is_correct(completion, "4")
    assert extract_answer("Earlier: \\boxed{4}\nAnswer:") is None
    assert extract_answer("Answer:\n" + r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert extract_answer(r"Therefore \boxed{3}.") == "3"


def test_commas_only_group_thousands_in_numeric_answers():
    assert not is_correct("Answer: 1,2", "12")
    assert not is_correct("Answer: 12", "1,2")
    assert is_correct("Answer: 1,234", "1234")
    assert is_correct("Answer: -1,234,567.50", "-1234567.5")
    assert is_correct("Answer: (1,2)", "(1,2)")
