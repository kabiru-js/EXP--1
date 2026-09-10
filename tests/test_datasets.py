"""Tests for dataset loading and answer extraction."""

import pytest
from src.datasets.gsm8k import GSM8KLoader, _extract_gsm8k_answer, _normalize_number
from src.datasets.synthetic import SyntheticArithmeticLoader


class TestGSM8KAnswerExtraction:
    def test_standard_format(self):
        assert _extract_gsm8k_answer("The answer is 42 #### 42") == "42"

    def test_with_commas(self):
        assert _extract_gsm8k_answer("Total is 1,234 #### 1,234") == "1,234"

    def test_negative(self):
        assert _extract_gsm8k_answer("Result: -5 #### -5") == "-5"

    def test_decimal(self):
        assert _extract_gsm8k_answer("Answer: 3.14 #### 3.14") == "3.14"

    def test_no_format(self):
        result = _extract_gsm8k_answer("The answer is 42")
        assert result == "42"


class TestNumberNormalization:
    def test_simple(self):
        assert _normalize_number("42") == 42.0

    def test_comma_separated(self):
        assert _normalize_number("1,234") == 1234.0

    def test_dollar(self):
        assert _normalize_number("$50.00") == 50.0

    def test_percent(self):
        assert _normalize_number("50%") == 50.0

    def test_invalid(self):
        assert _normalize_number("abc") is None


class TestGSM8KCorrectness:
    def setup_method(self):
        self.loader = GSM8KLoader()

    def test_exact_match(self):
        assert self.loader.check_correctness("42", "42")

    def test_numeric_close(self):
        assert self.loader.check_correctness("42.0", "42")

    def test_wrong(self):
        assert not self.loader.check_correctness("43", "42")

    def test_string_match(self):
        assert self.loader.check_correctness("abc", "abc")


class TestSyntheticLoader:
    def test_load(self):
        loader = SyntheticArithmeticLoader()
        samples = loader.load(num_samples=10)
        assert len(samples) == 10
        for s in samples:
            assert s.dataset == "synthetic_arithmetic"
            assert s.question.startswith("What is")
            assert s.ground_truth is not None

    def test_answer_extraction(self):
        loader = SyntheticArithmeticLoader()
        assert loader.extract_answer("The answer is 5") == "5"
        assert loader.extract_answer("Result: -3.14") == "-3.14"

    def test_correctness(self):
        loader = SyntheticArithmeticLoader()
        assert loader.check_correctness("42", "42")
        assert loader.check_correctness("42.0", "42.0")
        assert not loader.check_correctness("43", "42")


class TestDatasetRegistry:
    def test_available(self):
        from src.datasets.registry import available_datasets
        ds = available_datasets()
        assert "gsm8k" in ds
        assert "synthetic_arithmetic" in ds

    def test_unknown_dataset(self):
        from src.datasets.registry import get_loader
        with pytest.raises(ValueError):
            get_loader("nonexistent_dataset")
