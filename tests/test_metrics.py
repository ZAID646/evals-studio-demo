import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import normalize, accuracy_score, f1_score, rouge_l_score


class TestNormalize:
    def test_lowercase(self):
        assert normalize("Hello") == "hello"

    def test_strip_whitespace(self):
        assert normalize("  hello  ") == "hello"

    def test_remove_punctuation(self):
        assert normalize("hello!") == "hello"
        assert normalize("hello, world") == "hello, world"

    def test_empty_string(self):
        assert normalize("") == ""

    def test_only_punctuation(self):
        assert normalize(".,!?") == ""


class TestAccuracy:
    def test_exact_match(self):
        assert accuracy_score(["hello"], ["hello"]) == 1.0

    def test_no_match(self):
        assert accuracy_score(["hello"], ["world"]) == 0.0

    def test_partial_match(self):
        score = accuracy_score(["hello", "world"], ["hello", "there"])
        assert score == 0.5

    def test_case_insensitive(self):
        assert accuracy_score(["Hello"], ["hello"]) == 1.0

    def test_punctuation_insensitive(self):
        assert accuracy_score(["hello!"], ["hello"]) == 1.0

    def test_empty(self):
        assert accuracy_score([], []) == 0.0


class TestF1:
    def test_exact_match(self):
        assert f1_score(["hello world"], ["hello world"]) == 1.0

    def test_no_overlap(self):
        assert f1_score(["hello"], ["world"]) == 0.0

    def test_partial(self):
        score = f1_score(["hello world"], ["hello there"])
        assert 0 < score < 1.0

    def test_empty_prediction(self):
        assert f1_score([""], ["hello"]) == 0.0


class TestRougeL:
    def test_exact_match(self):
        assert rouge_l_score(["hello world"], ["hello world"]) == 1.0

    def test_no_match(self):
        assert rouge_l_score(["hello"], ["world"]) == 0.0

    def test_partial(self):
        score = rouge_l_score(["hello world"], ["hello there"])
        assert 0 < score < 1.0
