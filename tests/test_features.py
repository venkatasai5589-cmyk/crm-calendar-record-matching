import pytest

from matching.features import confidence, email_local_part, overlap, tokens

WEIGHTS = {"a": 2.0, "b": 1.0, "c": 1.0}


def test_email_local_part_handles_apostrophes():
    assert email_local_part("Kevin O'Brien") == "kevin.obrien"


def test_missing_signal_does_not_lower_confidence():
    assert confidence({"a": 1.0, "b": 1.0, "c": None}, WEIGHTS) == 1.0


def test_confidence_is_weighted_average_of_available_signals():
    assert confidence({"a": 1.0, "b": 0.0, "c": None}, WEIGHTS) == pytest.approx(2 / 3)


def test_zero_weight_signal_is_ignored():
    assert confidence({"a": 1.0, "b": 0.0, "c": None}, {**WEIGHTS, "b": 0.0}) == 1.0


def test_no_evidence_gives_zero_confidence():
    assert confidence({"a": None, "b": None, "c": None}, WEIGHTS) == 0.0


def test_overlap_ignores_stopwords_and_punctuation():
    # "discussion" is a stopword, so this compares {esg, compliance} with {esg, compliance, review}.
    assert overlap(tokens("ESG Compliance Discussion"), tokens("ESG + Compliance Review")) == 1.0
    assert overlap(tokens("Annual Allocation Review"), tokens("Year-End Review")) == pytest.approx(1 / 3)
    assert overlap(tokens(None), tokens("Anything")) is None
