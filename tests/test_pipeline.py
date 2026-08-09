"""
Smoke tests for the Spam Detection pipeline. Run with:
    pytest tests/ -q
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.text_cleaning import clean_text
from src.ml.metric.classification_metric import get_classification_score


def test_clean_text_lowercases_and_strips_punctuation():
    result = clean_text("FREE!!! Win a PRIZE now!!!")
    assert result == result.lower()
    assert "!" not in result


def test_clean_text_handles_non_string_input():
    assert clean_text(None) == ""
    assert isinstance(clean_text(12345), str)


def test_classification_metric_perfect_predictions():
    metric = get_classification_score([0, 1, 1, 0], [0, 1, 1, 0])
    assert metric.accuracy == 1.0
    assert metric.f1_score == 1.0


def test_bundled_dataset_loads():
    from src.components.data_ingestion import DataIngestion
    df = DataIngestion._load_bundled_dataset()
    assert len(df) > 0
    assert set(["Label", "Message"]).issubset(df.columns)
    assert set(df["Label"].str.lower().unique()).issubset({"ham", "spam"})
