"""
SpamDetectionModel bundles the fitted TF-IDF vectorizer and the fitted
classifier into a single object, so the prediction pipeline only
needs to load and call ONE artifact.
"""
import sys

from src.exception import SpamDetectionException
from src.utils.text_cleaning import clean_text


class SpamDetectionModel:
    def __init__(self, vectorizer, model):
        self.vectorizer = vectorizer
        self.model = model

    def predict(self, messages):
        """
        messages: list[str] of raw (uncleaned) messages.
        Returns: list[int] predictions (0 = ham, 1 = spam).
        """
        try:
            cleaned = [clean_text(m) for m in messages]
            features = self.vectorizer.transform(cleaned)
            return self.model.predict(features)
        except Exception as e:
            raise SpamDetectionException(e, sys) from e

    def predict_proba(self, messages):
        try:
            cleaned = [clean_text(m) for m in messages]
            features = self.vectorizer.transform(cleaned)
            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(features)
            return None
        except Exception as e:
            raise SpamDetectionException(e, sys) from e

    def __repr__(self):
        return f"{type(self.model).__name__}()"

    def __str__(self):
        return f"{type(self.model).__name__}()"
