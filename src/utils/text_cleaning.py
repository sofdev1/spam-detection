"""
Text preprocessing utilities shared by both the training pipeline
(data_transformation.py) and the prediction pipeline, so a message
typed into the web form is cleaned in exactly the same way as the
training data.

Steps (matching the EDA notebook's approach): lowercase -> strip
non-word characters -> collapse whitespace -> remove stopwords ->
lemmatize.
"""
import re
import sys

from src.exception import SpamDetectionException
from src.logger import logging

_NLTK_READY = False


def _ensure_nltk_data():
    """Downloads required NLTK corpora on first use if not already present."""
    global _NLTK_READY
    if _NLTK_READY:
        return
    import nltk
    for pkg in ["stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(
                f"corpora/{pkg}" if pkg != "punkt" else "tokenizers/punkt"
            )
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception as e:  # pragma: no cover - offline environments
                logging.warning(f"Could not download NLTK package '{pkg}': {e}")
    _NLTK_READY = True


def clean_text(message: str) -> str:
    """
    Clean and lemmatize a raw message string. Falls back to simple
    lowercase + whitespace cleanup (no stopword removal/lemmatization)
    if NLTK corpora aren't available (e.g. fully offline environment),
    so the pipeline never hard-fails on preprocessing.
    """
    try:
        if not isinstance(message, str):
            message = "" if message is None else str(message)

        message = message.lower()
        message = re.sub(r"\W+", " ", message)
        message = re.sub(r"\s+", " ", message).strip()

        try:
            _ensure_nltk_data()
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer

            stop_words = set(stopwords.words("english"))
            lemmatizer = WordNetLemmatizer()
            tokens = [
                lemmatizer.lemmatize(word)
                for word in message.split()
                if word not in stop_words
            ]
            return " ".join(tokens)
        except Exception as e:
            logging.warning(f"Falling back to basic cleaning (NLTK unavailable): {e}")
            return message
    except Exception as e:
        raise SpamDetectionException(e, sys) from e
