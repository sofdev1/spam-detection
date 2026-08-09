"""
Prediction Pipeline.

Loads the trained SpamDetectionModel from saved_models/ and exposes a
single predict_message() function that the Flask /predict endpoint
calls with the raw text a user typed into the form.
"""
import sys

from src.entity.config_entity import ModelPusherConfig
from src.exception import SpamDetectionException
from src.logger import logging
from src.utils.main_utils import load_object

_model_cache = None

LABEL_MAP = {0: "ham", 1: "spam"}


def _load_model():
    global _model_cache
    if _model_cache is None:
        model_path = ModelPusherConfig().model_file_path
        logging.info(f"Loading model from {model_path}")
        _model_cache = load_object(model_path)
    return _model_cache


def predict_message(message: str) -> dict:
    """
    Returns a dict: {"label": "spam"|"ham", "confidence": float|None}
    """
    try:
        model = _load_model()
        prediction = model.predict([message])[0]
        label = LABEL_MAP.get(int(prediction), str(prediction))

        confidence = None
        proba = model.predict_proba([message])
        if proba is not None:
            confidence = round(float(max(proba[0])), 4)

        return {"label": label, "confidence": confidence}
    except Exception as e:
        raise SpamDetectionException(e, sys) from e
