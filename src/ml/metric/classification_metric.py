"""
Computes the classification metrics (accuracy, precision, recall, F1)
used to evaluate the model trainer's candidate models and pick the
best one.
"""
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.entity.artifact_entity import ClassificationMetricArtifact
from src.exception import SpamDetectionException


def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    try:
        return ClassificationMetricArtifact(
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            f1_score=float(f1_score(y_true, y_pred, zero_division=0)),
        )
    except Exception as e:
        raise SpamDetectionException(e, sys) from e
