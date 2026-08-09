"""
Artifact entities: dataclasses returned *by* each pipeline stage,
carrying forward exactly what the next stage needs.
"""
from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    trained_file_path: str
    test_file_path: str


@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str


@dataclass
class DataTransformationArtifact:
    vectorizer_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str


@dataclass
class ClassificationMetricArtifact:
    accuracy: float
    precision: float
    recall: float
    f1_score: float


@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    train_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact


@dataclass
class ModelPusherArtifact:
    saved_model_path: str
    model_file_path: str
