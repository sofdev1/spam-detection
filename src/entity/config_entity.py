"""
Config entities: dataclasses that define *where things live* for each
pipeline stage, all rooted under a single timestamped artifact
directory per training run.
"""
import os
from dataclasses import dataclass
from datetime import datetime

from src.constant import training_pipeline as tp


@dataclass
class TrainingPipelineConfig:
    artifact_dir: str = os.path.join(
        tp.ARTIFACT_DIR, datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    )
    timestamp: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        base = os.path.join(training_pipeline_config.artifact_dir, tp.DATA_INGESTION_DIR_NAME)
        self.feature_store_file_path = os.path.join(
            base, tp.DATA_INGESTION_FEATURE_STORE_DIR, tp.FILE_NAME
        )
        self.training_file_path = os.path.join(
            base, tp.DATA_INGESTION_INGESTED_DIR, tp.TRAIN_FILE_NAME
        )
        self.testing_file_path = os.path.join(
            base, tp.DATA_INGESTION_INGESTED_DIR, tp.TEST_FILE_NAME
        )
        self.train_test_split_ratio = tp.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        self.collection_name = tp.COLLECTION_NAME


class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        base = os.path.join(training_pipeline_config.artifact_dir, tp.DATA_VALIDATION_DIR_NAME)
        self.valid_train_file_path = os.path.join(base, tp.DATA_VALIDATION_VALID_DIR, tp.TRAIN_FILE_NAME)
        self.valid_test_file_path = os.path.join(base, tp.DATA_VALIDATION_VALID_DIR, tp.TEST_FILE_NAME)
        self.invalid_train_file_path = os.path.join(base, tp.DATA_VALIDATION_INVALID_DIR, tp.TRAIN_FILE_NAME)
        self.invalid_test_file_path = os.path.join(base, tp.DATA_VALIDATION_INVALID_DIR, tp.TEST_FILE_NAME)
        self.drift_report_file_path = os.path.join(
            base, tp.DATA_VALIDATION_DRIFT_REPORT_DIR, tp.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME
        )


class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        base = os.path.join(training_pipeline_config.artifact_dir, tp.DATA_TRANSFORMATION_DIR_NAME)
        self.transformed_train_file_path = os.path.join(
            base, tp.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, "train.npz"
        )
        self.transformed_test_file_path = os.path.join(
            base, tp.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, "test.npz"
        )
        self.vectorizer_object_file_path = os.path.join(
            base, tp.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR, tp.VECTORIZER_FILE_NAME
        )


class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        base = os.path.join(training_pipeline_config.artifact_dir, tp.MODEL_TRAINER_DIR_NAME)
        self.trained_model_file_path = os.path.join(
            base, tp.MODEL_TRAINER_TRAINED_MODEL_DIR, tp.MODEL_TRAINER_TRAINED_MODEL_NAME
        )
        self.expected_score = tp.MODEL_TRAINER_EXPECTED_SCORE
        self.model_config_file_path = tp.MODEL_CONFIG_FILE_PATH


class ModelPusherConfig:
    def __init__(self):
        self.saved_model_dir = "saved_models"
        self.model_file_path = os.path.join(self.saved_model_dir, tp.MODEL_FILE_NAME)
        self.vectorizer_file_path = os.path.join(self.saved_model_dir, tp.VECTORIZER_FILE_NAME)
