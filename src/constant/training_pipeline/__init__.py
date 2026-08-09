"""
All static constants used across the training pipeline: artifact
directory names, file names, MongoDB collection name, text/label
column names, etc.
"""
import os

TEXT_COLUMN: str = "Message"
LABEL_COLUMN: str = "Label"
TARGET_COLUMN: str = "label_num"  # 0 = ham, 1 = spam, after encoding

PIPELINE_NAME: str = "spam_detection"
ARTIFACT_DIR: str = "artifacts"
FILE_NAME: str = "spamham.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
MODEL_FILE_NAME: str = "model.pkl"
VECTORIZER_FILE_NAME: str = "vectorizer.pkl"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")
MODEL_CONFIG_FILE_PATH = os.path.join("config", "model.yaml")

# MongoDB related constants
DATABASE_NAME: str = "pws_projects"
COLLECTION_NAME: str = "spam_ham"
MONGODB_URL_KEY: str = "MONGODB_URL"

# Data Ingestion
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

# Data Validation
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"
MIN_MESSAGE_LENGTH: int = 1

# Data Transformation (text cleaning + TF-IDF)
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"
TFIDF_MAX_FEATURES: int = 5000

# Model Trainer
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = MODEL_FILE_NAME
MODEL_TRAINER_EXPECTED_SCORE: float = 0.90  # expected minimum F1 score

# Model Pusher / registry
MODEL_BUCKET_NAME = "spam-detection-model-registry"
MODEL_PUSHER_S3_KEY = "model-registry"

APP_HOST = "0.0.0.0"
APP_PORT = 8080
