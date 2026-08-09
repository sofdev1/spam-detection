"""
Data Validation component.

Checks the ingested train/test CSVs against the expected schema
(Label + Message columns present, no fully-empty messages, labels
only "ham"/"spam"), and runs a simple class-balance drift check
between train and test splits, writing a YAML drift report.
"""
import os
import sys

import pandas as pd

from src.constant.training_pipeline import LABEL_COLUMN, MIN_MESSAGE_LENGTH, TEXT_COLUMN
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.exception import SpamDetectionException
from src.logger import logging
from src.utils.main_utils import write_yaml_file

EXPECTED_COLUMNS = [LABEL_COLUMN, TEXT_COLUMN]
VALID_LABELS = {"ham", "spam"}


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_config = data_validation_config

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)

    def is_column_exist(self, dataframe: pd.DataFrame) -> bool:
        return all(col in dataframe.columns for col in EXPECTED_COLUMNS)

    def has_valid_labels(self, dataframe: pd.DataFrame) -> bool:
        return set(dataframe[LABEL_COLUMN].str.lower().unique()).issubset(VALID_LABELS)

    def has_no_empty_messages(self, dataframe: pd.DataFrame) -> bool:
        return (dataframe[TEXT_COLUMN].astype(str).str.strip().str.len() >= MIN_MESSAGE_LENGTH).all()

    def check_class_balance(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
        train_ratio = train_df[LABEL_COLUMN].value_counts(normalize=True).to_dict()
        test_ratio = test_df[LABEL_COLUMN].value_counts(normalize=True).to_dict()
        return {"train_class_ratio": train_ratio, "test_class_ratio": test_ratio}

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            errors = []
            for name, df in [("train", train_df), ("test", test_df)]:
                if not self.is_column_exist(df):
                    errors.append(f"{name} dataframe is missing expected columns")
                elif not self.has_valid_labels(df):
                    errors.append(f"{name} dataframe contains unexpected label values")
                elif not self.has_no_empty_messages(df):
                    errors.append(f"{name} dataframe contains empty messages")

            status = len(errors) == 0
            os.makedirs(os.path.dirname(self.data_validation_config.drift_report_file_path), exist_ok=True)

            if status:
                report = self.check_class_balance(train_df, test_df)
                write_yaml_file(self.data_validation_config.drift_report_file_path, report)

                os.makedirs(os.path.dirname(self.data_validation_config.valid_train_file_path), exist_ok=True)
                train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False)
                test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False)
                valid_train_path = self.data_validation_config.valid_train_file_path
                valid_test_path = self.data_validation_config.valid_test_file_path
                invalid_train_path = None
                invalid_test_path = None
            else:
                logging.warning(f"Data validation failed: {errors}")
                write_yaml_file(self.data_validation_config.drift_report_file_path, {"errors": errors})
                os.makedirs(os.path.dirname(self.data_validation_config.invalid_train_file_path), exist_ok=True)
                train_df.to_csv(self.data_validation_config.invalid_train_file_path, index=False)
                test_df.to_csv(self.data_validation_config.invalid_test_file_path, index=False)
                valid_train_path = None
                valid_test_path = None
                invalid_train_path = self.data_validation_config.invalid_train_file_path
                invalid_test_path = self.data_validation_config.invalid_test_file_path

            return DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=valid_train_path,
                valid_test_file_path=valid_test_path,
                invalid_train_file_path=invalid_train_path,
                invalid_test_file_path=invalid_test_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
        except Exception as e:
            raise SpamDetectionException(e, sys) from e
