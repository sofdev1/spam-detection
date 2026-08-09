"""
Data Ingestion component.

Pulls raw records from MongoDB (falling back to the bundled
data/spamham.csv — the ham+spam combination of sms.csv and emails.csv
produced in the EDA notebook — when no live Mongo instance is
configured), writes to the feature store, and performs a train/test
split.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.constant.training_pipeline import LABEL_COLUMN, TEXT_COLUMN
from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig
from src.exception import SpamDetectionException
from src.logger import logging

BUNDLED_DATA_PATH = os.path.join("data", "spamham.csv")


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

    def export_collection_as_dataframe(self) -> pd.DataFrame:
        """
        Attempt to read from MongoDB. If MONGODB_URL isn't configured
        or the read fails, fall back to the bundled data/spamham.csv
        so the pipeline can still run fully offline.
        """
        try:
            from src.configuration.mongo_db_connection import MongoDBClient

            mongo_client = MongoDBClient()
            collection = mongo_client.database[self.data_ingestion_config.collection_name]
            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns:
                df = df.drop(columns=["_id"], axis=1)
            df.replace({"na": np.nan}, inplace=True)
            if df.empty:
                raise ValueError("Empty collection")
            logging.info("Loaded data from MongoDB collection")
            return df
        except Exception as e:
            logging.warning(f"Falling back to bundled dataset (reason: {e})")
            return self._load_bundled_dataset()

    @staticmethod
    def _load_bundled_dataset() -> pd.DataFrame:
        df = pd.read_csv(BUNDLED_DATA_PATH, encoding="latin-1")
        df = df[[LABEL_COLUMN, TEXT_COLUMN]].dropna()
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def export_data_into_feature_store(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(os.path.dirname(feature_store_file_path), exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe
        except Exception as e:
            raise SpamDetectionException(e, sys) from e

    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> None:
        try:
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42,
                stratify=dataframe[LABEL_COLUMN],
            )
            os.makedirs(os.path.dirname(self.data_ingestion_config.training_file_path), exist_ok=True)
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            logging.info("Exported train and test file paths")
        except Exception as e:
            raise SpamDetectionException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)
            return DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )
        except Exception as e:
            raise SpamDetectionException(e, sys) from e
