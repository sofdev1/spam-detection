"""
Data Transformation component.

Cleans message text (lowercase, strip non-word chars, remove
stopwords, lemmatize — via src/utils/text_cleaning.py), encodes
labels (ham=0, spam=1), fits a TF-IDF vectorizer on the training
split, transforms both splits, and persists the fitted vectorizer +
transformed sparse matrices for the model trainer stage.
"""
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from src.constant.training_pipeline import LABEL_COLUMN, TEXT_COLUMN, TFIDF_MAX_FEATURES
from src.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from src.entity.config_entity import DataTransformationConfig
from src.exception import SpamDetectionException
from src.logger import logging
from src.utils.main_utils import save_object
from src.utils.text_cleaning import clean_text


class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        self.data_validation_artifact = data_validation_artifact
        self.data_transformation_config = data_transformation_config

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)

            logging.info("Cleaning message text (lowercase, stopwords, lemmatization)")
            train_df["clean_message"] = train_df[TEXT_COLUMN].apply(clean_text)
            test_df["clean_message"] = test_df[TEXT_COLUMN].apply(clean_text)

            label_encoder = LabelEncoder()
            # Fit on both splits' labels to guarantee ham/spam -> 0/1 consistently
            label_encoder.fit(["ham", "spam"])
            y_train = label_encoder.transform(train_df[LABEL_COLUMN].str.lower())
            y_test = label_encoder.transform(test_df[LABEL_COLUMN].str.lower())

            vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
            x_train = vectorizer.fit_transform(train_df["clean_message"])
            x_test = vectorizer.transform(test_df["clean_message"])

            save_object(self.data_transformation_config.vectorizer_object_file_path, vectorizer)

            import os
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_file_path), exist_ok=True)
            sparse.save_npz(self.data_transformation_config.transformed_train_file_path, x_train)
            sparse.save_npz(self.data_transformation_config.transformed_test_file_path, x_test)
            np.save(self.data_transformation_config.transformed_train_file_path + ".labels.npy", y_train)
            np.save(self.data_transformation_config.transformed_test_file_path + ".labels.npy", y_test)

            logging.info("Data transformation completed")

            return DataTransformationArtifact(
                vectorizer_object_file_path=self.data_transformation_config.vectorizer_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
        except Exception as e:
            raise SpamDetectionException(e, sys) from e
