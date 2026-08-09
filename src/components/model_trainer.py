"""
Model Trainer component.

Loads model definitions + grid-search params from config/model.yaml
(same structure as the provided train_and_export.py script), trains
each candidate classifier on the TF-IDF features, picks the best by
test F1 score, and saves the winner bundled with the fitted
vectorizer via SpamDetectionModel.
"""
import importlib
import sys

import numpy as np
from scipy import sparse
from sklearn.model_selection import GridSearchCV

from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.entity.config_entity import ModelTrainerConfig
from src.exception import SpamDetectionException
from src.logger import logging
from src.ml.metric.classification_metric import get_classification_score
from src.ml.model.estimator import SpamDetectionModel
from src.utils.main_utils import load_object, read_yaml_file, save_object


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact

    @staticmethod
    def _load_data(transformed_path: str):
        x = sparse.load_npz(transformed_path)
        y = np.load(transformed_path + ".labels.npy")
        return x, y

    def train_and_select_best_model(self, x_train, y_train, x_test, y_test):
        config = read_yaml_file(self.model_trainer_config.model_config_file_path)
        model_configs = config["model_selection"]
        cv = config.get("grid_search", {}).get("params", {}).get("cv", 3)
        verbose = config.get("grid_search", {}).get("params", {}).get("verbose", 1)

        best_model = None
        best_model_name = None
        best_test_metric = None
        best_train_metric = None
        best_f1 = float("-inf")

        for module_key, model_def in model_configs.items():
            class_name = model_def["class"]
            module_name = model_def["module"]
            param_grid = model_def.get("search_param_grid", {})

            model_class = getattr(importlib.import_module(module_name), class_name)
            estimator = model_class()

            if param_grid:
                search = GridSearchCV(estimator, param_grid=param_grid, cv=cv, verbose=verbose, n_jobs=-1)
                search.fit(x_train, y_train)
                model = search.best_estimator_
            else:
                model = estimator.fit(x_train, y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_metric = get_classification_score(y_train, y_train_pred)
            test_metric = get_classification_score(y_test, y_test_pred)

            logging.info(
                f"{class_name} -> train F1={train_metric.f1_score:.4f}, "
                f"test F1={test_metric.f1_score:.4f}, test accuracy={test_metric.accuracy:.4f}"
            )

            if test_metric.f1_score > best_f1:
                best_f1 = test_metric.f1_score
                best_model = model
                best_model_name = class_name
                best_test_metric = test_metric
                best_train_metric = train_metric

        logging.info(f"Best model selected: {best_model_name} (test F1={best_f1:.4f})")
        return best_model, best_train_metric, best_test_metric

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            x_train, y_train = self._load_data(self.data_transformation_artifact.transformed_train_file_path)
            x_test, y_test = self._load_data(self.data_transformation_artifact.transformed_test_file_path)

            model, train_metric, test_metric = self.train_and_select_best_model(
                x_train, y_train, x_test, y_test
            )

            if test_metric.f1_score < self.model_trainer_config.expected_score:
                logging.warning(
                    f"Best model F1={test_metric.f1_score:.4f} is below expected "
                    f"score {self.model_trainer_config.expected_score}. Proceeding anyway."
                )

            vectorizer = load_object(self.data_transformation_artifact.vectorizer_object_file_path)
            spam_detection_model = SpamDetectionModel(vectorizer=vectorizer, model=model)
            save_object(self.model_trainer_config.trained_model_file_path, spam_detection_model)

            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metric,
                test_metric_artifact=test_metric,
            )
        except Exception as e:
            raise SpamDetectionException(e, sys) from e
