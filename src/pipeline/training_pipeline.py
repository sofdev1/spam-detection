"""
Training Pipeline orchestrator: wires all components together in
order — Data Ingestion -> Data Validation -> Data Transformation ->
Model Trainer -> Model Pusher.
"""
import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.components.model_pusher import ModelPusher
from src.components.model_trainer import ModelTrainer
from src.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelPusherConfig,
    ModelTrainerConfig,
    TrainingPipelineConfig,
)
from src.exception import SpamDetectionException
from src.logger import logging


class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()

    def start_data_ingestion(self):
        config = DataIngestionConfig(self.training_pipeline_config)
        logging.info("Starting data ingestion")
        return DataIngestion(config).initiate_data_ingestion()

    def start_data_validation(self, data_ingestion_artifact):
        config = DataValidationConfig(self.training_pipeline_config)
        logging.info("Starting data validation")
        return DataValidation(data_ingestion_artifact, config).initiate_data_validation()

    def start_data_transformation(self, data_validation_artifact):
        config = DataTransformationConfig(self.training_pipeline_config)
        logging.info("Starting data transformation")
        return DataTransformation(data_validation_artifact, config).initiate_data_transformation()

    def start_model_trainer(self, data_transformation_artifact):
        config = ModelTrainerConfig(self.training_pipeline_config)
        logging.info("Starting model training")
        return ModelTrainer(config, data_transformation_artifact).initiate_model_trainer()

    def start_model_pusher(self, model_trainer_artifact):
        config = ModelPusherConfig()
        logging.info("Starting model pusher")
        return ModelPusher(config, model_trainer_artifact).initiate_model_pusher()

    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)

            if not data_validation_artifact.validation_status:
                raise Exception("Data validation failed — aborting pipeline")

            data_transformation_artifact = self.start_data_transformation(data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)
            model_pusher_artifact = self.start_model_pusher(model_trainer_artifact)

            logging.info(f"Training pipeline completed. Metrics: {model_trainer_artifact.test_metric_artifact}")
            return model_pusher_artifact
        except Exception as e:
            raise SpamDetectionException(e, sys) from e
