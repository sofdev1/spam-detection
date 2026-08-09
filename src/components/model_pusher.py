"""
Model Pusher component.

Copies the freshly trained SpamDetectionModel from the run's artifact
directory into saved_models/ (the "model registry" the Flask
prediction pipeline always loads from). Optionally syncs to S3 if
AWS credentials + MODEL_REGISTRY_BUCKET are configured.
"""
import os
import shutil
import sys

from src.entity.artifact_entity import ModelPusherArtifact, ModelTrainerArtifact
from src.entity.config_entity import ModelPusherConfig
from src.exception import SpamDetectionException
from src.logger import logging


class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig,
                 model_trainer_artifact: ModelTrainerArtifact):
        self.model_pusher_config = model_pusher_config
        self.model_trainer_artifact = model_trainer_artifact

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            os.makedirs(self.model_pusher_config.saved_model_dir, exist_ok=True)
            shutil.copy(
                self.model_trainer_artifact.trained_model_file_path,
                self.model_pusher_config.model_file_path,
            )
            logging.info(f"Model pushed to registry at {self.model_pusher_config.model_file_path}")

            bucket = os.getenv("MODEL_REGISTRY_BUCKET")
            if bucket:
                try:
                    import boto3
                    s3 = boto3.client("s3")
                    s3.upload_file(
                        self.model_pusher_config.model_file_path, bucket,
                        f"model-registry/{os.path.basename(self.model_pusher_config.model_file_path)}",
                    )
                    logging.info(f"Uploaded model to s3://{bucket}/model-registry/")
                except Exception as e:
                    logging.warning(f"Skipped S3 upload (reason: {e})")

            return ModelPusherArtifact(
                saved_model_path=self.model_pusher_config.saved_model_dir,
                model_file_path=self.model_pusher_config.model_file_path,
            )
        except Exception as e:
            raise SpamDetectionException(e, sys) from e
