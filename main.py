"""
Entry point for running the full end-to-end training pipeline:
Data Ingestion -> Data Validation -> Data Transformation ->
Model Training -> Model Pusher.

Usage:
    python main.py
"""
from src.pipeline.training_pipeline import TrainingPipeline

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    artifact = pipeline.run_pipeline()
    print(f"Training pipeline finished. Model registered at: {artifact.model_file_path}")
