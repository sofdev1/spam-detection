"""
app.py
------
Flask web interface for the Spam Detection project.

Run with:
    python app.py
Then open:
    http://localhost:8080

Endpoints:
    GET  /          - form to paste a message and classify it
    POST /predict    - classifies the submitted message
    GET  /train       - runs the full training pipeline
    GET  /health       - simple healthcheck
"""
import os
import sys

from flask import Flask, render_template, request, flash, redirect, url_for

from src.constant.training_pipeline import APP_HOST, APP_PORT
from src.exception import SpamDetectionException
from src.logger import logging
from src.pipeline.prediction_pipeline import predict_message
from src.pipeline.training_pipeline import TrainingPipeline

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-in-production")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    message = request.form.get("message", "").strip()

    if not message:
        flash("Please enter a message to classify.")
        return redirect(url_for("index"))

    try:
        result = predict_message(message)
    except FileNotFoundError:
        flash("No trained model found yet. Please run the training pipeline first "
              "(visit /train, or run `python main.py`).")
        return redirect(url_for("index"))
    except Exception as exc:
        logging.exception("Unexpected prediction error")
        flash(f"Something went wrong while predicting: {exc}")
        return redirect(url_for("index"))

    return render_template("result.html", message=message, result=result)


@app.route("/train")
def train_route():
    """Kicks off the full training pipeline (data ingestion -> model pusher)."""
    try:
        artifact = TrainingPipeline().run_pipeline()
        return {
            "status": "success",
            "message": "Training pipeline completed successfully.",
            "model_path": artifact.model_file_path,
        }
    except Exception as e:
        raise SpamDetectionException(e, sys) from e


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(debug=True, host=APP_HOST, port=APP_PORT)
