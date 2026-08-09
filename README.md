# Spam Detection

An end-to-end MLOps project that classifies SMS/email messages as
**spam** or **ham** (legitimate), built from the provided datasets and
notebooks (`sms.csv`, `emails.csv`, `EDA_final.ipynb`,
`Model Training.ipynb`, `upload_data_mongodb.py`, `train_and_export.py`).

## Project Overview

- **Objective:** classify incoming SMS/email messages as spam or ham.
- **Data:** `data/sms.csv` (SMS messages) and `data/emails.csv`
  (email messages) are combined and deduplicated into
  `data/spamham.csv` — the same combination step performed in
  `notebooks/EDA_final.ipynb`.
- **Approach:** clean text (lowercase, strip punctuation, remove
  stopwords, lemmatize) → TF-IDF vectorize → train and compare
  multiple classifiers (Naive Bayes, Logistic Regression, Random
  Forest, Linear SVM) → pick the best by F1 score → serve predictions
  through a Flask web app.

## Tech Stack

| Layer            | Technology |
|-------------------|------------|
| Language            | Python 3.10 |
| NLP                    | NLTK (stopwords, lemmatization) |
| ML                       | scikit-learn (TF-IDF, classifiers, GridSearchCV) |
| Web                        | Flask |
| Database                     | MongoDB (optional data source) |
| Containerization                | Docker |
| CI/CD                              | GitHub Actions |

## Project Structure

```
spam-detection/
├── app.py                        # Flask web app (message classifier UI)
├── main.py                       # Training pipeline entrypoint
├── setup.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .env.example
├── .github/workflows/main.yaml   # CI -> Build/Push -> Deploy
├── config/
│   ├── model.yaml                 # candidate models + grid search params
│   └── schema.yaml
├── data/
│   ├── sms.csv                      # provided SMS dataset
│   ├── emails.csv                    # provided email dataset
│   └── spamham.csv                    # combined dataset (from EDA notebook)
├── scripts/
│   └── upload_data_mongodb.py          # provided: pushes spamham.csv to MongoDB
├── src/
│   ├── components/                      # data_ingestion, data_validation,
│   │                                     # data_transformation, model_trainer,
│   │                                     # model_pusher
│   ├── configuration/                     # mongo_db_connection
│   ├── constant/training_pipeline/         # all pipeline constants
│   ├── entity/                              # config_entity, artifact_entity
│   ├── exception/                            # custom exception class
│   ├── logger/                                # logging setup
│   ├── ml/
│   │   ├── model/estimator.py                  # SpamDetectionModel wrapper
│   │   └── metric/classification_metric.py       # accuracy/precision/recall/F1
│   ├── pipeline/
│   │   ├── training_pipeline.py                    # orchestrates all components
│   │   └── prediction_pipeline.py                    # loads model, serves /predict
│   └── utils/
│       ├── main_utils.py                               # yaml/pickle/numpy IO helpers
│       └── text_cleaning.py                              # shared text preprocessing
├── templates/                                              # index.html, result.html
├── static/style.css
├── notebooks/
│   ├── EDA_final.ipynb                                        # provided EDA notebook
│   └── Model Training.ipynb                                    # provided training notebook
├── tests/                                                        # pytest smoke tests
├── artifacts/                                                     # per-run pipeline outputs (gitignored)
└── saved_models/                                                   # "model registry" the app loads from
```

## Flow of the Project

1. **Data Ingestion** — loads `data/spamham.csv` (or pulls from
   MongoDB if `MONGODB_URL` is configured — see
   `scripts/upload_data_mongodb.py` for how the collection is
   populated), then splits into train/test.
2. **Data Validation** — checks expected columns, valid label values
   (`ham`/`spam`), no empty messages, and reports class balance.
3. **Data Transformation** — cleans text (lowercase → strip
   punctuation → remove stopwords → lemmatize, via
   `src/utils/text_cleaning.py`), fits a TF-IDF vectorizer on the
   training split, transforms both splits.
4. **Model Training & Evaluation** — trains Naive Bayes, Logistic
   Regression, Random Forest, and Linear SVM (config-driven via
   `config/model.yaml`, mirroring the provided `train_and_export.py`
   structure), picks the best by F1 score.
5. **Model Pusher** — copies the winning model into `saved_models/`
   (optionally syncs to S3 if `MODEL_REGISTRY_BUCKET` is set).
6. **Flask Deployment** — `app.py` serves a simple web form; paste a
   message, get a spam/ham prediction with confidence.

## Setup & Installation

### 1. Clone and create environment

```bash
git clone <your-repo-url> spam-detection
cd spam-detection
conda create -n spam-detection python=3.10 -y
conda activate spam-detection
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 2. Configure environment variables (optional)

```bash
cp .env.example .env
# fill in MONGODB_URL if you want to pull from MongoDB instead of the bundled CSV
```

If `MONGODB_URL` isn't set, the pipeline automatically falls back to
the bundled `data/spamham.csv`, so you can run everything end-to-end
without any live database.

### 3. Run the training pipeline

```bash
python main.py
```

This runs Data Ingestion → Data Validation → Data Transformation →
Model Trainer → Model Pusher, and saves the final model to
`saved_models/model.pkl`.

### 4. Run the web app

```bash
python app.py
```

Navigate to **http://localhost:8080**, paste a message, and click
**Check Message**. You can also trigger training from the browser via
`GET /train`.

### 5. Run with Docker

```bash
docker build -t spam-detection .
docker run -p 8080:8080 --env-file .env spam-detection
```

## Uploading data to MongoDB (optional)

The provided `scripts/upload_data_mongodb.py` pushes `data/spamham.csv`
into a `pws_projects.spam_ham` MongoDB collection:

```bash
cp .env.example .env   # set MONGODB_URL
python scripts/upload_data_mongodb.py
```

## Running Tests

```bash
pytest tests/ -q
```

## Notes on the Provided Notebooks/Scripts

- `notebooks/EDA_final.ipynb` — the original exploratory analysis:
  loads `sms.csv` and `emails.csv`, deduplicates, combines them into
  `spamham.csv`, and explores word frequency / class balance. This
  logic is what produced the `data/spamham.csv` bundled here.
- `notebooks/Model Training.ipynb` — original modeling exploration.
- `scripts/upload_data_mongodb.py` — unmodified from what was
  provided; pushes `spamham.csv` into MongoDB.
- `docker_cheatsheet.pdf` — kept at the project root as a reference
  (not part of the runnable pipeline).

The `src/` package reimplements this same flow as a structured,
testable, config-driven MLOps pipeline rather than notebook cells, so
it can run repeatably via `main.py` and be served via `app.py`.

## License

Provided for educational purposes
See `LICENSE`
