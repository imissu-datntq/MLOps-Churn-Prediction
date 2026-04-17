# End-to-End MLOps Pipeline for Customer Churn Prediction

## Overview

This repository contains a churn model training pipeline with MLflow experiment tracking and Docker support. The focus is on training multiple classifiers, logging metrics and artifacts, and keeping the results easy to compare across runs.

The goal is to provide a simple, reproducible workflow that takes the prepared churn dataset, trains models, and records the results in MLflow.

---

## Tech Stack

| Layer | Tool |
|---|---|
| **Language** | Python 3.9+ |
| **Experiment Tracking** | [MLflow](https://mlflow.org/) |
| **Containerization** | Docker, Docker Compose & Kubernetes |
| **Modeling** | scikit-learn, XGBoost |
| **CI/CD & Testing** | GitHub Actions, Pytest |
| **UI** | Streamlit |

---

## Project Architecture

The project follows a compact training pipeline structure.

```
Raw Data --> Ingestion --> Preprocessing --> Training --> MLflow Tracking
            (pandas)      (scikit-learn)    (MLflow)
```

1. **Data Ingestion** - Reads the raw churn CSV from `data/raw/data.csv` and creates train/test splits.
2. **Preprocessing & Feature Engineering** - Handles encoding, scaling, and target preparation for the churn dataset.
3. **Model Training** - Trains multiple classifiers and logs hyperparameters, metrics, and artifacts to MLflow.
4. **Experiment Tracking** - Compare runs in the MLflow UI without changing the training code.

---

## Key Features

### 1. Data & Model Versioning
Model artifacts are saved under `models/`, and MLflow keeps a run history in `mlflow/mlflow.db`.

### 2. Automated Experiment Tracking
Every training run is logged via MLflow. Compare different algorithms and hyperparameter configurations through the MLflow UI at `http://localhost:5000`.

### 3. Containerized Execution & MLOps Orchestration
The training job and MLflow UI can be run locally with Docker Compose, or scaled dynamically using Kubernetes for Enterprise-grade High Availability and Rolling Updates. CI/CD is enforced via GitHub Actions and Pytest.

### 4. Streamlit Prediction Interface
Interactive web UI for making predictions using either local models or MLflow-tracked runs.

---

## Repository Structure

```text
├── .github/workflows/        # Automated CI/CD pipelines (GitHub Actions)
├── data/
│   ├── raw/                  # Raw churn CSV files
│   └── processed/            # Train/test splits
├── kubernetes/               # Enterprise K8s Manifests (Deployments, Services, PVC)
├── mlflow/                   # Local MLflow tracking store and artifacts
├── models/                   # Saved model & preprocessor artifacts
├── src/
│   ├── __init__.py
│   ├── config.py             # Paths, model definitions, MLflow config
│   ├── exception.py          # Custom exception handling
│   ├── logger.py             # Logging configuration
│   ├── utils.py              # Utility functions
│   └── components/
│       ├── data_ingestion.py
│       ├── data_transformation.py
│       ├── model_trainer.py
│       └── preprocessor.py
├── tests/                    # Unit tests for CI pipeline (Pytest)
├── main.py                   # Entry point for training and logging
├── streamlit_app.py          # Streamlit prediction interface
├── Dockerfile                # Container configuration
├── compose.yaml              # MLflow UI + training orchestration
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- Docker & Docker Compose
- Kubernetes (optional, for enterprise deployment)

### Installation

1. Clone the repository

   ```bash
   git clone https://github.com/YN2TB/MLOps-Churn-Prediction.git
   cd MLOps-Churn-Prediction
   ```

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

### Running the Training Job

Run the training pipeline locally:

```bash
python main.py
```

Or run the training job and MLflow UI together:

```bash
docker compose up --build
```

Then open:

- MLflow UI: `http://localhost:5000`
- Streamlit app: `http://localhost:8501`

### Inspecting MLflow

| Service | URL |
|---|---|
| MLflow UI | `http://localhost:5000` |

The training job writes MLflow metadata to `./mlflow/mlflow.db` and artifacts to `./mlflow/artifacts`.

### Streamlit Prediction With MLflow-Tracked Models

The Streamlit app supports two inference sources:

- Local models folder: loads `models/<model_name>.pkl` and `models/preprocessor.pkl`
- MLflow runs: loads both model and preprocessor from the selected run

For older runs that were logged before this change, the app automatically falls back to `models/preprocessor.pkl` if the run does not contain `preprocessing/preprocessor.pkl`.

Run the UI:

```bash
streamlit run streamlit_app.py
```

In the app, choose Model Source and then pick either a local model or an MLflow run.

### Running Tests & CI/CD Pipeline

Run the test suite locally:

```bash
pytest tests/
```

Tests are automatically executed on every push via GitHub Actions workflows.

### Enterprise Kubernetes Deployment

For a highly available production setup with load balancing, deploy using Kubernetes:

1. Build Local Docker Image:

   ```bash
   docker build -t mlops-churn-prediction:latest .
   ```

2. Apply Kubernetes Manifests:

   ```bash
   kubectl apply -f kubernetes/
   ```

3. Access the Microservices:

   - Streamlit Inference Web UI: `http://localhost:8501` (Load-balanced, Auto-scaled)
   - Internal MLflow Tracking: Port-forward with `kubectl port-forward svc/mlflow-service 5000:5000`, then open `http://localhost:5000`

To clean up:

```bash
kubectl delete -f kubernetes/
```

---

## Experiment Tracking

Training runs are tracked automatically. To compare experiments:

1. Start the MLflow UI: `mlflow ui --backend-store-uri sqlite:///./mlflow/mlflow.db --default-artifact-root ./mlflow/artifacts`
2. Open `http://localhost:5000`
3. Navigate to the **churn-prediction** experiment to compare runs

If you encounter a migration error, run:

```bash
mlflow db upgrade sqlite:///./mlflow/mlflow.db
```

To change the model or hyperparameters, edit the values in `src/config.py` and rerun `python main.py`.

---

## Results & Monitoring

Model performance from the latest recorded run:

| Model | CV Best Score | Test Accuracy | Test F1 | Test Precision | Test Recall | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8047 | 0.8051 | 0.7998 | 0.7975 | 0.8051 | 0.8410 |
| Decision Tree | 0.7830 | 0.7916 | 0.7802 | 0.7788 | 0.7916 | 0.8269 |
| Random Forest | 0.7983 | 0.7973 | 0.7888 | 0.7867 | 0.7973 | 0.8409 |
| XGBoost | 0.8047 | 0.7959 | 0.7873 | 0.7851 | 0.7959 | 0.8441 |
| SVM (RBF) | 0.8038 | 0.7959 | 0.7847 | 0.7837 | 0.7959 | N/A |