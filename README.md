# ChurnGuard: An End-to-End MLOps Pipeline for Customer Churn Prediction

## 📌 Overview

This repository contains a production-ready MLOps pipeline designed to predict customer churn. Unlike traditional machine learning projects that focus solely on model accuracy, **ChurnGuard** emphasizes the entire ML lifecycle—including data versioning, automated experiment tracking, continuous integration, and containerized deployment.

The goal is to provide a robust framework that allows data scientists to move from raw data to a monitored, scalable API with minimal manual intervention.

---

## 🛠 Tech Stack

| Layer | Tool |
|---|---|
| **Language** | Python 3.9+ |
| **Orchestration** | [DVC](https://dvc.org/) (Data Version Control) |
| **Experiment Tracking** | [MLflow](https://mlflow.org/) |
| **Containerization** | Docker & Docker Compose |
| **CI/CD** | GitHub Actions |
| **Model Serving** | [FastAPI](https://fastapi.tiangolo.com/) |
| **Testing** | Pytest (Unit & Integration tests) |
| **Data Validation** | [Evidently AI](https://www.evidentlyai.com/) |

---

## 🏗 Project Architecture

The project follows a modular pipeline structure to ensure maintainability and scalability.

```
Raw Data ──► Ingestion ──► Preprocessing ──► Training ──► Evaluation ──► Registry ──► API
               (DVC)         (scikit-learn)    (MLflow)    (Evidently)    (MLflow)   (FastAPI)
```

1. **Data Ingestion** — Automated scripts fetch and version raw datasets using DVC.
2. **Preprocessing & Feature Engineering** — Modular scripts handle missing values, encoding, and scaling; the same logic is applied at inference time.
3. **Model Training** — Training scripts integrate with MLflow to log hyperparameters, metrics, and model artifacts.
4. **Evaluation & Validation** — Automated checks for model drift and performance before promotion to production.
5. **Model Registry** — Centralized versioned model storage via MLflow.
6. **Deployment** — REST API built with FastAPI, containerized with Docker for environment consistency.

---

## 🚀 Key Features

### 1. Data & Model Versioning
**DVC** manages large datasets and model files that are too heavy for Git. Every experiment is 100% reproducible by linking specific code versions to specific data states.

### 2. Automated Experiment Tracking
Every training run is logged via **MLflow**. Compare different algorithms (Random Forest, XGBoost, LightGBM) and hyperparameter configurations through the MLflow UI at `http://localhost:5000`.

### 3. Continuous Integration (CI)
GitHub Actions runs automatically on every push and pull request:
- Lint code with `flake8` and `black`.
- Run unit tests on data processing functions.
- Upload coverage reports as artifacts.

### 4. Containerized Deployment
The prediction service is wrapped in a **Docker** container, enabling seamless execution across Development, Staging, and Production environments.

### 5. Data Drift Monitoring
**Evidently AI** generates data profile reports to detect feature drift, ensuring the model stays accurate as customer behaviour evolves.

---

## 📁 Repository Structure

```text
├── .github/workflows/
│   └── ci.yml                # CI/CD pipeline (lint + test)
├── data/
│   ├── raw/                  # DVC-tracked raw CSV files
│   └── processed/            # DVC-tracked processed features
├── models/                   # Saved model & preprocessor artifacts
├── reports/                  # Metrics and drift reports
├── src/
│   ├── __init__.py
│   ├── ingestion.py          # Data loading logic
│   ├── preprocessing.py      # Feature engineering & preprocessing
│   ├── train.py              # Model training & MLflow logging
│   └── predict.py            # Inference logic
├── tests/
│   ├── test_ingestion.py
│   ├── test_preprocessing.py
│   └── test_train.py
├── app.py                    # FastAPI application
├── Dockerfile                # Container configuration
├── docker-compose.yml        # Multi-service orchestration (API + MLflow)
├── dvc.yaml                  # DVC pipeline definition
├── params.yaml               # Hyperparameters managed by DVC
└── requirements.txt          # Project dependencies
```

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.9+
- Docker & Docker Compose
- DVC (`pip install dvc`)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/YN2TB/MLOps-Churn-Prediction.git
   cd MLOps-Churn-Prediction
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Pull versioned data** (requires DVC remote access)

   ```bash
   dvc pull
   ```

### Running the Pipeline

Execute the full MLOps pipeline end-to-end:

```bash
dvc repro
```

DVC automatically detects which stages are stale and only reruns what has changed.

### Starting the API

Run the full stack (prediction API + MLflow tracking server):

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| Prediction API | `http://localhost:8000/docs` |
| MLflow UI | `http://localhost:5000` |

**Interactive API docs** are available at `http://localhost:8000/docs` (Swagger UI).

### Running Tests

```bash
pytest tests/ -v --cov=src
```

---

## 🔍 API Usage

### Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 24,
    "MonthlyCharges": 65.5,
    "TotalCharges": 1572.0,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "PaymentMethod": "Electronic check"
  }'
```

**Response:**

```json
{
  "churn": true,
  "churn_probability": 0.7831
}
```

### Health Check

```bash
curl http://localhost:8000/health
```

---

## 🧪 Experiment Tracking

Training runs are tracked automatically. To compare experiments:

1. Start the MLflow UI: `mlflow ui` (or via Docker Compose)
2. Open `http://localhost:5000`
3. Navigate to the **churn-prediction** experiment to compare runs

To change the model or hyperparameters, edit `params.yaml` and re-run `dvc repro`.

---

## 📊 Results & Monitoring

The current champion model achieves (example values; update after training):

| Metric | Score |
|---|---|
| F1-Score | 0.XX |
| Precision | 0.XX |
| Recall | 0.XX |
| ROC-AUC | 0.XX |
| Inference Latency | < 50 ms |

**Evidently AI** generates data profile reports stored in `reports/` to monitor for feature drift, ensuring the model remains accurate as customer behaviour evolves over time.

---

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-feature`.
3. Make your changes and run tests: `pytest tests/ -v`.
4. Ensure code quality: `black . && flake8 src/ app.py tests/`.
5. Submit a pull request.

---

## 📄 License

This project is licensed under the MIT License.