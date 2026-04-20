import os
import sys
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import mlflow
from mlflow import sklearn
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.config import DataIngestionConfig, models, param_grids, ModelTrainerConfig, MLflowConfig, MLFLOW_DB_PATH, MLFLOW_ARTIFACTS_PATH
from src.components.preprocessor import Preprocessor


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.mlflow_config = MLflowConfig()
        self.data_ingestion_config = DataIngestionConfig()

    def _prepare_target(self, y):
        # ...existing code...
        # Convert DataFrame with one column -> Series
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError("Target y must have exactly one column for classification.")
            y = y.iloc[:, 0]

        # Convert to 1D array
        y = np.ravel(y)

        # Normalize string labels if needed
        if y.dtype.kind in {"O", "U", "S"}:
            y_series = pd.Series(y).astype(str).str.strip().str.lower()
            binary_map = {
                "yes": 1, "no": 0,
                "true": 1, "false": 0,
                "1": 1, "0": 0
            }
            if y_series.isin(binary_map.keys()).all():
                y = y_series.map(binary_map).to_numpy(dtype=int)
            else:
                y = pd.factorize(y_series)[0]

        return y

    def train_model(self, model, model_name, param_grid, X_train, y_train, X_test, y_test):
        logging.info(f"Training {model_name}")

        y_train = self._prepare_target(y_train)
        y_test = self._prepare_target(y_test)

        if os.path.exists(os.path.join(self.model_trainer_config.trained_model_file_path, f"{model_name}_params.pkl")):
            with open(os.path.join(self.model_trainer_config.trained_model_file_path, f"{model_name}_params.pkl"), "rb") as f:
                best_params = pickle.load(f)
                for key, value in best_params.items():
                    best_params[key] = [value]
                    
                if model_name == "logistic_regression":
                    best_params.pop("penalty", None)
                    best_params.pop("l1_ratio", None)
        else:
            logging.warning(f"No best parameters found for {model_name}, using default parameter grid.")
            best_params = param_grid

        grid_search = GridSearchCV(
            model,
            best_params,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            error_score='raise'
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        logging.info(f"Best {model_name} model: {best_model}")
        logging.info(f"Best {model_name} parameters: {best_params}")
        logging.info(f"Best {model_name} score: {best_score}")

        save_object(
            file_path=os.path.join(self.model_trainer_config.trained_model_file_path, f"{model_name}.pkl"),
            obj=best_model
        )
        save_object(
            file_path=os.path.join(self.model_trainer_config.trained_model_file_path, f"{model_name}_params.pkl"),
            obj=best_params
        )

        y_pred = best_model.predict(X_test)
        
        # Calculate evaluation metrics
        test_accuracy = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        test_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        test_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Calculate ROC AUC if probability predictions are supported
        test_roc_auc = None
        try:
            if hasattr(best_model, "predict_proba"):
                y_pred_proba = best_model.predict_proba(X_test)
                if y_pred_proba.shape[1] == 2:
                    y_pred_proba = y_pred_proba[:, 1]
                test_roc_auc = roc_auc_score(y_test, y_pred_proba)
        except Exception as e:
            logging.warning(f"Could not calculate ROC AUC for {model_name}: {e}")

        logging.info(f"{model_name} test accuracy: {test_accuracy}")
        logging.info(f"{model_name} metrics - F1: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")

        mlflow.log_params({f"best_{key}": value for key, value in best_params.items()})
        mlflow.log_metric("cv_best_score", float(best_score))
        mlflow.log_metric("test_accuracy", float(test_accuracy))
        mlflow.log_metric("test_f1_score", float(test_f1))
        mlflow.log_metric("test_precision", float(test_precision))
        mlflow.log_metric("test_recall", float(test_recall))
        if test_roc_auc is not None:
            mlflow.log_metric("test_roc_auc", float(test_roc_auc))
        mlflow.log_param("model_name", model_name)
        sklearn.log_model(best_model, name="model")

        return best_model, best_params, best_score, test_accuracy, test_f1, test_precision, test_recall, test_roc_auc

    def _initialize_mlflow_tracking(self):
        """Initialize MLflow and recover from incompatible Alembic revision DB errors."""
        mlflow.set_tracking_uri(self.mlflow_config.tracking_uri)

        def _ensure_experiment_exists():
            client = MlflowClient()
            experiment = client.get_experiment_by_name(self.mlflow_config.experiment_name)
            if experiment is None:
                mlflow.create_experiment(
                    name=self.mlflow_config.experiment_name,
                    artifact_location=self.mlflow_config.artifact_location,
                )
            mlflow.set_experiment(self.mlflow_config.experiment_name)

        try:
            _ensure_experiment_exists()
        except Exception as exc:
            message = str(exc)
            if "Can't locate revision identified by" not in message:
                raise

            if MLFLOW_DB_PATH.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = MLFLOW_DB_PATH.with_name(f"mlflow_incompatible_{timestamp}.db")
                try:
                    os.replace(MLFLOW_DB_PATH, backup_path)
                    logging.warning(
                        "Detected incompatible MLflow DB revision. Backed up old DB to %s and creating a new tracking DB.",
                        backup_path,
                    )
                except PermissionError:
                    fallback_path = MLFLOW_DB_PATH.with_name(f"mlflow_fallback_{timestamp}.db")
                    self.mlflow_config.tracking_uri = f"sqlite:///{fallback_path.as_posix()}"
                    logging.warning(
                        "Detected incompatible MLflow DB revision but %s is locked by another process. "
                        "Switching this training run to fallback tracking DB: %s",
                        MLFLOW_DB_PATH,
                        fallback_path,
                    )

            mlflow.set_tracking_uri(self.mlflow_config.tracking_uri)
            _ensure_experiment_exists()

    def initiate_model_training(self, X_train, y_train, X_test, y_test):
        try:
            os.makedirs(self.model_trainer_config.trained_model_file_path, exist_ok=True)
            os.makedirs(MLFLOW_DB_PATH.parent, exist_ok=True)
            os.makedirs(MLFLOW_ARTIFACTS_PATH, exist_ok=True)

            self._initialize_mlflow_tracking()

            # Remove target leakage if Churn is still in features
            if isinstance(X_train, pd.DataFrame):
                for col in ["Churn", "churn"]:
                    if col in X_train.columns:
                        X_train = X_train.drop(columns=[col])
                    if isinstance(X_test, pd.DataFrame) and col in X_test.columns:
                        X_test = X_test.drop(columns=[col])

            preprocessor = Preprocessor()
            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

            with mlflow.start_run(run_name=self.mlflow_config.run_name_prefix):
                mlflow.log_param("test_size", self.data_ingestion_config.test_size)
                mlflow.log_param("train_rows", int(len(X_train)))
                mlflow.log_param("test_rows", int(len(X_test)))
                mlflow.log_param("feature_count", int(X_train.shape[1]))

                save_object(
                    file_path=os.path.join(self.model_trainer_config.trained_model_file_path, "preprocessor.pkl"),
                    obj=preprocessor
                )
                mlflow.log_artifact(os.path.join(self.model_trainer_config.trained_model_file_path, "preprocessor.pkl"))

                model_results = {}
                for model_name, model in models.items():
                    try:
                        param_grid = param_grids.get(model_name, {})
                        with mlflow.start_run(run_name=model_name, nested=True):
                            mlflow.log_artifact(
                                os.path.join(self.model_trainer_config.trained_model_file_path, "preprocessor.pkl"),
                                artifact_path="preprocessing"
                            )
                            best_model, best_params, best_score, test_accuracy, test_f1, test_precision, test_recall, test_roc_auc = self.train_model(
                                model, model_name, param_grid, X_train, y_train, X_test, y_test
                            )
                            model_results[model_name] = {
                                'best_model': best_model,
                                'best_params': best_params,
                                'best_score': best_score,
                                'test_accuracy': test_accuracy,
                                'test_f1': test_f1,
                                'test_precision': test_precision,
                                'test_recall': test_recall,
                                'test_roc_auc': test_roc_auc
                            }
                    except Exception as e:
                        self.handle_training_exception(e, model_name)

                return model_results
        except Exception as e:
            self.handle_training_exception(e, "initiate_model_training")
    
    def handle_training_exception(self, exception, model_name="Unknown"):
        """Handle and log training exceptions with custom error details."""
        error_message = f"Error occurred during training of {model_name}: {str(exception)}"
        raise CustomException(error_message, sys)