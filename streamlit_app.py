from pathlib import Path
import pickle
from dataclasses import dataclass
from typing import Any
import os

import mlflow
from mlflow import artifacts as mlflow_artifacts
from mlflow import sklearn as mlflow_sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import streamlit as st

from src.components.data_transformation import DataTransformation
from src.config import MLflowConfig

MODELS_DIR = Path("models")
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"


@dataclass
class MlflowRunOption:
    run_id: str
    model_name: str
    test_accuracy: float | None
    start_time: int | None
    has_preprocessor: bool


def has_preprocessor_artifact(client: MlflowClient, run_id: str) -> bool:
    try:
        artifacts = client.list_artifacts(run_id=run_id, path="preprocessing")
        return any(artifact.path.endswith("preprocessor.pkl") for artifact in artifacts)
    except Exception:
        return False


def available_model_names() -> list[str]:
    model_files = []
    for model_path in MODELS_DIR.glob("*.pkl"):
        if model_path.name.endswith("_params.pkl") or model_path.name == "preprocessor.pkl":
            continue
        model_files.append(model_path.stem)
    return sorted(model_files)


@st.cache_resource
def load_artifacts(model_name: str):
    model_path = MODELS_DIR / f"{model_name}.pkl"

    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)

    with open(PREPROCESSOR_PATH, "rb") as preprocessor_file:
        preprocessor = pickle.load(preprocessor_file)

    return model, preprocessor


@st.cache_resource
def load_local_preprocessor():
    with open(PREPROCESSOR_PATH, "rb") as preprocessor_file:
        preprocessor = pickle.load(preprocessor_file)
    return preprocessor


def configure_mlflow():
    mlflow.set_tracking_uri(MLflowConfig().tracking_uri)


@st.cache_data(ttl=30)
def available_mlflow_runs() -> list[MlflowRunOption]:
    configure_mlflow()
    client = MlflowClient()
    exp = client.get_experiment_by_name(MLflowConfig().experiment_name)
    if exp is None:
        return []

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.status = 'FINISHED' and params.model_name != ''",
        order_by=["metrics.test_accuracy DESC", "attributes.start_time DESC"],
        max_results=200,
    )

    options: list[MlflowRunOption] = []
    for run in runs:
        model_name = run.data.params.get("model_name")
        if not model_name:
            continue
        test_accuracy = run.data.metrics.get("test_accuracy")
        options.append(
            MlflowRunOption(
                run_id=run.info.run_id,
                model_name=model_name,
                test_accuracy=float(test_accuracy) if test_accuracy is not None else None,
                start_time=run.info.start_time,
                has_preprocessor=has_preprocessor_artifact(client, run.info.run_id),
            )
        )

    return options


@st.cache_resource
def load_model_from_mlflow(run_id: str):
    configure_mlflow()
    return mlflow_sklearn.load_model(f"runs:/{run_id}/model")


@st.cache_resource
def load_preprocessor_from_mlflow(run_id: str):
    configure_mlflow()
    preprocessor_uri = f"runs:/{run_id}/preprocessing/preprocessor.pkl"
    local_path = mlflow_artifacts.download_artifacts(artifact_uri=preprocessor_uri)

    if os.path.isdir(local_path):
        local_path = os.path.join(local_path, "preprocessor.pkl")

    with open(local_path, "rb") as preprocessor_file:
        return pickle.load(preprocessor_file)


def build_input_frame() -> pd.DataFrame:
    st.subheader("Customer Inputs")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("gender", ["Female", "Male"])
        senior_citizen = st.selectbox("SeniorCitizen", [0, 1], index=0)
        partner = st.selectbox("Partner", ["No", "Yes"], index=0)
        dependents = st.selectbox("Dependents", ["No", "Yes"], index=0)
        tenure = st.number_input("tenure", min_value=0, max_value=100, value=12, step=1)
        phone_service = st.selectbox("PhoneService", ["No", "Yes"], index=1)
        multiple_lines = st.selectbox("MultipleLines", ["No", "Yes", "No phone service"], index=0)
        internet_service = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"], index=0)
        online_security = st.selectbox("OnlineSecurity", ["No", "Yes", "No internet service"], index=0)

    with col2:
        online_backup = st.selectbox("OnlineBackup", ["No", "Yes", "No internet service"], index=0)
        device_protection = st.selectbox("DeviceProtection", ["No", "Yes", "No internet service"], index=0)
        tech_support = st.selectbox("TechSupport", ["No", "Yes", "No internet service"], index=0)
        streaming_tv = st.selectbox("StreamingTV", ["No", "Yes", "No internet service"], index=0)
        streaming_movies = st.selectbox("StreamingMovies", ["No", "Yes", "No internet service"], index=0)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], index=0)
        paperless_billing = st.selectbox("PaperlessBilling", ["No", "Yes"], index=1)
        payment_method = st.selectbox(
            "PaymentMethod",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            index=0,
        )
        monthly_charges = st.number_input("MonthlyCharges", min_value=0.0, value=29.85, step=0.1, format="%.2f")
        total_charges = st.number_input("TotalCharges", min_value=0.0, value=358.20, step=0.1, format="%.2f")

    input_data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    return pd.DataFrame([input_data])


def preprocess_for_inference(input_df: pd.DataFrame, model, preprocessor) -> pd.DataFrame:
    transformer = DataTransformation()
    transformed_df = transformer.transform(input_df.copy())

    if transformed_df.empty:
        raise ValueError("Input row became empty after preprocessing. Please review the input values.")

    processed_df = preprocessor.transform(transformed_df)

    expected_cols = getattr(model, "feature_names_in_", None)
    if expected_cols is not None:
        processed_df = processed_df.reindex(columns=expected_cols, fill_value=0)

    return processed_df


def main():
    st.set_page_config(page_title="Churn Predictor", layout="wide")
    st.title("Customer Churn Prediction")
    st.caption("Run predictions from local .pkl files or from MLflow-tracked runs.")

    source = st.radio("Model Source", ["Local models folder", "MLflow runs"], horizontal=True)

    model: Any = None
    preprocessor: Any = None

    if source == "Local models folder":
        model_names = available_model_names()
        if not model_names:
            st.error("No trained model files were found in the models folder.")
            st.stop()

        if not PREPROCESSOR_PATH.exists():
            st.error("preprocessor.pkl is missing in the models folder.")
            st.stop()

        default_model = "random_forest" if "random_forest" in model_names else model_names[0]
        selected_model = st.selectbox("Model", model_names, index=model_names.index(default_model))
        model, preprocessor = load_artifacts(selected_model)
    else:
        run_options = available_mlflow_runs()
        if not run_options:
            st.error("No finished MLflow model runs were found.")
            st.stop()

        run_labels = []
        for opt in run_options:
            acc_part = f"acc={opt.test_accuracy:.4f}" if opt.test_accuracy is not None else "acc=n/a"
            prep_part = "prep=run" if opt.has_preprocessor else "prep=local-fallback"
            run_labels.append(f"{opt.model_name} | {acc_part} | {prep_part} | run_id={opt.run_id[:8]}")

        health_df = pd.DataFrame(
            [
                {
                    "model": opt.model_name,
                    "run_id": opt.run_id[:8],
                    "test_accuracy": opt.test_accuracy,
                    "preprocessor_source": "run artifact" if opt.has_preprocessor else "local fallback needed",
                }
                for opt in run_options
            ]
        )
        with st.expander("MLflow run artifact health", expanded=False):
            st.dataframe(health_df)

        selected_idx = 0
        selected_label = st.selectbox("MLflow Run", run_labels, index=selected_idx)
        selected_option = run_options[run_labels.index(selected_label)]

        model = load_model_from_mlflow(selected_option.run_id)
        try:
            preprocessor = load_preprocessor_from_mlflow(selected_option.run_id)
            st.info(
                f"Loaded model and preprocessor from MLflow run {selected_option.run_id} "
                f"({selected_option.model_name})."
            )
        except Exception:
            if not PREPROCESSOR_PATH.exists():
                st.error(
                    "This MLflow run does not include preprocessing/preprocessor.pkl and local "
                    "models/preprocessor.pkl is also missing. Retrain or provide a local preprocessor."
                )
                st.stop()
            preprocessor = load_local_preprocessor()
            st.warning(
                "Selected MLflow run does not contain preprocessor artifact; using "
                "local models/preprocessor.pkl fallback."
            )

    if model is None or preprocessor is None:
        st.error("Model artifacts could not be loaded.")
        st.stop()

    input_df = build_input_frame()

    with st.expander("Preview raw input", expanded=False):
        st.dataframe(input_df)

    predict_clicked = st.button("Predict Churn", type="primary")

    if predict_clicked:
        try:
            features_df = preprocess_for_inference(input_df, model, preprocessor)
            prediction = model.predict(features_df)[0]

            st.subheader("Prediction Result")
            predicted_label = "Yes" if int(prediction) == 1 else "No"
            st.success(f"Predicted churn: {predicted_label}")

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features_df)[0]
                if len(proba) > 1:
                    st.info(f"Churn probability: {proba[1]:.4f}")

            with st.expander("Model input after preprocessing", expanded=False):
                st.dataframe(features_df)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


if __name__ == "__main__":
    main()
