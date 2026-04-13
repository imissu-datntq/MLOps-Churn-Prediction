"""FastAPI prediction service for ChurnGuard.

Start with:
    uvicorn app:app --host 0.0.0.0 --port 8000

Or via Docker Compose:
    docker-compose up --build
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import load_artifacts, predict, predict_batch

# ---------------------------------------------------------------------------
# Lifespan: load artifacts once at startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="ChurnGuard API",
    description=(
        "Production-ready REST API for customer churn prediction. "
        "Send a customer feature record and receive a churn probability."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class CustomerFeatures(BaseModel):
    tenure: float = Field(..., ge=0, description="Number of months as a customer.")
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charge amount.")
    TotalCharges: float = Field(..., ge=0, description="Total charges to date.")
    gender: str = Field("Male", description="Male or Female.")
    SeniorCitizen: str = Field("0", description="1 if senior citizen, else 0.")
    Partner: str = Field("No", description="Yes or No.")
    Dependents: str = Field("No", description="Yes or No.")
    PhoneService: str = Field("Yes", description="Yes or No.")
    MultipleLines: str = Field("No", description="Yes, No, or No phone service.")
    InternetService: str = Field("Fiber optic", description="DSL, Fiber optic, or No.")
    OnlineSecurity: str = Field("No", description="Yes, No, or No internet service.")
    OnlineBackup: str = Field("No", description="Yes, No, or No internet service.")
    DeviceProtection: str = Field("No", description="Yes, No, or No internet service.")
    TechSupport: str = Field("No", description="Yes, No, or No internet service.")
    StreamingTV: str = Field("No", description="Yes, No, or No internet service.")
    StreamingMovies: str = Field("No", description="Yes, No, or No internet service.")
    Contract: str = Field("Month-to-month", description="Month-to-month, One year, or Two year.")
    PaperlessBilling: str = Field("Yes", description="Yes or No.")
    PaymentMethod: str = Field(
        "Electronic check",
        description="Electronic check, Mailed check, Bank transfer, or Credit card.",
    )

    model_config = {"json_schema_extra": {"example": {
        "tenure": 24,
        "MonthlyCharges": 65.5,
        "TotalCharges": 1572.0,
        "gender": "Male",
        "SeniorCitizen": "0",
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
    }}}


class PredictionResponse(BaseModel):
    churn: bool = Field(..., description="True if the customer is predicted to churn.")
    churn_probability: float = Field(..., description="Probability of churn (0–1).")


class BatchRequest(BaseModel):
    records: list[CustomerFeatures] = Field(..., min_length=1)


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", summary="Health check")
def health() -> dict:
    """Return service health status."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse, summary="Single prediction")
def predict_endpoint(customer: CustomerFeatures) -> PredictionResponse:
    """Predict churn for a single customer record."""
    try:
        result = predict(customer.model_dump())
        return PredictionResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict/batch", response_model=BatchResponse, summary="Batch prediction")
def predict_batch_endpoint(batch: BatchRequest) -> BatchResponse:
    """Predict churn for a batch of customer records."""
    try:
        records = [r.model_dump() for r in batch.records]
        results = predict_batch(records)
        return BatchResponse(predictions=[PredictionResponse(**r) for r in results])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
