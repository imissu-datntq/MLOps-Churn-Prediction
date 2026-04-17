import pytest
from src.config import DataTransformationConfig, ModelTrainerConfig, MLflowConfig

def test_config_initialization():
    """Test if configuration objects instantiate without errors."""
    dt_config = DataTransformationConfig()
    mt_config = ModelTrainerConfig()
    mlflow_config = MLflowConfig()
    
    assert dt_config is not None
    assert mt_config is not None
    assert mlflow_config is not None
    assert mlflow_config.experiment_name == "churn-prediction"

def test_preprocessing_columns():
    """Ensure categorical and continuous columns are defined properly."""
    dt_config = DataTransformationConfig()
    assert isinstance(dt_config.categorical_columns, list)
    assert isinstance(dt_config.continuous_columns, list)
    assert len(dt_config.categorical_columns) > 0
