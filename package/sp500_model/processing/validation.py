from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from sp500_model.config.core import config


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for na values and filter."""
    validated_data = input_data.copy()
    new_vars_with_na = [
        var
        for var in config.ml_config.features
        if validated_data[var].isnull().sum() > 0
    ]
    validated_data.dropna(subset=new_vars_with_na, inplace=True)

    return validated_data


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    relevant_data = input_data[config.ml_config.features].copy()
    validated_data = drop_na_inputs(input_data=relevant_data)
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class DataInputSchema(BaseModel):
    """Schema para validar inputs del modelo S&P 500."""

    SMA_20: Optional[float]
    SMA_50: Optional[float]
    EMA_12: Optional[float]
    EMA_26: Optional[float]
    RSI_14: Optional[float]
    MACD: Optional[float]
    MACD_signal: Optional[float]
    MACD_diff: Optional[float]
    BB_upper: Optional[float]
    BB_middle: Optional[float]
    BB_lower: Optional[float]
    BB_width: Optional[float]
    ATR_14: Optional[float]
    OBV: Optional[float]
    Returns: Optional[float]
    Volatility_10: Optional[float]
    Volume_change: Optional[float]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]
