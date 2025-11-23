from typing import Any, List, Optional

from pydantic import BaseModel
from sp500_model.processing.validation import DataInputSchema

# Esquema de los resultados de predicción
class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[int]]
    probabilities: Optional[List[float]]

# Esquema para inputs múltiples
class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "SMA_20": 150.25,
                        "SMA_50": 148.80,
                        "EMA_12": 151.10,
                        "EMA_26": 149.50,
                        "RSI_14": 65.5,
                        "MACD": 2.30,
                        "MACD_signal": 1.80,
                        "MACD_diff": 0.50,
                        "BB_upper": 155.00,
                        "BB_middle": 150.00,
                        "BB_lower": 145.00,
                        "BB_width": 10.00,
                        "ATR_14": 3.50,
                        "OBV": 1000000.0,
                        "Returns": 0.015,
                        "Volatility_10": 0.025,
                        "Volume_change": 0.10
                    }
                ]
            }
        }
