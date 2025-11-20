import typing as t

import pandas as pd

from sp500_model import __version__ as _version
from sp500_model.config.core import config
from sp500_model.processing.data_manager import load_pipeline
from sp500_model.processing.validation import validate_inputs

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
_sp500_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(
    *,
    input_data: t.Union[pd.DataFrame, dict],
) -> dict:
    """Make a prediction using a saved model pipeline."""

    data = pd.DataFrame(input_data)
    validated_data, errors = validate_inputs(input_data=data)
    results = {"predictions": None, "version": _version, "errors": errors}

    if not errors:
        predictions = _sp500_pipe.predict(
            X=validated_data[config.ml_config.features]
        )
        # También retornamos probabilidades
        probabilities = _sp500_pipe.predict_proba(
            X=validated_data[config.ml_config.features]
        )[:, 1]

        results = {
            "predictions": [int(pred) for pred in predictions],
            "probabilities": [float(prob) for prob in probabilities],
            "version": _version,
            "errors": errors,
        }

    return results
