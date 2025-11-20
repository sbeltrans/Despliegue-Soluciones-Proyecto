import typing as t
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from sp500_model import __version__ as _version
from sp500_model.config.core import TRAINED_MODEL_DIR, config


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    # Guardamos el pipeline con versionamiento y eliminamos los antiguos
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(*, file_name: str) -> Pipeline:
    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    # Mantenemos solo la versión actual para evitar confusión
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
