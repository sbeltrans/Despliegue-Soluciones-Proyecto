# Usamos este script para entrenar y guardar nuestro modelo final

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sp500_model.config.core import config
from sp500_model.pipeline import sp500_pipe
from sp500_model.processing.data_manager import save_pipeline


def run_training():
    # Cargamos nuestros datos
    project_root = Path(__file__).resolve().parent.parent
    train_path = project_root / "data" / "processed" / "ml_ready" / "train.parquet"
    test_path = project_root / "data" / "processed" / "ml_ready" / "test.parquet"

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    X_train = train[config.ml_config.features]
    y_train = train[config.ml_config.target]
    X_test = test[config.ml_config.features]
    y_test = test[config.ml_config.target]

    # Entrenamos nuestro pipeline
    sp500_pipe.fit(X_train, y_train)

    # Evaluamos en test set
    y_pred = sp500_pipe.predict(X_test)
    y_pred_proba = sp500_pipe.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["Sell (0)", "Buy (1)"]))

    # Guardamos nuestro modelo entrenado
    save_pipeline(pipeline_to_persist=sp500_pipe)


if __name__ == "__main__":
    run_training()
