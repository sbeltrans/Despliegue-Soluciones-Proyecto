"""
Script para entrenar el modelo final XGBoost + PCA-15
y guardarlo en el paquete sp500_model.
"""

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

# Agregar el paquete al path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sp500_model.config.core import config
from sp500_model.pipeline import sp500_pipe
from sp500_model.processing.data_manager import save_pipeline


def run_training() -> None:
    """Entrena el modelo final y lo guarda."""

    # Rutas a los datos (relativos al proyecto)
    project_root = Path(__file__).resolve().parent.parent
    train_path = project_root / "data" / "processed" / "ml_ready" / "train.parquet"
    test_path = project_root / "data" / "processed" / "ml_ready" / "test.parquet"

    print("=" * 70)
    print("ENTRENAMIENTO DEL MODELO FINAL - S&P 500")
    print("=" * 70)

    # Cargar datos
    print(f"\nCargando datos de entrenamiento desde: {train_path}")
    train = pd.read_parquet(train_path)

    print(f"Cargando datos de prueba desde: {test_path}")
    test = pd.read_parquet(test_path)

    # Separar features y target
    X_train = train[config.ml_config.features]
    y_train = train[config.ml_config.target]
    X_test = test[config.ml_config.features]
    y_test = test[config.ml_config.target]

    print(f"\nDatos cargados:")
    print(f"  Train: {X_train.shape}")
    print(f"  Test: {X_test.shape}")
    print(f"  Features: {len(config.ml_config.features)}")

    # Entrenar pipeline
    print("\n" + "-" * 70)
    print("Entrenando pipeline: StandardScaler → PCA-15 → XGBoost")
    print("-" * 70)

    sp500_pipe.fit(X_train, y_train)

    print("Modelo entrenado exitosamente!")

    # Evaluar en test set
    print("\n" + "-" * 70)
    print("Evaluación en Test Set")
    print("-" * 70)

    y_pred = sp500_pipe.predict(X_test)
    y_pred_proba = sp500_pipe.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"\nMétricas:")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    print("\nMatriz de Confusión:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"\nTrue Negatives (TN):  {cm[0,0]}")
    print(f"False Positives (FP): {cm[0,1]}")
    print(f"False Negatives (FN): {cm[1,0]}")
    print(f"True Positives (TP):  {cm[1,1]}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test, y_pred, target_names=["Vender (0)", "Comprar (1)"]
        )
    )

    # Guardar pipeline
    print("\n" + "-" * 70)
    print("Guardando modelo entrenado...")
    print("-" * 70)

    save_pipeline(pipeline_to_persist=sp500_pipe)

    print("\n" + "=" * 70)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 70)
    print(f"\nModelo final: XGBoost + PCA-15")
    print(f"Versión: {config.app_config.package_name}")
    print(f"ROC-AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    run_training()
