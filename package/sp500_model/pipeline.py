from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from sp500_model.config.core import config

# Nuestro pipeline: StandardScaler -> PCA -> XGBoost
sp500_pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "pca",
            PCA(
                n_components=config.ml_config.pca_components,
                random_state=config.ml_config.random_state,
            ),
        ),
        (
            "xgboost",
            XGBClassifier(
                n_estimators=config.ml_config.n_estimators,
                max_depth=config.ml_config.max_depth,
                learning_rate=config.ml_config.learning_rate,
                subsample=config.ml_config.subsample,
                colsample_bytree=config.ml_config.colsample_bytree,
                min_child_weight=config.ml_config.min_child_weight,
                gamma=config.ml_config.gamma,
                random_state=config.ml_config.random_state,
                n_jobs=-1,
                eval_metric="logloss",
            ),
        ),
    ]
)
