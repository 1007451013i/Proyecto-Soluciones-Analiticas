from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from model.config.core import config

heart_pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("logistic_regression", LogisticRegression(max_iter=1000, solver="liblinear", random_state=config.model_config.random_state)),
    ]
)
