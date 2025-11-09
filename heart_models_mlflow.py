# heart_models_mlflow.py

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

# -----------------------------
# Cargar datos
# -----------------------------
url = "https://raw.github.com/1007451013i/Proyecto-Soluciones-Analiticas/main/heart.csv"
df = pd.read_csv(url)

target_col = "condition"
X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -----------------------------
# Función para calcular métricas
# -----------------------------
def calcular_metricas(y_true, y_pred, y_proba=None):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan
    }


# -----------------------------
# Experimentos con MLflow
# -----------------------------
# Definir el servidor MLflow (cámbialo si usas otra dirección)
mlflow.set_tracking_uri("http://localhost:5000")

# Registrar el experimento
experiment = mlflow.set_experiment("Heart Disease Prediction")

# -----------------------------
# 1️. Logistic Regression
# -----------------------------
with mlflow.start_run(run_name="LogisticRegression_v1"):
    lr = LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)
    lr.fit(X_train_scaled, y_train)
    y_pred = lr.predict(X_test_scaled)
    y_proba = lr.predict_proba(X_test_scaled)[:, 1]
    metrics = calcular_metricas(y_test, y_pred, y_proba)
    mlflow.log_params(lr.get_params())
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(lr, "logistic_regression_model_v1")

with mlflow.start_run(run_name="LogisticRegression_v2"):
    lr2 = LogisticRegression(max_iter=2000, solver="lbfgs", C=0.5, random_state=42)
    lr2.fit(X_train_scaled, y_train)
    y_pred = lr2.predict(X_test_scaled)
    y_proba = lr2.predict_proba(X_test_scaled)[:, 1]
    metrics = calcular_metricas(y_test, y_pred, y_proba)
    mlflow.log_params(lr2.get_params())
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(lr2, "logistic_regression_model_v2")

# -----------------------------
# 2️. Random Forest
# -----------------------------
with mlflow.start_run(run_name="RandomForest_v1"):
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=8, max_features="sqrt",
        min_samples_split=2, min_samples_leaf=1, bootstrap=True, random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]
    metrics = calcular_metricas(y_test, y_pred, y_proba)
    mlflow.log_params(rf.get_params())
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(rf, "random_forest_model_v1")

with mlflow.start_run(run_name="RandomForest_v2"):
    rf2 = RandomForestClassifier(
        n_estimators=600, max_depth=10, max_features="sqrt",
        min_samples_split=4, min_samples_leaf=2, bootstrap=True, random_state=42
    )
    rf2.fit(X_train, y_train)
    y_pred = rf2.predict(X_test)
    y_proba = rf2.predict_proba(X_test)[:, 1]
    metrics = calcular_metricas(y_test, y_pred, y_proba)
    mlflow.log_params(rf2.get_params())
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(rf2, "random_forest_model_v2")

# -----------------------------
# 3️. Gradient Boosting
# -----------------------------
with mlflow.start_run(run_name="GradientBoosting_v1"):
    gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    y_proba = gb.predict_proba(X_test)[:, 1]
    metrics = calcular_metricas(y_test, y_pred, y_proba)
    mlflow.log_params(gb.get_params())
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(gb, "gradient_boosting_model_v1")

with mlflow.start_run(run_name="GradientBoosting_v2"):
    gb2 = GradientBoostingClassifier(n_estimators=500, learning_rate=0.03, max_depth=4, random_state=42)
    gb2.fit(X_train, y_train)
    y_pred = gb2.predict(X_test)
    y_proba = gb2.predict_proba(X_test)[:, 1]
    metrics = calcular_metricas(y_test, y_pred, y_proba)
    mlflow.log_params(gb2.get_params())
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(gb2, "gradient_boosting_model_v2")

# -----------------------------
# 4️. XGBoost
# -----------------------------
with mlflow.start_run(run_name="XGBoost_v1"):
    xgb = XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=3, random_state=42)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    y_proba = xgb.predict_proba(X_test)[:, 1]
    metrics = calcular_metricas(y_test, y_pred, y_proba)
    mlflow.log_params(xgb.get_params())
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(xgb, "xgboost_model_v1")

with mlflow.start_run(run_name="XGBoost_v2"):
    xgb2 = XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42)
    xgb2.fit(X_train, y_train)
    y_pred = xgb2.predict(X_test)
    y_proba = xgb2.predict_proba(X_test)[:, 1]
    metrics = calcular_metricas(y_test, y_pred, y_proba)
    mlflow.log_params(xgb2.get_params())
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(xgb2, "xgboost_model_v2")

print("Todos los modelos registrados en MLflow exitosamente.")
