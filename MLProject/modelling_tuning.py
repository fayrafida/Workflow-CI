import pandas as pd
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import time

# Atur tracking URI ke server MLflow
mlflow.set_tracking_uri("http://localhost:5000")

# Load dataset hasil preprocessing
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn_preprocessing.csv")

# Pisahkan fitur dan label
X = df.drop("Churn_True", axis=1)
y = df["Churn_True"]

# Bagi data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definisikan grid search untuk tuning
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)

with mlflow.start_run():
    start_time = time.time()
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    latency = (time.time() - start_time) * 1000  # konversi ke milidetik

    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Logging ke MLflow
    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("latency_ms", latency)

    # Logging ke file JSON untuk Prometheus Exporter
    with open("metrics.json", "w") as f:
        json.dump({
            "accuracy": acc,
            "precision": prec,
            "rec": rec,
            "f1_score": f1,
            "latency_ms": latency
        }, f)

    # Simpan model ke MLflow artifacts
    mlflow.sklearn.log_model(best_model, "model")

    print("Model dan metrik berhasil dicatat di MLflow & disiapkan untuk Prometheus.")
