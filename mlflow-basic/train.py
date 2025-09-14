import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. Load dataset
# -----------------------------
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# -----------------------------
# 2. MLflow experiment setup
# -----------------------------
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("iris-rf-demo")

# -----------------------------
# 3. Train inside MLflow run
# -----------------------------
with mlflow.start_run():
    # hyperparameters
    n_estimators = 100
    max_depth = 3

    # model
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    clf.fit(X_train, y_train)

    # evaluate
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # -----------------------------
    # 4. Log params, metrics, model
    # -----------------------------
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(clf, "model")

    print(f"Logged model with accuracy {acc:.2f}")



"""
pip install -r requirements.txt
python train.py




Now go to MLflow UI → http://localhost:5000
You’ll see:

Parameters (n_estimators=100, max_depth=3)

Metrics (accuracy=0.93 for example)

Model artifact (RandomForest saved)
"""