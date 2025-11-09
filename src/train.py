import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = load_wine()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_tracking_uri("https://mlflow-super-g57-137680020436.us-central1.run.app")
mlflow.set_experiment("test-wine-hello")

with mlflow.start_run() as run:
    random_state =7
    model = RandomForestClassifier(n_estimators=50, random_state=random_state)
    mlflow.log_param("random_state", random_state)
    # Se fitea
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Log metrics & mod
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, name="model", input_example=X_train)
