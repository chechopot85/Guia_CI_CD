import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 🔧 rutas seguras
tracking_path = os.path.abspath("mlruns")
tracking_uri = f"file://{tracking_path}"

# crear carpeta
os.makedirs(tracking_path, exist_ok=True)

# configurar MLflow
mlflow.set_tracking_uri(tracking_uri)

# 🔥 definir experimento correctamente
experiment_name = "ci-cd-mlflow-local"

if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(
        experiment_name,
        artifact_location=tracking_uri  # 👈 CLAVE
    )

mlflow.set_experiment(experiment_name)

# Datos
X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Predicción
predictions = model.predict(X_test)

# Métrica
mse = mean_squared_error(y_test, predictions)

# Registro
with mlflow.start_run():
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, name="model")

print(f"✅ MSE: {mse:.4f}")
