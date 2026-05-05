import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 🔧 asegurar carpeta local
os.makedirs("mlruns", exist_ok=True)

# 🔧 usar ruta absoluta del entorno (IMPORTANTE en CI)
mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))

mlflow.set_experiment("ci-cd-mlflow-local")

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
    mlflow.sklearn.log_model(model, name="model")  # 👈 también corregido

print(f"✅ MSE: {mse:.4f}")
