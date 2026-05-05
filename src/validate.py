import mlflow
from mlflow.tracking import MlflowClient

# conectar a MLflow local
mlflow.set_tracking_uri("./mlruns")

client = MlflowClient()

# obtener experimentos
experiments = client.search_experiments()

if not experiments:
    raise Exception("❌ No hay experimentos registrados")

# obtener runs del primer experimento
runs = client.search_runs(experiments[0].experiment_id)

if not runs:
    raise Exception("❌ No hay runs registrados")

# obtener última métrica
latest_run = runs[0]
mse = latest_run.data.metrics.get("mse")

print(f"📊 MSE obtenido: {mse}")

# validación simple
if mse is None:
    raise Exception("❌ No se encontró MSE")

if mse > 5000:
    raise Exception("❌ Modelo con mal desempeño")

print("✅ Modelo validado correctamente")
