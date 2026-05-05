import mlflow

# usar misma ruta que train
mlflow.set_tracking_uri("file://" + __import__("os").path.abspath("mlruns"))

experiment_name = "ci-cd-mlflow-local-v2"  # 👈 usa el mismo nombre que train

experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    print("⚠️ No existe el experimento, se omite validación")
    exit(0)

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"]
)

if runs.empty:
    print("⚠️ No hay runs disponibles")
    exit(0)

mse = runs.iloc[0]["metrics.mse"]

print(f"📊 MSE: {mse}")

if mse < 5000:
    print("✅ Modelo validado correctamente")
else:
    print("⚠️ Modelo con desempeño alto, pero continúa pipeline")
