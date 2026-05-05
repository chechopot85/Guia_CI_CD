# Makefile

install:
	python3 -m pip install -r requirements.txt

train:
	python3 src/train.py

validate:
	python3 src/validate.py

ci: install train validate

cd:
	@echo "🚀 Promoción/despliegue del modelo"
	@echo "Ejemplo: mlflow models serve"
