install:
	pip install -r requirements.txt

train:
	python src/train.py

evaluate:
	python src/evaluate.py

all: install train evaluate

