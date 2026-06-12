.PHONY: all data load transform train api frontend docs check

all: data load transform train

data:
	python data/generate_synthetic_data.py

load:
	python ingestion/load_to_bigquery.py

transform:
	cd dbt && dbt build

train:
	python ml/train_model.py

api:
	uvicorn ml.risk_api:app --reload

docs:
	cd dbt && dbt docs generate && dbt docs serve

frontend:
	streamlit run frontend/app.py

check:
	cd dbt && dbt test
