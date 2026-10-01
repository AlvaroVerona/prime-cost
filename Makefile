.PHONY: install generate-data validate profitability food-cost menu forecast staffing purchasing scenarios headline test lint dashboard all clean

PYTHON := python3

install:
	$(PYTHON) -m pip install -e ".[dev]"

generate-data:
	$(PYTHON) -m prime_cost.data.generator

validate:
	$(PYTHON) -m prime_cost.quality.validation

profitability:
	$(PYTHON) -m prime_cost.analytics.profitability

food-cost:
	$(PYTHON) -m prime_cost.analytics.food_cost

menu:
	$(PYTHON) -m prime_cost.analytics.menu_engineering

forecast:
	$(PYTHON) -m prime_cost.forecasting.demand

staffing:
	$(PYTHON) -m prime_cost.optimization.staffing

purchasing:
	$(PYTHON) -m prime_cost.optimization.purchasing

scenarios:
	$(PYTHON) -m prime_cost.simulation.scenarios

headline:
	$(PYTHON) -m prime_cost.reporting.headline

test:
	$(PYTHON) -m pytest -q

lint:
	ruff check src tests app

dashboard:
	streamlit run app/app.py

all: generate-data validate profitability food-cost menu forecast staffing purchasing scenarios headline test

clean:
	rm -rf data/raw/*.parquet data/raw/_truth.json data/processed/*.parquet data/processed/*.pkl
