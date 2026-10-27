.PHONY: install demo test lint clean

install:
	pip install -e ".[dev]"

demo:            ## regenerate data + all figures and tables
	covsens generate
	covsens run --config configs/demo.yaml

test:
	pytest -q

lint:
	ruff check src tests

clean:
	rm -f outputs/figures/*.png outputs/tables/*.csv
