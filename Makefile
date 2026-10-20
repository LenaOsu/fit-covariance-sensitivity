.PHONY: install demo test lint clean

install:
	pip install -e ".[dev,ml]"

demo:            ## regenerate data + all figures and tables (core + ML extension)
	covsens generate
	covsens run --config configs/demo.yaml
	covsens surrogate --config configs/demo.yaml

test:
	pytest -q

lint:
	ruff check src tests

clean:
	rm -f outputs/figures/*.png outputs/tables/*.csv
