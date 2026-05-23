.PHONY: data fit report docs lint format clean all

data:
	@echo 'Data is baked into the Docker image.'

fit:
	python -m garch_btc_sp

report:
	mkdir -p outputs
	quarto render report.qmd --to html
	cp -r report.html outputs/
	if [ -d report_files ]; then cp -r report_files outputs/; fi

docs:
	cd docs && make html

lint:
	ruff check .

format:
	ruff format .

clean:
	rm -rf outputs/ _freeze/ docs/_build/

all: fit report
