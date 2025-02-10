# The Makefile for formatting and linting the Python project
SRC_PATHS = pygarment/ omages/ src/ configs/
MY_SRC_PATHS = src/ configs/
CONFIG_FILE = pyproject.toml
FLAKE8_CONFIG_FILE = .flake8

# Format Python files using black
format_black:
	@echo "Running black..."
	black $(SRC_PATHS) --config $(CONFIG_FILE)

# Sort imports using isort
format_isort:
	@echo "Running isort..."
	isort $(SRC_PATHS) --settings-path=$(CONFIG_FILE)

# Lint Python files using flake8
# add flake8-pyproject to allow toml
format_flake8:
	@echo "Running flake8..."
	flake8 $(MY_SRC_PATHS) --config=$(FLAKE8_CONFIG_FILE)

format_mypy:
	@echo "Running mypy..."
	python -m mypy $(SRC_PATHS) --config-file $(CONFIG_FILE)

# Run all formatters and linters
format: format_black format_isort format_flake8

.PHONY: format
