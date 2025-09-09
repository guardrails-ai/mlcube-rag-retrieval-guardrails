
# Use uv as dependency manager, pip should work anyway
dev:
# 	pip install -e ".[dev]"
	uv sync --all-extras --no-cache

lint:
	ruff check .

test:
	pytest ./tests

type:
	pyright validator

qa:
	make lint
	make type
	make test