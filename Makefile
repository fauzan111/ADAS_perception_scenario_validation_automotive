.PHONY: setup test lint fmt

setup:
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check .

fmt:
	uv run ruff format . && uv run ruff check --fix .

.PHONY: setup-all

setup-all:
	uv sync --all-extras

.PHONY: setup-sim scenarios

setup-sim:              # MetaDrive (needs Python 3.11: its panda3d pin has no 3.12 wheel on arm64)
	uv sync --extra sim

scenarios:              # catalogue x 30 perception seeds -> results/catalogue/report.md
	uv run python -m adas_validation.experiments --seeds 30
