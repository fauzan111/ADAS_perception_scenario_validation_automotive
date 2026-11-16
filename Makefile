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

.PHONY: render-image render-calibration fit-calibration scenarios-measured

render-image:           # headless MetaDrive + torchvision (Docker; macOS can't render offscreen)
	docker build -f docker/Dockerfile.render -t adas-render .

render-calibration: render-image   # ~10 min: detector recall vs range on MetaDrive renders
	docker run --rm -v $(PWD):/work -v $(PWD)/results/torch_cache:/root/.cache/torch \
	  adas-render python scripts/calibrate_detector.py --out results/calibration

fit-calibration:        # -> configs/perception_model_measured.yaml + docs/figures/detector_recall.png
	uv run python scripts/fit_perception_model.py results/calibration

scenarios-measured:     # catalogue with the measured perception model
	uv run python -m adas_validation.experiments --seeds 30 \
	  --perception configs/perception_model_measured.yaml --out results/catalogue_measured
