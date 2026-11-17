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

.PHONY: scenarios-nuscenes sotif sweeps sotif-edge

scenarios-nuscenes:     # catalogue with the nuScenes-anchored perception model
	uv run python -m adas_validation.experiments --seeds 30 \
	  --perception configs/perception_model_nuscenes_anchored.yaml --out results/catalogue_nuscenes

sotif:                  # failure attribution (SOTIF table) for each perception model
	for m in assumed:perception_model measured:perception_model_measured \
	         nuscenes:perception_model_nuscenes_anchored; do \
	  uv run python -m adas_validation.sotif --perception configs/$${m#*:}.yaml \
	    --out results/sotif_$${m%%:*} || exit 1; done

sweeps:                 # pass/fail boundaries (~11 min) -> docs/figures/sweep_*.png
	uv run python -m adas_validation.sweeps \
	  --perception configs/perception_model_measured.yaml \
	               configs/perception_model_nuscenes_anchored.yaml \
	  --labels sim-measured nuScenes-anchored --out results/sweeps

sotif-edge:             # attribute the sweep-found edge cases (sim-measured model)
	uv run python -m adas_validation.sotif --catalogue configs/scenarios_edge.yaml \
	  --perception configs/perception_model_measured.yaml --out results/sotif_edge
