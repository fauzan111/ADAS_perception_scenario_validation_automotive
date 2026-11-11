# Compute plan (Apple Silicon)

CARLA has no macOS build and needs a Linux machine with an NVIDIA GPU.

| Stage | Where | Tool |
|---|---|---|
| Perception experiments | Mac (MPS) | PyTorch + nuScenes-mini (about 4 GB) |
| Scenario testing, fast loop | Mac | MetaDrive (`uv sync --extra sim`) |
| CARLA ScenarioRunner showcase | Cloud Linux GPU, rented by the hour | CARLA + ScenarioRunner |
| Heavy 3D detector training | Cloud GPU, or pretrained weights | MMDetection3D |

Develop metrics and reporting locally, then run one short cloud session for the final CARLA
batch, results table and video.
