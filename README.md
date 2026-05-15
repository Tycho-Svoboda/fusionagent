# FusionAgent

RL-driven Triton kernel fusion agent.

## Quick Start

```bash
pip install -e ".[dev]"
pytest -q
pytest -q -m "not gpu"
pytest -q -m gpu
```

## Docker

```bash
docker build -t fusionagent .
docker run --gpus all fusionagent
```

The default Docker command runs the GPU-marked test tier.
