# FusionAgent

RL-driven Triton kernel fusion agent.

## Quick Start

```bash
pip install -e ".[dev]"
pytest -m gpu
```

## Docker

```bash
docker build -t fusionagent .
docker run --gpus all fusionagent
```
