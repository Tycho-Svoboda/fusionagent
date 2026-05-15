"""Live smoke test for ResearchRetriever — prints results to stdout."""

import logging
import os
import sys
from pathlib import Path

from fusionagent.research.retriever import ResearchRetriever
from fusionagent.types import FusionCandidate


def _load_local_env() -> None:
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


_load_local_env()

# Show all log messages so you can see cache hits, warnings, etc.
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout,
                    format="%(levelname)s %(name)s: %(message)s")

candidate = FusionCandidate(
    ops=["rmsnorm", "silu", "mul"],
    input_shapes=[(2, 64, 128)],
    output_shape=(2, 64, 128),
    memory_bound=True,
    launch_overhead_us=7.0,
    graph_position=0,
)

print("=" * 60)
print("Retrieving research context for:", candidate.ops)
print("=" * 60)

retriever = ResearchRetriever()
ctx = retriever.retrieve(candidate)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(f"\nNovelty score: {ctx.novelty_score}")

print(f"\nPrior implementations ({len(ctx.prior_implementations)}):")
for i, impl in enumerate(ctx.prior_implementations, 1):
    print(f"  {i}. {impl}")

print(f"\nKnown pitfalls ({len(ctx.known_pitfalls)}):")
for i, pit in enumerate(ctx.known_pitfalls, 1):
    print(f"  {i}. {pit}")

print(f"\nSuggested tile sizes ({len(ctx.suggested_tile_sizes)}):")
for ts in ctx.suggested_tile_sizes:
    print(f"  {ts}")

print("\n" + "=" * 60)
print("Second call (should hit cache):")
print("=" * 60)
ctx2 = retriever.retrieve(candidate)
print(f"Novelty score: {ctx2.novelty_score}")
print("Done.")
