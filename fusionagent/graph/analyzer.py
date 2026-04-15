"""torch.fx graph analyzer for fusion candidate detection. 
The analyzer looks at a PyTorch model and finds groups of operations that could be merged into a single GPU kernel to run faster.

Why this matters: Every time the GPU runs an operation (like add, relu, multiply), there's overhead — launching the kernel, reading data from memory,
writing it back. If you can combine 3 operations into 1, you eliminate 2 kernel launches and 2 round-trips to memory. That's free speed.                    
                                                                
How it works:

1. It uses torch.fx.symbolic_trace to convert the model's forward() into a graph of nodes (each node = one operation)
2. It walks through the graph looking for 4 patterns:
- Elementwise chains — 2+ simple ops in a row (add → mul → relu). These are the easiest wins because they're all memory-bound (the GPU spends most of its
time moving data, not computing)
- Norm + activation + elementwise — like RMSNorm → SiLU → multiply, which is exactly what LLaMA does
- Linear + activation — a matrix multiply followed by relu/silu/gelu that aren't already fused
- Any sole-consumer pair — if op A's output goes to only op B, they might be fusible (catch-all)
3. For each match, it records: the op names, tensor shapes, whether it's memory-bound or compute-bound, estimated microseconds saved, and position in the
graph
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import operator
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
from torch.fx import Node
from torch.fx.passes.shape_prop import ShapeProp

from fusionagent.types import FusionCandidate

# ---------------------------------------------------------------------------
# Op classification tables
# ---------------------------------------------------------------------------

# Functions that are elementwise (1-to-1 mapping, no reduction / reshape).
_ELEMENTWISE_FUNCTIONS: Set = {
    torch.add, torch.sub, torch.mul, torch.div,
    torch.neg, torch.abs, torch.pow, torch.sqrt, torch.rsqrt,
    torch.exp, torch.log, torch.clamp,
    torch.sigmoid, torch.tanh,
    F.relu, F.silu, F.gelu, F.leaky_relu, F.elu,
    F.dropout, F.sigmoid, F.tanh,
    # torch.relu is a distinct builtin from F.relu (traced from torch.relu(x))
    torch.relu,
    # Python operators — torch.fx traces `x + y` as `operator.add`, etc.
    operator.add, operator.sub, operator.mul, operator.truediv,
    operator.floordiv, operator.mod, operator.pow,
    operator.neg, operator.abs,
    operator.iadd, operator.isub, operator.imul, operator.itruediv,
}

_ELEMENTWISE_METHODS: Set[str] = {
    "add", "sub", "mul", "div", "neg", "abs", "pow",
    "sqrt", "rsqrt", "exp", "log", "clamp",
    "sigmoid", "tanh", "relu", "silu", "gelu",
    "add_", "mul_", "sub_", "div_",  # in-place variants
    "mean",  # reduction but appears in decomposed norms
}

# Shape manipulation / dtype cast ops — zero compute, not worth fusing alone.
_RESHAPE_OPS: Set[str] = {
    "view", "reshape", "contiguous", "to", "type", "float", "half", "bfloat16",
    "transpose", "permute", "unsqueeze", "squeeze", "flatten", "unflatten",
    "expand", "repeat", "t",
}
_RESHAPE_FUNCTIONS: Set = {
    torch.transpose, torch.reshape, torch.flatten, torch.unsqueeze, torch.squeeze,
}
_ACCESSOR_FUNCTIONS: Set = {getattr, operator.getitem}

# Activation functions specifically (subset of elementwise).
_ACTIVATION_FUNCTIONS: Set = {
    F.relu, F.silu, F.gelu, F.leaky_relu, F.elu,
    F.sigmoid, F.tanh, torch.sigmoid, torch.tanh,
    torch.relu,
}

_ACTIVATION_METHODS: Set[str] = {"relu", "silu", "gelu", "sigmoid", "tanh"}

# Module types that count as activations.
_ACTIVATION_MODULES: Tuple[type, ...] = (
    nn.ReLU, nn.SiLU, nn.GELU, nn.LeakyReLU, nn.ELU,
    nn.Sigmoid, nn.Tanh,
)

# Module types that are elementwise (activations + dropout).
_ELEMENTWISE_MODULES: Tuple[type, ...] = _ACTIVATION_MODULES + (nn.Dropout,)

# Norm module types.
_NORM_MODULES: Tuple[type, ...] = (nn.LayerNorm,)

# Kernel launch overhead estimate (μs) per fused op eliminated.
_LAUNCH_OVERHEAD_PER_OP_US = 3.5

# Arithmetic intensity threshold (FLOPs / bytes) below which we call it
# memory-bound. H100 SXM: ~3 TFLOP/s FP32, ~3.35 TB/s HBM → crossover ≈ 900.
# In practice most elementwise chains are obviously memory-bound, so we use a
# conservative threshold.
_COMPUTE_BOUND_AI_THRESHOLD = 50.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _op_label(node: Node, modules: Dict[str, nn.Module]) -> str:
    """Return a short human-readable label for *node*."""
    if node.op == "call_function":
        return node.target.__name__
    if node.op == "call_method":
        return node.target
    if node.op == "call_module":
        mod = modules.get(node.target)
        if mod is None:
            return node.target
        return type(mod).__name__.lower()
    return node.name


def _is_elementwise(node: Node, modules: Dict[str, nn.Module]) -> bool:
    if node.op == "call_function":
        return node.target in _ELEMENTWISE_FUNCTIONS
    if node.op == "call_method":
        return node.target in _ELEMENTWISE_METHODS
    if node.op == "call_module":
        mod = modules.get(node.target)
        return isinstance(mod, _ELEMENTWISE_MODULES) if mod else False
    return False


def _is_activation(node: Node, modules: Dict[str, nn.Module]) -> bool:
    if node.op == "call_function":
        return node.target in _ACTIVATION_FUNCTIONS
    if node.op == "call_method":
        return node.target in _ACTIVATION_METHODS
    if node.op == "call_module":
        mod = modules.get(node.target)
        return isinstance(mod, _ACTIVATION_MODULES) if mod else False
    return False


def _is_norm(node: Node, modules: Dict[str, nn.Module]) -> bool:
    if node.op == "call_module":
        mod = modules.get(node.target)
        if mod is None:
            return False
        # Check for RMSNorm by name since it's often a custom module.
        if isinstance(mod, _NORM_MODULES):
            return True
        cls_name = type(mod).__name__
        return "RMSNorm" in cls_name or "rmsnorm" in cls_name.lower()
    return False


def _is_linear(node: Node, modules: Dict[str, nn.Module]) -> bool:
    if node.op == "call_module":
        mod = modules.get(node.target)
        return isinstance(mod, nn.Linear) if mod else False
    if node.op == "call_function":
        return node.target is F.linear
    return False


def _node_shape(node: Node) -> Optional[tuple]:
    """Extract shape from node meta (set by ShapeProp)."""
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor):
        return tuple(val.shape)
    shape = node.meta.get("tensor_meta")
    if shape is not None:
        return tuple(shape.shape)
    # Fallback: ShapeProp stores directly in some versions.
    example = node.meta.get("example_value")
    if isinstance(example, torch.Tensor):
        return tuple(example.shape)
    return None


def _input_nodes(node: Node) -> List[Node]:
    """Return nodes that are *data* inputs to *node*."""
    out: List[Node] = []
    for a in node.args:
        if isinstance(a, Node):
            out.append(a)
    for v in node.kwargs.values():
        if isinstance(v, Node):
            out.append(v)
    return out


def _sole_user(node: Node) -> Optional[Node]:
    """If *node* feeds into exactly one other (non-output) node, return it."""
    real = [u for u in node.users if u.op != "output"]
    return real[0] if len(real) == 1 else None


def _is_reshape_or_accessor(node: Node, modules: Dict[str, nn.Module]) -> bool:
    """True for pure shape manipulation, dtype casts, or attribute access."""
    if node.op == "call_method":
        return node.target in _RESHAPE_OPS
    if node.op == "call_function":
        return node.target in (_RESHAPE_FUNCTIONS | _ACCESSOR_FUNCTIONS) or node.target is getattr
    return False


def _is_compute_node(node: Node) -> bool:
    return node.op in ("call_function", "call_method", "call_module")


def _estimate_memory_bound(
    ops: List[str],
    input_shapes: List[tuple],
    output_shape: Optional[tuple],
    modules: Dict[str, nn.Module],
    nodes: List[Node],
) -> bool:
    """Heuristic: if the chain is all elementwise / norm, it's memory-bound.
    Linear layers (matmuls) may be compute-bound for large enough N*K."""
    has_matmul = False
    for n in nodes:
        if _is_linear(n, modules):
            has_matmul = True
            break
        if n.op == "call_function" and n.target in (torch.matmul, torch.bmm):
            has_matmul = True
            break

    if not has_matmul:
        # Pure elementwise / norm chain → memory-bound.
        return True

    # For chains containing a linear: estimate arithmetic intensity.
    # Linear (M, K) @ (K, N) → 2*M*K*N FLOPs, reads M*K + K*N, writes M*N
    # We approximate using the output shape.
    if output_shape and len(output_shape) >= 2:
        m = 1
        for d in output_shape[:-1]:
            m *= d
        n = output_shape[-1]
        # Rough: 2*m*n FLOPs, (m + n + m*n)*element_size bytes
        flops = 2.0 * m * n
        bytes_moved = (m * n + m + n) * 4  # fp32
        ai = flops / max(bytes_moved, 1)
        return ai < _COMPUTE_BOUND_AI_THRESHOLD

    return True  # default to memory-bound


def _estimate_launch_overhead(num_ops: int) -> float:
    """Estimated μs saved by fusing *num_ops* into one kernel."""
    return max(0.0, (num_ops - 1) * _LAUNCH_OVERHEAD_PER_OP_US)


# ---------------------------------------------------------------------------
# Pattern detectors
# ---------------------------------------------------------------------------

_NORM_SIGNATURE: Set[str] = {"mean", "rsqrt"}


def _pattern_decomposed_norm(
    node: Node,
    modules: Dict[str, nn.Module],
) -> Optional[Tuple[List[Node], List[str]]]:
    """Detect an inlined RMSNorm/LayerNorm decomposed by symbolic_trace.

    Walks a sole-consumer chain starting from a `pow` or `float` call_method.
    If the chain contains `mean` AND `rsqrt` and has >= 4 ops, it's a
    decomposed norm.  Returns (chain_nodes, semantic_ops) or None.

    After the norm chain ends, absorbs downstream activation and elementwise
    ops if they are sole-consumer continuations.
    """
    if node.op != "call_method":
        return None
    if node.target not in ("pow", "float"):
        return None

    # --- Phase 1: walk the norm body ---
    chain: List[Node] = [node]
    seen_ops: Set[str] = set()
    label = node.target if node.op == "call_method" else _op_label(node, modules)
    seen_ops.add(label)

    cur = node
    while True:
        nxt = _sole_user(cur)
        if nxt is None:
            break
        if not (_is_elementwise(nxt, modules) or _is_reshape_or_accessor(nxt, modules)):
            break
        chain.append(nxt)
        lbl = nxt.target if nxt.op == "call_method" else _op_label(nxt, modules)
        seen_ops.add(lbl)
        cur = nxt

    # Must contain the defining ops of a norm and be long enough.
    if not _NORM_SIGNATURE.issubset(seen_ops):
        return None
    if len(chain) < 4:
        return None

    # --- Phase 2: absorb downstream activation + elementwise ---
    semantic_ops: List[str] = ["rmsnorm"]
    absorbed_nodes: List[Node] = list(chain)

    nxt = _sole_user(cur)
    if nxt is not None and _is_activation(nxt, modules):
        absorbed_nodes.append(nxt)
        semantic_ops.append(_op_label(nxt, modules))
        cur = nxt
        # One more: absorb an elementwise op after the activation (e.g. mul).
        nxt = _sole_user(cur)
        if nxt is not None and _is_elementwise(nxt, modules) and not _is_activation(nxt, modules):
            absorbed_nodes.append(nxt)
            semantic_ops.append(_op_label(nxt, modules))

    return absorbed_nodes, semantic_ops


def _chain_elementwise(
    start: Node,
    modules: Dict[str, nn.Module],
    visited: Set[str],
) -> Optional[List[Node]]:
    """Walk a chain of consecutive elementwise ops starting at *start*.

    Returns the chain (length >= 2) or None.
    Rejects chains where every op is a reshape/accessor (zero compute).
    """
    if not _is_elementwise(start, modules):
        return None

    chain = [start]
    cur = start
    while True:
        nxt = _sole_user(cur)
        if nxt is None or not _is_elementwise(nxt, modules):
            break
        chain.append(nxt)
        cur = nxt

    if len(chain) < 2:
        return None

    # Reject if ALL ops are reshape/accessor — no compute to fuse.
    if all(_is_reshape_or_accessor(n, modules) for n in chain):
        return None

    return chain


def _pattern_norm_act_elem(
    node: Node,
    modules: Dict[str, nn.Module],
) -> Optional[List[Node]]:
    """Norm → activation → elementwise (e.g. RMSNorm → SiLU → mul)."""
    if not _is_norm(node, modules):
        return None
    act = _sole_user(node)
    if act is None or not _is_activation(act, modules):
        return None
    elem = _sole_user(act)
    if elem is None or not _is_elementwise(elem, modules):
        return None
    return [node, act, elem]


def _pattern_linear_act(
    node: Node,
    modules: Dict[str, nn.Module],
) -> Optional[List[Node]]:
    """Linear → activation (when not already fused)."""
    if not _is_linear(node, modules):
        return None
    act = _sole_user(node)
    if act is None or not _is_activation(act, modules):
        return None
    return [node, act]


def _pattern_sole_pair(
    node: Node,
    modules: Dict[str, nn.Module],
    visited: Set[str],
) -> Optional[List[Node]]:
    """Any two ops where output of first feeds *only* into second (no branches)."""
    if not _is_compute_node(node):
        return None
    nxt = _sole_user(node)
    if nxt is None or not _is_compute_node(nxt):
        return None
    # Only fire if this pair wasn't already captured by another pattern.
    if node.name in visited or nxt.name in visited:
        return None
    # Skip if BOTH nodes are reshape/accessor — no compute to fuse.
    if _is_reshape_or_accessor(node, modules) and _is_reshape_or_accessor(nxt, modules):
        return None
    return [node, nxt]


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

_logger = logging.getLogger(__name__)

_LLM_SYSTEM_PROMPT = """You are a GPU kernel fusion expert. Given a description of \
PyTorch graph nodes (op name, inputs, output shape), identify groups of nodes \
that could be fused into a single Triton kernel for better performance.

Return ONLY valid JSON in this exact format:
{
  "candidates": [
    {
      "node_names": ["node_a", "node_b"],
      "ops": ["rmsnorm", "silu"],
      "reason": "short explanation",
      "memory_bound": true,
      "estimated_benefit": "high|medium|low"
    }
  ]
}
Return {"candidates": []} if no opportunities found. No other text."""

_LLM_WINDOW_SIZE = 8


def _serialize_subgraph(nodes: List[Node], modules: Dict[str, nn.Module]) -> str:
    """Compact text description of unvisited nodes for LLM consumption."""
    lines = []
    for n in nodes:
        label = _op_label(n, modules)
        inputs = [i.name for i in _input_nodes(n) if isinstance(i, Node)]
        shape = _node_shape(n)
        lines.append(f"  {n.name} ({label}) <- {inputs} | out_shape={shape}")
    return "\n".join(lines)


def _llm_analyze_subgraph(
    description: str,
    llm_model: str,
) -> List[dict]:
    """Call the OpenAI API and return parsed candidate dicts.

    Returns an empty list on any error — never raises.
    """
    try:
        import openai  # type: ignore
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user", "content": description},
            ],
            temperature=0.0,
        )
        raw = response.choices[0].message.content or ""
        parsed = json.loads(raw)
        return parsed.get("candidates", [])
    except Exception as exc:
        _logger.warning("LLM subgraph analysis failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class GraphAnalyzer:
    """Analyze a ``torch.nn.Module`` for kernel-fusion opportunities.

    Uses ``torch.fx.symbolic_trace`` to capture the computation graph, then
    applies a set of pattern detectors to identify fusion candidates.
    """

    def __init__(
        self,
        model: nn.Module,
        sample_input: Optional[torch.Tensor] = None,
        use_llm: bool = False,
        llm_model: str = "gpt-5.4",
    ) -> None:
        self._model = model
        self._sample_input = sample_input
        self._use_llm = use_llm
        self._llm_model = llm_model

    # ------------------------------------------------------------------
    def analyze(self) -> List[FusionCandidate]:
        """Return a list of :class:`FusionCandidate` objects."""
        traced = fx.symbolic_trace(self._model)
        modules: Dict[str, nn.Module] = dict(traced.named_modules())

        # Shape propagation (best-effort).
        if self._sample_input is not None:
            try:
                ShapeProp(traced).propagate(self._sample_input)
            except Exception:
                pass  # shapes will be None – still useful without them

        # Build node index for graph_position.
        node_list: List[Node] = list(traced.graph.nodes)
        node_idx = {n.name: i for i, n in enumerate(node_list)}

        visited: Set[str] = set()
        candidates: List[FusionCandidate] = []

        for node in node_list:
            if node.name in visited:
                continue

            chain: Optional[List[Node]] = None
            semantic_ops: Optional[List[str]] = None

            # Pattern 0 (NEW): decomposed norm (inlined RMSNorm/LayerNorm).
            # Highest priority — most specific pattern.
            result = _pattern_decomposed_norm(node, modules)
            if result is not None:
                chain, semantic_ops = result

            # Pattern 1: norm + activation + elementwise  (module-level norms).
            if chain is None:
                chain = _pattern_norm_act_elem(node, modules)

            # Pattern 2: linear + activation
            if chain is None:
                chain = _pattern_linear_act(node, modules)

            # Pattern 3: elementwise chain (>= 2)
            if chain is None:
                chain = _chain_elementwise(node, modules, visited)

            # Pattern 4: sole-consumer pair (catch-all)
            if chain is None:
                chain = _pattern_sole_pair(node, modules, visited)

            if chain is None:
                continue

            # Mark all nodes in this chain as visited.
            for n in chain:
                visited.add(n.name)

            # Gather metadata.
            ops = semantic_ops if semantic_ops is not None else [_op_label(n, modules) for n in chain]
            graph_position = node_idx[chain[0].name]

            # Shapes — collect input shapes from the first node's data inputs
            # and the output shape from the last node.
            first_inputs = _input_nodes(chain[0])
            input_shapes = [
                _node_shape(inp)
                for inp in first_inputs
                if _node_shape(inp) is not None
            ]
            output_shape = _node_shape(chain[-1])

            memory_bound = _estimate_memory_bound(
                ops, input_shapes, output_shape, modules, chain,
            )
            launch_overhead = _estimate_launch_overhead(len(chain))

            candidates.append(
                FusionCandidate(
                    ops=ops,
                    input_shapes=input_shapes,
                    output_shape=output_shape,
                    memory_bound=memory_bound,
                    launch_overhead_us=launch_overhead,
                    graph_position=graph_position,
                )
            )

        if self._use_llm:
            llm_candidates = self._llm_pass(node_list, modules, visited)
            candidates.extend(llm_candidates)

        return candidates

    # ------------------------------------------------------------------
    def _llm_pass(
        self,
        node_list: List[Node],
        modules: Dict[str, nn.Module],
        visited: Set[str],
    ) -> List[FusionCandidate]:
        """Second analysis pass: send unvisited compute subgraphs to the LLM.

        Never raises — returns an empty list on any error.
        """
        if not os.environ.get("OPENAI_API_KEY"):
            _logger.warning(
                "LLM pass requested but OPENAI_API_KEY is not set; skipping."
            )
            return []

        node_idx = {n.name: i for i, n in enumerate(node_list)}

        # Collect unvisited compute nodes (skip pure reshape/accessor).
        unvisited = [
            n for n in node_list
            if n.name not in visited
            and _is_compute_node(n)
            and not _is_reshape_or_accessor(n, modules)
        ]

        if not unvisited:
            return []

        candidates: List[FusionCandidate] = []
        seen_sets: List[frozenset] = []  # for deduplication across windows

        # Slide a window of up to _LLM_WINDOW_SIZE nodes over the unvisited list.
        step = max(1, _LLM_WINDOW_SIZE // 2)
        for start in range(0, len(unvisited), step):
            window = unvisited[start: start + _LLM_WINDOW_SIZE]
            if not window:
                break

            description = _serialize_subgraph(window, modules)
            llm_results = _llm_analyze_subgraph(description, self._llm_model)

            for item in llm_results:
                node_names = item.get("node_names", [])
                if not node_names:
                    continue

                key = frozenset(node_names)
                if key in seen_sets:
                    continue
                seen_sets.append(key)

                # Resolve nodes by name.
                name_to_node = {n.name: n for n in window}
                matched = [name_to_node[nm] for nm in node_names if nm in name_to_node]
                if not matched:
                    continue

                ops = item.get("ops") or [_op_label(n, modules) for n in matched]
                memory_bound = bool(item.get("memory_bound", True))
                first_inputs = _input_nodes(matched[0])
                input_shapes = [
                    _node_shape(inp)
                    for inp in first_inputs
                    if _node_shape(inp) is not None
                ]
                output_shape = _node_shape(matched[-1])
                graph_position = node_idx[matched[0].name]

                candidates.append(
                    FusionCandidate(
                        ops=ops,
                        input_shapes=input_shapes,
                        output_shape=output_shape,
                        memory_bound=memory_bound,
                        launch_overhead_us=_estimate_launch_overhead(len(matched)),
                        graph_position=graph_position,
                    )
                )

        return candidates

    # ------------------------------------------------------------------
    @staticmethod
    def model_hash(model: nn.Module) -> str:
        """Deterministic 8-char hex hash: sha256(ClassName + param_count)."""
        param_count = sum(p.numel() for p in model.parameters())
        key = model.__class__.__name__ + str(param_count)
        return hashlib.sha256(key.encode()).hexdigest()[:8]

    # ------------------------------------------------------------------
    def save_candidates(
        self,
        candidates: List[FusionCandidate],
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Serialize *candidates* to ``candidates.json`` and return the path."""
        if output_dir is None:
            h = self.model_hash(self._model)
            output_dir = Path("runs") / h
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "candidates.json"
        out_path.write_text(
            json.dumps([dataclasses.asdict(c) for c in candidates], indent=2)
        )
        return out_path

    # ------------------------------------------------------------------
    def analyze_and_save(
        self,
        output_dir: Optional[Path] = None,
    ) -> Tuple[List[FusionCandidate], Path]:
        """Run analysis then persist results. Returns (candidates, json_path)."""
        candidates = self.analyze()
        path = self.save_candidates(candidates, output_dir=output_dir)
        return candidates, path
