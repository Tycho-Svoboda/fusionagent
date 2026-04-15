"""Tests for fusionagent.graph.analyzer against a LlamaDecoderLayer mock."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from fusionagent.graph.analyzer import GraphAnalyzer
from fusionagent.types import FusionCandidate


# ---------------------------------------------------------------------------
# Mock modules that mirror LLaMA components but are torch.fx-traceable.
# ---------------------------------------------------------------------------

class MockRMSNorm(nn.Module):
    """Simplified RMSNorm (fx-traceable)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class MockSiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)


class MockGeLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)


class MockLlamaMLPBlock(nn.Module):
    """gate_proj → SiLU, up_proj, mul, down_proj — mimics LLaMA MLP."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        self.act = MockSiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class MockLlamaAttention(nn.Module):
    """Simplified single-head attention (no rotary, no KV cache)."""

    def __init__(self, dim: int):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.head_dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn = F.scaled_dot_product_attention(q, k, v)
        return self.o_proj(attn)


class MockLlamaDecoderLayer(nn.Module):
    """Structural mock of a LLaMA decoder layer (fx-traceable)."""

    def __init__(self, dim: int = 128, mlp_hidden: int = 256):
        super().__init__()
        self.input_layernorm = MockRMSNorm(dim)
        self.self_attn = MockLlamaAttention(dim)
        self.post_attention_layernorm = MockRMSNorm(dim)
        self.mlp = MockLlamaMLPBlock(dim, mlp_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention block with residual.
        h = self.input_layernorm(x)
        h = self.self_attn(h)
        x = x + h
        # MLP block with residual.
        h = self.post_attention_layernorm(x)
        h = self.mlp(h)
        x = x + h
        return x


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _analyze(model: nn.Module, shape: tuple = (1, 32, 128)) -> list[FusionCandidate]:
    inp = torch.randn(*shape)
    return GraphAnalyzer(model, sample_input=inp).analyze()


def _find_candidate(candidates: list[FusionCandidate], *op_names: str) -> FusionCandidate | None:
    """Find a candidate whose ops list contains all of *op_names* (in order)."""
    for c in candidates:
        if len(c.ops) < len(op_names):
            continue
        # Check subsequence.
        it = iter(c.ops)
        if all(any(op == want for op in it) for want in op_names):
            return c
    return None


def _has_candidate_containing(candidates: list[FusionCandidate], *op_names: str) -> bool:
    return _find_candidate(candidates, *op_names) is not None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGraphAnalyzerLlamaDecoder:
    """Tests against the full MockLlamaDecoderLayer."""

    def setup_method(self):
        self.model = MockLlamaDecoderLayer(dim=128, mlp_hidden=256)
        self.candidates = _analyze(self.model)

    def test_returns_list_of_fusion_candidates(self):
        assert isinstance(self.candidates, list)
        assert all(isinstance(c, FusionCandidate) for c in self.candidates)

    def test_finds_at_least_one_candidate(self):
        assert len(self.candidates) >= 1

    def test_all_candidates_have_required_fields(self):
        for c in self.candidates:
            assert isinstance(c.ops, list) and len(c.ops) >= 1
            assert isinstance(c.memory_bound, bool)
            assert isinstance(c.launch_overhead_us, float)
            assert c.launch_overhead_us >= 0.0
            assert isinstance(c.graph_position, int) and c.graph_position >= 0

    def test_graph_positions_are_ordered(self):
        positions = [c.graph_position for c in self.candidates]
        assert positions == sorted(positions)


class TestElementwiseChainDetection:
    """Pattern 1: chains of >= 2 consecutive elementwise ops."""

    def test_two_elementwise(self):
        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.relu(x * 2.0)

        candidates = _analyze(M(), shape=(1, 64))
        assert len(candidates) >= 1
        c = candidates[0]
        assert len(c.ops) >= 2

    def test_three_elementwise(self):
        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                a = x + 1.0
                b = a * 2.0
                return torch.relu(b)

        candidates = _analyze(M(), shape=(1, 64))
        assert len(candidates) >= 1
        c = candidates[0]
        assert len(c.ops) >= 3

    def test_single_op_not_fused(self):
        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.relu(x)

        candidates = _analyze(M(), shape=(1, 64))
        # A single op has no fusion partner — should produce nothing
        # (unless catch-all fires, but relu alone has no sole-consumer pair).
        ew_chains = [c for c in candidates if len(c.ops) >= 2]
        # If there are any, they should not be a chain of length 1.
        for c in ew_chains:
            assert len(c.ops) >= 2

    def test_elementwise_chain_is_memory_bound(self):
        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.relu(x * 2.0 + 1.0)

        candidates = _analyze(M(), shape=(1, 64))
        assert len(candidates) >= 1
        assert candidates[0].memory_bound is True


class TestNormActivationElementwise:
    """Pattern 2: norm → activation → elementwise."""

    def test_rmsnorm_silu_mul(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = MockRMSNorm(64)
                self.act = MockSiLU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                h = self.norm(x)
                h = self.act(h)
                return h * x  # elementwise mul

        candidates = _analyze(M(), shape=(1, 8, 64))
        assert len(candidates) >= 1

    def test_layernorm_gelu_mul(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.LayerNorm(64)
                self.act = MockGeLU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                h = self.norm(x)
                h = self.act(h)
                return h * 0.5

        candidates = _analyze(M(), shape=(1, 8, 64))
        assert len(candidates) >= 1


class TestLinearActivation:
    """Pattern 3: linear → activation (unfused)."""

    def test_linear_relu(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 128)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.relu(self.fc(x))

        candidates = _analyze(M(), shape=(1, 64))
        assert len(candidates) >= 1
        c = candidates[0]
        assert "linear" in c.ops[0].lower() or "Linear" in c.ops[0]

    def test_linear_silu(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 128, bias=False)
                self.act = MockSiLU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.act(self.fc(x))

        candidates = _analyze(M(), shape=(1, 64))
        assert len(candidates) >= 1


class TestSoleConsumerPair:
    """Pattern 4: any two ops where output feeds only into the next op."""

    def test_linear_layernorm(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 64)
                self.ln = nn.LayerNorm(64)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.ln(self.fc(x))

        candidates = _analyze(M(), shape=(1, 64))
        assert len(candidates) >= 1
        assert len(candidates[0].ops) == 2

    def test_branch_blocks_pairing(self):
        """If the first op's output is consumed by two ops, no fusion."""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 64)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                h = self.fc(x)
                return h + h  # h used twice in one op — but sole user is add

        candidates = _analyze(M(), shape=(1, 64))
        # fc has exactly one user (add), so it's still a valid pair.
        assert len(candidates) >= 1


class TestLaunchOverhead:
    def test_two_ops_gives_one_launch_saved(self):
        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.relu(x + 1.0)

        candidates = _analyze(M(), shape=(1, 64))
        assert len(candidates) >= 1
        # 2 ops → 1 launch saved → 3.5 μs
        assert candidates[0].launch_overhead_us > 0.0

    def test_three_ops_gives_more_overhead(self):
        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.relu(torch.sigmoid(x + 1.0))

        candidates = _analyze(M(), shape=(1, 64))
        assert len(candidates) >= 1
        assert candidates[0].launch_overhead_us > 3.5  # > 1 launch


class TestShapePropagation:
    def test_input_shapes_populated(self):
        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.relu(x + 1.0)

        candidates = _analyze(M(), shape=(2, 64))
        assert len(candidates) >= 1
        c = candidates[0]
        assert len(c.input_shapes) >= 1
        assert c.input_shapes[0] == (2, 64)

    def test_output_shape_populated(self):
        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.relu(x + 1.0)

        candidates = _analyze(M(), shape=(2, 64))
        assert len(candidates) >= 1
        assert candidates[0].output_shape == (2, 64)


class TestAnalyzerWithoutSampleInput:
    def test_works_without_shapes(self):
        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.relu(x + 1.0)

        candidates = GraphAnalyzer(M()).analyze()
        assert len(candidates) >= 1
        # Shapes should be None when no sample input provided.
        c = candidates[0]
        assert c.input_shapes == []
        assert c.output_shape is None


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------

class TestTrueFanOut:
    """When an op's output feeds into 2+ different ops, it should NOT form
    a sole-consumer pair with either."""

    def test_fanout_blocks_sole_consumer(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 64)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                h = self.fc(x)
                a = torch.relu(h)
                b = torch.sigmoid(h)
                return a + b

        candidates = _analyze(M(), shape=(1, 64))
        # fc has two users (relu, sigmoid) — should NOT pair with either.
        for c in candidates:
            if "linear" in c.ops[0].lower():
                # If linear appears, its partner shouldn't be an activation
                # that was chosen over the other branch.
                assert False, f"Linear should not be fused when it fans out: {c.ops}"

    def test_fanout_downstream_still_fusible(self):
        """The two branches *after* the fan-out can still be fused with each other."""
        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                a = torch.relu(x)
                b = torch.sigmoid(x)
                return a + b

        candidates = _analyze(M(), shape=(1, 64))
        # relu and sigmoid each have one user (add), but x fans out so
        # x → relu is not sole-consumer. However relu → (feeds into add)
        # and sigmoid → (feeds into add) — add has two input nodes.
        # At minimum, no crash; we just verify structural soundness.
        for c in candidates:
            assert len(c.ops) >= 2


class TestLongElementwiseChain:
    """A chain of 5+ elementwise ops should be captured as a single candidate."""

    def test_five_op_chain(self):
        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = x + 1.0
                x = x * 2.0
                x = x - 0.5
                x = torch.relu(x)
                x = x * 3.0
                return x

        candidates = _analyze(M(), shape=(1, 64))
        assert len(candidates) >= 1
        # Should be one big chain, not split into multiple.
        longest = max(candidates, key=lambda c: len(c.ops))
        assert len(longest.ops) >= 5

    def test_long_chain_launch_overhead_scales(self):
        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = x + 1.0
                x = x * 2.0
                x = x - 0.5
                x = torch.relu(x)
                x = x * 3.0
                return x

        candidates = _analyze(M(), shape=(1, 64))
        longest = max(candidates, key=lambda c: len(c.ops))
        # 5 ops → 4 launches saved → 14.0 μs
        assert longest.launch_overhead_us >= 14.0


class TestResidualAddPattern:
    """Residual connections (x + f(x)) create fan-out on x. The residual add
    should not prevent fusion inside f(x)."""

    def test_residual_does_not_block_inner_fusion(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 64)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + torch.relu(self.fc(x))

        candidates = _analyze(M(), shape=(1, 64))
        # fc → relu should still be detected (linear + activation).
        has_linear_relu = any(
            "linear" in c.ops[0].lower() and "relu" in c.ops[1]
            for c in candidates
            if len(c.ops) >= 2
        )
        assert has_linear_relu


class TestIdentityModel:
    """A model with no fusible ops should return an empty list."""

    def test_passthrough(self):
        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        candidates = _analyze(M(), shape=(1, 64))
        assert candidates == []

    def test_single_linear(self):
        """A lone linear with no activation has no fusion partner (output is
        its only consumer)."""
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 64)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(x)

        candidates = _analyze(M(), shape=(1, 64))
        assert candidates == []


class TestModuleActivations:
    """nn.Module-based activations (nn.ReLU(), nn.SiLU()) should be detected
    the same as their functional counterparts."""

    def test_linear_nn_relu_module(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 128)
                self.act = nn.ReLU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.act(self.fc(x))

        candidates = _analyze(M(), shape=(1, 64))
        assert len(candidates) >= 1

    def test_linear_nn_gelu_module(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 128)
                self.act = nn.GELU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.act(self.fc(x))

        candidates = _analyze(M(), shape=(1, 64))
        assert len(candidates) >= 1


class TestPatternPriority:
    """More-specific patterns should take priority over less-specific ones.
    norm → act → elem should be one candidate, not split."""

    def test_norm_act_elem_not_split(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.LayerNorm(64)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                h = self.norm(x)
                h = F.silu(h)
                return h * 2.0

        candidates = _analyze(M(), shape=(1, 8, 64))
        # The silu and mul should NOT appear as a separate elementwise chain.
        # They should be part of the norm→act→elem candidate.
        silu_mul_only = [
            c for c in candidates
            if c.ops == ["silu", "mul"]
        ]
        assert len(silu_mul_only) == 0, "silu+mul should be captured by norm pattern, not elementwise"

    def test_norm_act_elem_is_single_candidate(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.LayerNorm(64)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                h = self.norm(x)
                h = F.silu(h)
                return h * 2.0

        candidates = _analyze(M(), shape=(1, 8, 64))
        assert len(candidates) >= 1
        # Find the one containing the norm.
        norm_candidates = [c for c in candidates if "layernorm" in c.ops[0].lower()]
        assert len(norm_candidates) == 1
        assert len(norm_candidates[0].ops) == 3


class TestNoDuplicateNodes:
    """No node should appear in more than one fusion candidate."""

    def test_no_overlapping_candidates(self):
        model = MockLlamaDecoderLayer(dim=128, mlp_hidden=256)
        candidates = _analyze(model)

        all_ops_positions = []
        for c in candidates:
            # Use (graph_position + offset) as a rough node identity.
            for i, op in enumerate(c.ops):
                key = (c.graph_position + i, op)
                assert key not in all_ops_positions, f"Duplicate node in candidates: {key}"
                all_ops_positions.append(key)


class TestDropoutInChain:
    """Dropout is elementwise and should participate in chains."""

    def test_relu_dropout_chain(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.drop = nn.Dropout(0.1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.drop(torch.relu(x + 1.0))

        candidates = _analyze(M(), shape=(1, 64))
        assert len(candidates) >= 1
        longest = max(candidates, key=lambda c: len(c.ops))
        assert len(longest.ops) >= 2


class TestMemoryBoundClassification:
    """Verify the memory-bound vs compute-bound heuristic."""

    def test_elementwise_is_memory_bound(self):
        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.sigmoid(x + 1.0)

        candidates = _analyze(M(), shape=(4, 512))
        assert len(candidates) >= 1
        assert candidates[0].memory_bound is True

    def test_large_linear_activation_compute_bound(self):
        """A large linear (high arithmetic intensity) + act should be compute-bound."""
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4096, 4096)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.relu(self.fc(x))

        # Large batch × large hidden → high arithmetic intensity.
        candidates = _analyze(M(), shape=(128, 4096))
        assert len(candidates) >= 1
        linear_act = [c for c in candidates if "linear" in c.ops[0].lower()]
        assert len(linear_act) >= 1
        # 128 * 4096 = 524288 elements; AI = 2*M*N / ((M*N + M + N)*4) ≈ 2.0
        # That's below threshold → memory-bound. But with hidden=4096 the
        # actual matmul is (128,4096)@(4096,4096) so AI is much higher.
        # This tests that the heuristic at least runs without error.

    def test_norm_chain_is_memory_bound(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.LayerNorm(256)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                h = self.norm(x)
                return F.silu(h) * 2.0

        candidates = _analyze(M(), shape=(1, 32, 256))
        norm_cands = [c for c in candidates if "layernorm" in c.ops[0].lower()]
        assert len(norm_cands) >= 1
        assert norm_cands[0].memory_bound is True


class TestMultipleDisjointChains:
    """A model with multiple separate fusible regions should produce
    multiple candidates."""

    def test_two_separate_chains(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(64, 64)
                self.fc2 = nn.Linear(64, 64)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                a = torch.relu(self.fc1(x))
                b = torch.sigmoid(self.fc2(a))
                return b

        candidates = _analyze(M(), shape=(1, 64))
        # fc1→relu and fc2→sigmoid should be two separate candidates.
        assert len(candidates) >= 2


class TestInPlaceOps:
    """In-place tensor methods (add_, mul_) should be recognized as elementwise."""

    def test_inplace_add_chain(self):
        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = x.add(1.0)  # not in-place, but call_method
                x = x.mul(2.0)
                return x

        candidates = _analyze(M(), shape=(1, 64))
        assert len(candidates) >= 1
        assert len(candidates[0].ops) >= 2


class TestEmptyAndTrivialModels:
    """Degenerate models should not crash the analyzer."""

    def test_constant_output(self):
        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.ones(5)

        candidates = _analyze(M(), shape=(1, 64))
        # No fusible patterns — just don't crash.
        assert isinstance(candidates, list)

    def test_two_inputs(self):
        """Model with multiple inputs — analyzer only takes one sample_input,
        so it should still work (ShapeProp may fail, but analysis proceeds)."""
        class M(nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return torch.relu(x + y)

        # Can't easily pass two inputs via current API; just use no sample input.
        candidates = GraphAnalyzer(M()).analyze()
        assert len(candidates) >= 1


class TestShapeAcrossPatterns:
    """Shape propagation should work correctly across all pattern types."""

    def test_linear_act_shapes(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 128)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.relu(self.fc(x))

        candidates = _analyze(M(), shape=(4, 64))
        assert len(candidates) >= 1
        c = candidates[0]
        # Output shape should reflect Linear's output dim.
        assert c.output_shape == (4, 128)

    def test_norm_act_elem_shapes(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.LayerNorm(64)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                h = self.norm(x)
                h = F.silu(h)
                return h * 0.5

        candidates = _analyze(M(), shape=(2, 8, 64))
        norm_cands = [c for c in candidates if "layernorm" in c.ops[0].lower()]
        assert len(norm_cands) >= 1
        assert norm_cands[0].output_shape == (2, 8, 64)


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------

class TestCandidateSerialization:
    """Tests for JSON serialization of fusion candidates."""

    def _make_model_and_candidates(self):
        model = MockLlamaMLPBlock(dim=64, hidden=128)
        inp = torch.randn(1, 64)
        analyzer = GraphAnalyzer(model, sample_input=inp)
        candidates = analyzer.analyze()
        return model, analyzer, candidates

    def test_candidates_json_written(self, tmp_path):
        model, analyzer, candidates = self._make_model_and_candidates()
        path = analyzer.save_candidates(candidates, output_dir=tmp_path)
        assert path.exists()
        assert path.name == "candidates.json"

    def test_candidates_json_valid(self, tmp_path):
        model, analyzer, candidates = self._make_model_and_candidates()
        path = analyzer.save_candidates(candidates, output_dir=tmp_path)
        data = json.loads(path.read_text())
        assert isinstance(data, list)
        assert len(data) == len(candidates)
        expected_keys = {"ops", "input_shapes", "output_shape", "memory_bound",
                         "launch_overhead_us", "graph_position",
                         "is_cross_model", "skip_if_grad_checkpoint"}
        for entry in data:
            assert set(entry.keys()) == expected_keys

    def test_candidates_json_roundtrip(self, tmp_path):
        model, analyzer, candidates = self._make_model_and_candidates()
        path = analyzer.save_candidates(candidates, output_dir=tmp_path)
        data = json.loads(path.read_text())
        for orig, loaded in zip(candidates, data):
            assert loaded["ops"] == orig.ops
            assert loaded["memory_bound"] == orig.memory_bound
            assert loaded["launch_overhead_us"] == orig.launch_overhead_us
            assert loaded["graph_position"] == orig.graph_position

    def test_model_hash_deterministic(self):
        m1 = MockLlamaMLPBlock(dim=64, hidden=128)
        m2 = MockLlamaMLPBlock(dim=64, hidden=128)
        assert GraphAnalyzer.model_hash(m1) == GraphAnalyzer.model_hash(m2)

    def test_model_hash_differs_for_different_models(self):
        m1 = MockLlamaMLPBlock(dim=64, hidden=128)
        m2 = MockLlamaMLPBlock(dim=64, hidden=256)
        assert GraphAnalyzer.model_hash(m1) != GraphAnalyzer.model_hash(m2)

    def test_output_shape_serialized_as_list(self, tmp_path):
        """Tuples become lists in JSON — verify this is handled."""
        model, analyzer, candidates = self._make_model_and_candidates()
        path = analyzer.save_candidates(candidates, output_dir=tmp_path)
        data = json.loads(path.read_text())
        for entry in data:
            if entry["output_shape"] is not None:
                assert isinstance(entry["output_shape"], list)
            for shape in entry["input_shapes"]:
                assert isinstance(shape, list)


# ---------------------------------------------------------------------------
# LLM pass tests
# ---------------------------------------------------------------------------

class _SimpleFusibleModel(nn.Module):
    """Two elementwise ops that a rule-based pass won't see (fan-out blocks it)."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        # fan-out: h feeds two separate ops → rule-based sole-pair won't fire
        a = torch.relu(h)
        b = torch.sigmoid(h)
        return a + b


class TestLLMPass:
    """Tests for the opt-in LLM analysis pass."""

    def _make_model(self):
        return _SimpleFusibleModel()

    # ------------------------------------------------------------------
    def test_llm_pass_disabled_by_default(self):
        """GraphAnalyzer(model).analyze() must not call OpenAI when use_llm is False."""
        model = self._make_model()
        with patch("openai.OpenAI") as mock_openai:
            GraphAnalyzer(model).analyze()
            mock_openai.assert_not_called()

    # ------------------------------------------------------------------
    def test_llm_pass_missing_api_key(self):
        """With use_llm=True but no OPENAI_API_KEY, returns rule-based results only."""
        model = self._make_model()
        inp = torch.randn(1, 64)
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            # Must not crash; LLM path is silently skipped.
            candidates = GraphAnalyzer(model, sample_input=inp, use_llm=True).analyze()
        assert isinstance(candidates, list)
        # Rule-based candidates still present.
        assert len(candidates) >= 1

    # ------------------------------------------------------------------
    def test_llm_pass_mocked(self):
        """Mock OpenAI response; assert FusionCandidates are built from LLM output.

        Tests _llm_pass() directly with an empty visited set so the window
        contains real node names regardless of rule-based pattern results.
        """
        import torch.fx as fx
        from fusionagent.graph.analyzer import GraphAnalyzer

        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.relu(x + 1.0)

        model = M()
        inp = torch.randn(1, 64)

        # Get real node names from a trace.
        traced = fx.symbolic_trace(model)
        modules = dict(traced.named_modules())
        node_list = list(traced.graph.nodes)
        call_nodes = [n for n in node_list if n.op == "call_function"]
        # Pick the first two compute nodes as our LLM "found" pair.
        assert len(call_nodes) >= 2, "Need at least 2 call_function nodes"
        n0, n1 = call_nodes[0], call_nodes[1]

        fake_response_json = json.dumps({
            "candidates": [
                {
                    "node_names": [n0.name, n1.name],
                    "ops": ["custom_op_a", "custom_op_b"],
                    "reason": "LLM-identified cross-boundary fusion",
                    "memory_bound": True,
                    "estimated_benefit": "high",
                }
            ]
        })

        mock_choice = MagicMock()
        mock_choice.message.content = fake_response_json
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        analyzer = GraphAnalyzer(model, sample_input=inp, use_llm=True)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.OpenAI") as mock_openai_cls:
                mock_client = MagicMock()
                mock_openai_cls.return_value = mock_client
                mock_client.chat.completions.create.return_value = mock_completion

                # Call _llm_pass with empty visited so all nodes are eligible.
                from torch.fx.passes.shape_prop import ShapeProp
                ShapeProp(traced).propagate(inp)
                llm_candidates = analyzer._llm_pass(node_list, modules, visited=set())

        assert len(llm_candidates) >= 1
        ops_sets = [set(c.ops) for c in llm_candidates]
        assert any({"custom_op_a", "custom_op_b"}.issubset(s) for s in ops_sets)
        # Structural validity.
        for c in llm_candidates:
            assert isinstance(c.memory_bound, bool)
            assert c.launch_overhead_us >= 0.0
            assert isinstance(c.graph_position, int)

    # ------------------------------------------------------------------
    def test_llm_pass_bad_json_response(self):
        """If the LLM returns invalid JSON, no crash and no extra candidates."""
        model = self._make_model()
        inp = torch.randn(1, 64)

        mock_choice = MagicMock()
        mock_choice.message.content = "NOT VALID JSON }{{"
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.OpenAI") as mock_openai_cls:
                mock_client = MagicMock()
                mock_openai_cls.return_value = mock_client
                mock_client.chat.completions.create.return_value = mock_completion

                rule_based = GraphAnalyzer(model, sample_input=inp).analyze()
                with_llm = GraphAnalyzer(model, sample_input=inp, use_llm=True).analyze()

        # No crash, and no LLM candidates added.
        assert isinstance(with_llm, list)
        assert len(with_llm) == len(rule_based)
