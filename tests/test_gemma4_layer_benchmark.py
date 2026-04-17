from __future__ import annotations

import ast
import unittest

from tests import _pathfix  # noqa: F401

from research_engine.gemma4_layer_benchmark import (
    GEMMA4_LAYER_CONFIGS,
    generate_gemma4_layer_benchmark_script,
)


class Gemma4LayerBenchmarkTests(unittest.TestCase):
    """Tests for the Gemma 4 decoder layer benchmark generator."""

    def test_layer_configs_defined(self) -> None:
        """At least 3 layer configurations are defined."""
        self.assertGreaterEqual(len(GEMMA4_LAYER_CONFIGS), 3)
        names = [c["name"] for c in GEMMA4_LAYER_CONFIGS]
        self.assertIn("gemma4_31b_local", names)
        self.assertIn("gemma4_31b_global", names)
        self.assertIn("gemma4_e2b_local", names)

    def test_benchmark_script_compiles(self) -> None:
        """Generated script is valid Python (parses without SyntaxError)."""
        script = generate_gemma4_layer_benchmark_script()
        ast.parse(script)

    def test_benchmark_script_has_both_paths(self) -> None:
        """Generated script contains both noeris_fused and pytorch_separated paths."""
        script = generate_gemma4_layer_benchmark_script()
        self.assertIn("noeris_fused", script)
        self.assertIn("pytorch_separated", script)

    def test_benchmark_script_has_qk_norm_rope(self) -> None:
        """Generated script uses the fused QK-RMSNorm+RoPE kernel."""
        script = generate_gemma4_layer_benchmark_script()
        self.assertIn("qk_norm_rope", script)
        self.assertIn("noeris_qk_norm_rope", script)

    def test_benchmark_script_has_geglu(self) -> None:
        """Generated script uses the fused GeGLU kernel."""
        script = generate_gemma4_layer_benchmark_script()
        self.assertIn("geglu", script.lower())
        self.assertIn("noeris_geglu", script)

    def test_benchmark_script_has_layer_speedup(self) -> None:
        """Generated script reports the layer_speedup metric."""
        script = generate_gemma4_layer_benchmark_script()
        self.assertIn("layer_speedup", script)

    def test_benchmark_script_has_per_step_timing(self) -> None:
        """Generated script breaks down timing by step."""
        script = generate_gemma4_layer_benchmark_script()
        self.assertIn("step_times", script)
        self.assertIn("noeris_step_times", script)
        self.assertIn("pytorch_step_times", script)
        # Ensure all major steps are individually timed
        for key in [
            "1_2_pre_attn_norm_qkv_fused",
            "4_attention",
            "6_7a_pre_mlp_norm_gateup_fused",
            "7c_down_proj",
        ]:
            self.assertIn(key, script, f"Missing per-step key: {key}")

    def test_benchmark_script_uses_all_noeris_kernels(self) -> None:
        """Generated script wires all 4 Noeris kernels into the fused path."""
        script = generate_gemma4_layer_benchmark_script()
        self.assertIn("noeris_fused_norm_linear", script)
        self.assertIn("noeris_qk_norm_rope", script)
        self.assertIn("noeris_attention", script)
        self.assertIn("noeris_geglu", script)
        # Verify the attention wrapper calls flash_attn
        self.assertIn("_noeris_flash_attn_raw", script)

    def test_benchmark_script_imports_fused_norm_linear(self) -> None:
        script = generate_gemma4_layer_benchmark_script()
        self.assertIn("triton_fused_norm_matmul", script)
        self.assertIn("_noeris_fused_norm_linear_raw", script)

    def test_benchmark_script_t4_friendly_attn_config(self) -> None:
        """Benchmark script carries both local and global attention configs."""
        script = generate_gemma4_layer_benchmark_script()
        self.assertIn("attention_config_for_shape", script)
        self.assertIn('"BLOCK_M": 32', script)
        self.assertIn('"BLOCK_N": 64', script)
        self.assertIn('"BLOCK_M": 64', script)

    def test_benchmark_script_uses_retuned_31b_fused_norm_linear_config(self) -> None:
        script = generate_gemma4_layer_benchmark_script()
        self.assertIn('_FUSED_NORM_LINEAR_CONFIG_31B = {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "num_warps": 8, "num_stages": 3}', script)

    def test_benchmark_script_uses_retuned_global_attention_config(self) -> None:
        script = generate_gemma4_layer_benchmark_script()
        self.assertIn('_ATTN_CONFIG_GLOBAL = {"BLOCK_M": 32, "BLOCK_N": 32, "num_warps": 4, "num_stages": 2}', script)

    def test_benchmark_script_uses_retuned_geglu_config(self) -> None:
        script = generate_gemma4_layer_benchmark_script()
        self.assertIn('_GEGLU_CONFIG = {"BLOCK_SIZE": 128, "num_warps": 16, "num_stages": 1}', script)

    def test_benchmark_script_has_hardware_shape_policy_helpers(self) -> None:
        script = generate_gemma4_layer_benchmark_script()
        self.assertIn("def gpu_family_name():", script)
        self.assertIn("def fused_norm_linear_profile", script)
        self.assertIn("def attention_profile", script)
        self.assertIn("def geglu_profile", script)
        self.assertIn("_ATTN_CONFIG_POLICY", script)
        self.assertIn("_GEGLU_CONFIG_POLICY", script)
        self.assertIn("_FUSED_NORM_LINEAR_CONFIG_POLICY", script)

    def test_configs_have_required_fields(self) -> None:
        """Each config has all required layer dimension fields."""
        required = {
            "name", "batch", "seq_len", "hidden_dim", "num_heads",
            "num_kv_heads", "head_dim", "ffn_dim", "window_size", "is_causal",
        }
        for cfg in GEMMA4_LAYER_CONFIGS:
            missing = required - set(cfg.keys())
            self.assertEqual(
                missing, set(),
                f"Config {cfg.get('name', '?')} missing fields: {missing}",
            )

    def test_benchmark_script_has_correctness_check(self) -> None:
        """Generated script verifies correctness (allclose / max_err)."""
        script = generate_gemma4_layer_benchmark_script()
        self.assertIn("max_err", script)
        self.assertIn("correct", script)
        self.assertIn("step_max_errs", script)


if __name__ == "__main__":
    unittest.main()
