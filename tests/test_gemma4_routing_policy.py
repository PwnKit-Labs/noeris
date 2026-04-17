from __future__ import annotations

import unittest

from tests import _pathfix  # noqa: F401

from research_engine.gemma4_routing_policy import (
    ATTN_CONFIG_GLOBAL,
    ATTN_CONFIG_LOCAL,
    FUSED_NORM_LINEAR_CONFIG_31B,
    FUSED_NORM_LINEAR_CONFIG_PREFILL,
    GEGLU_CONFIG,
    attention_config_for_shape,
    attention_profile,
    fused_norm_linear_config_for_shape,
    fused_norm_linear_profile,
    geglu_config_for_ffn_dim,
    geglu_profile,
    gpu_family_name,
)


class Gemma4RoutingPolicyTests(unittest.TestCase):
    def test_gpu_family_name(self) -> None:
        self.assertEqual(gpu_family_name("NVIDIA A100-SXM4-40GB"), "a100")
        self.assertEqual(gpu_family_name("NVIDIA H100 80GB HBM3"), "h100")
        self.assertEqual(gpu_family_name("NVIDIA L4"), "default")

    def test_fused_norm_linear_profile(self) -> None:
        self.assertEqual(fused_norm_linear_profile(2048, 16384, 5376), "31b_prefill")
        self.assertEqual(fused_norm_linear_profile(4096, 4608, 1536), "prefill")
        self.assertEqual(fused_norm_linear_profile(1, 16384, 5376), "decode")

    def test_attention_profile(self) -> None:
        self.assertEqual(attention_profile(512, -1), "31b_global")
        self.assertEqual(attention_profile(256, 1024), "local")

    def test_geglu_profile(self) -> None:
        self.assertEqual(geglu_profile(21504), "31b")
        self.assertEqual(geglu_profile(6144), "default")

    def test_fused_norm_linear_config_for_shape(self) -> None:
        self.assertEqual(
            fused_norm_linear_config_for_shape(2048, 16384, 5376, gpu_name="NVIDIA A100"),
            FUSED_NORM_LINEAR_CONFIG_31B,
        )
        self.assertEqual(
            fused_norm_linear_config_for_shape(4096, 4608, 1536, gpu_name="NVIDIA A100"),
            FUSED_NORM_LINEAR_CONFIG_PREFILL,
        )
        self.assertIsNone(
            fused_norm_linear_config_for_shape(1, 16384, 5376, gpu_name="NVIDIA A100")
        )

    def test_attention_config_for_shape(self) -> None:
        self.assertEqual(
            attention_config_for_shape(512, -1, gpu_name="NVIDIA A100"),
            ATTN_CONFIG_GLOBAL,
        )
        self.assertEqual(
            attention_config_for_shape(256, 1024, gpu_name="NVIDIA A100"),
            ATTN_CONFIG_LOCAL,
        )

    def test_geglu_config_for_ffn_dim(self) -> None:
        self.assertEqual(
            geglu_config_for_ffn_dim(21504, gpu_name="NVIDIA H100"),
            GEGLU_CONFIG,
        )
        self.assertEqual(
            geglu_config_for_ffn_dim(6144, gpu_name="NVIDIA H100"),
            GEGLU_CONFIG,
        )


if __name__ == "__main__":
    unittest.main()
