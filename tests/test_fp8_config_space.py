from __future__ import annotations

import unittest

from tests import _pathfix  # noqa: F401

from research_engine.fp8_config_space import (
    build_fp8_grouped_gemm_config_space,
    build_fp8_matmul_config_space,
)


class Fp8ConfigSpaceTests(unittest.TestCase):
    def test_matmul_space_non_empty_and_has_splitk(self) -> None:
        rows = build_fp8_matmul_config_space()
        self.assertGreater(len(rows), 0)
        self.assertIn("SPLIT_K", rows[0])

    def test_grouped_gemm_space_non_empty_and_has_group_size(self) -> None:
        rows = build_fp8_grouped_gemm_config_space()
        self.assertGreater(len(rows), 0)
        self.assertIn("GROUP_SIZE_M", rows[0])


if __name__ == "__main__":
    unittest.main()
