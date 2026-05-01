from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts/check_history_regressions.py"


class CheckHistoryRegressionsScriptTests(unittest.TestCase):
    def test_missing_file_is_non_blocking_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "history-regressions.json"
            result = subprocess.run(
                ["python3", str(SCRIPT), "--path", str(target)],
                cwd=REPO,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "missing")
        self.assertEqual(payload["blocking_threshold_regressions"], [])

    def test_regressions_for_target_benchmark_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "history-regressions.json"
            target.write_text(
                json.dumps(
                    {
                        "benchmark_id": "matmul-speedup",
                        "fp8_policy_regressions": ["reuse_1_kn_rate dropped"],
                    }
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                ["python3", str(SCRIPT), "--path", str(target), "--benchmark-id", "matmul-speedup"],
                cwd=REPO,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 1)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "regressions_found")

    def test_other_benchmark_regressions_do_not_block(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "history-regressions.json"
            target.write_text(
                json.dumps(
                    {
                        "benchmark_id": "tool-use-reliability",
                        "fp8_policy_regressions": ["ignored"],
                    }
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                ["python3", str(SCRIPT), "--path", str(target), "--benchmark-id", "matmul-speedup"],
                cwd=REPO,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "ok")

    def test_threshold_drop_regression_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            regressions_path = Path(temp_dir) / "history-regressions.json"
            summary_path = Path(temp_dir) / "history-summary.json"
            regressions_path.write_text(
                json.dumps(
                    {
                        "benchmark_id": "matmul-speedup",
                        "fp8_policy_regressions": [],
                    }
                ),
                encoding="utf-8",
            )
            summary_path.write_text(
                json.dumps(
                    {
                        "benchmark_id": "matmul-speedup",
                        "fp8_latest_alignment": {"reuse_1_kn_rate": 0.7},
                        "fp8_previous_alignment": {"reuse_1_kn_rate": 1.0},
                    }
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    "--path",
                    str(regressions_path),
                    "--summary-path",
                    str(summary_path),
                    "--benchmark-id",
                    "matmul-speedup",
                    "--default-drop-threshold",
                    "0.2",
                ],
                cwd=REPO,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 1)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "regressions_found")
        self.assertTrue(payload["blocking_threshold_regressions"])

    def test_env_threshold_override_by_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            regressions_path = Path(temp_dir) / "history-regressions.json"
            summary_path = Path(temp_dir) / "history-summary.json"
            regressions_path.write_text(
                json.dumps(
                    {
                        "benchmark_id": "matmul-speedup",
                        "fp8_policy_regressions": [],
                    }
                ),
                encoding="utf-8",
            )
            summary_path.write_text(
                json.dumps(
                    {
                        "benchmark_id": "matmul-speedup",
                        "fp8_latest_alignment": {"reuse_1_kn_rate": 0.75},
                        "fp8_previous_alignment": {"reuse_1_kn_rate": 1.0},
                    }
                ),
                encoding="utf-8",
            )
            env = dict(os.environ)
            env["NOERIS_FP8_DROP_THRESHOLDS_JSON"] = json.dumps(
                {
                    "matmul-speedup": {"reuse_1_kn_rate": 0.3},
                }
            )
            result = subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    "--path",
                    str(regressions_path),
                    "--summary-path",
                    str(summary_path),
                    "--benchmark-id",
                    "matmul-speedup",
                    "--default-drop-threshold",
                    "0.2",
                ],
                cwd=REPO,
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
        self.assertEqual(result.returncode, 0)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "ok")


if __name__ == "__main__":
    unittest.main()
