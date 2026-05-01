from __future__ import annotations

import json
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


if __name__ == "__main__":
    unittest.main()
