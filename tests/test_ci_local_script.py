from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts/ci_local.sh"


class LocalCiScriptTests(unittest.TestCase):
    def test_dry_run_lists_ci_parity_commands(self) -> None:
        env = dict(os.environ)
        env["CI_LOCAL_DRY_RUN"] = "1"
        env["PYTHON_BIN"] = "python3"
        result = subprocess.run(
            [str(SCRIPT)],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("python3 -m pytest tests/ -x -q", result.stdout)
        self.assertIn("python3 scripts/check_public_claim_artifacts.py", result.stdout)
        self.assertIn("python3 -m research_engine.cli benchmark-run matmul-speedup", result.stdout)
        self.assertIn(
            "python3 -m research_engine.cli export-history --benchmark-id matmul-speedup --output-dir .noeris/history",
            result.stdout,
        )
        self.assertIn(
            "python3 scripts/check_history_regressions.py --path .noeris/history/history-regressions.json --summary-path .noeris/history/history-summary.json --benchmark-id matmul-speedup --fail-on-missing",
            result.stdout,
        )


if __name__ == "__main__":
    unittest.main()
