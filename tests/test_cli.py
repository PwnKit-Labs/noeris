from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from tests import _pathfix  # noqa: F401

from research_engine.cli import _parse_operator_shape, _render_history_regressions_md, build_parser, main
from research_engine.llm import LlmConfigurationError
from research_engine.models import ResearchSource


class _FakeArxivProvider:
    def __init__(self, client: object, max_results: int) -> None:
        del client
        self.max_results = max_results

    def collect(self, topic) -> list[ResearchSource]:
        return [
            ResearchSource(
                identifier=f"arxiv://{topic.name}",
                kind="paper",
                title=f"{topic.name} paper",
                locator="https://example.com/paper",
                excerpt=f"max_results={self.max_results}",
            )
        ]


class _FakeGitHubProvider:
    def __init__(self, client: object, max_results: int) -> None:
        del client
        self.max_results = max_results

    def collect(self, topic) -> list[ResearchSource]:
        return [
            ResearchSource(
                identifier=f"github://{topic.name}",
                kind="repository",
                title=f"{topic.name} repo",
                locator="https://example.com/repo",
                excerpt=f"max_results={self.max_results}",
            )
        ]


@contextmanager
def _temp_workspace() -> Iterator[str]:
    with tempfile.TemporaryDirectory() as temp_dir:
        previous_cwd = Path.cwd()
        os.chdir(temp_dir)
        try:
            yield temp_dir
        finally:
            os.chdir(previous_cwd)


def _run_cli(*args: str) -> tuple[int, str]:
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(list(args))
    return exit_code, stdout.getvalue()


def _run_cli_json(*args: str) -> tuple[int, dict | list]:
    exit_code, output = _run_cli(*args)
    return exit_code, json.loads(output)


class CliTests(unittest.TestCase):
    def test_thesis_command_prints_product_thesis(self) -> None:
        exit_code, output = _run_cli("thesis")
        self.assertEqual(exit_code, 0)
        self.assertIn("autonomous R&D engine", output)

    def test_runs_command_returns_empty_list_for_fresh_workspace(self) -> None:
        with _temp_workspace():
            exit_code, payload = _run_cli_json("runs")
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload, [])

    def test_status_command_reports_empty_workspace(self) -> None:
        with _temp_workspace(), patch(
            "research_engine.cli._status_workflow_summary",
            return_value={"repo": "0sec-labs/noeris", "workflows": {}},
        ):
            exit_code, payload = _run_cli_json("status")
        self.assertEqual(exit_code, 0)
        self.assertIn("capabilities", payload)
        self.assertEqual(payload["workflow_summary"]["repo"], "0sec-labs/noeris")
        self.assertEqual(payload["run_count"], 0)
        self.assertEqual(payload["artifact_bundle_count"], 0)

    def test_run_command_persists_a_record(self) -> None:
        with _temp_workspace():
            exit_code, payload = _run_cli_json("run", "--topic", "long-context reasoning")
        self.assertEqual(exit_code, 0)
        self.assertIn("run_id", payload)
        self.assertTrue(payload["path"].endswith(".json"))
        self.assertEqual(payload["memo"]["topic"], "long-context reasoning")

    def test_show_run_command_returns_persisted_record(self) -> None:
        with _temp_workspace():
            exit_code, payload = _run_cli_json("run", "--topic", "tool use reliability")
            show_exit_code, shown = _run_cli_json("show-run", payload["run_id"])
        self.assertEqual(exit_code, 0)
        self.assertEqual(show_exit_code, 0)
        self.assertEqual(shown["run_id"], payload["run_id"])
        self.assertEqual(shown["memo"]["topic"], "tool use reliability")

    def test_history_command_summarizes_saved_runs(self) -> None:
        with _temp_workspace():
            first_exit_code, first_payload = _run_cli_json(
                "benchmark-run",
                "tool-use-reliability",
            )
            second_exit_code, second_payload = _run_cli_json(
                "benchmark-run",
                "tool-use-reliability",
            )
            history_exit_code, history_payload = _run_cli_json(
                "history",
                "--benchmark-id",
                "tool-use-reliability",
            )
        self.assertEqual(first_exit_code, 0)
        self.assertEqual(second_exit_code, 0)
        self.assertEqual(history_exit_code, 0)
        self.assertEqual(history_payload["benchmark_id"], "tool-use-reliability")
        self.assertEqual(history_payload["run_count"], 2)

    def test_history_brief_command_renders_markdown_summary(self) -> None:
        with _temp_workspace():
            _run_cli_json("benchmark-run", "tool-use-reliability")
            _run_cli_json("benchmark-run", "tool-use-reliability")
            exit_code, output = _run_cli(
                "history-brief",
                "--benchmark-id",
                "tool-use-reliability",
            )
        self.assertEqual(exit_code, 0)
        self.assertIn("# History Brief", output)
        self.assertIn("Benchmark", output)

    def test_export_history_command_writes_files(self) -> None:
        with _temp_workspace() as temp_dir:
            _run_cli_json("benchmark-run", "tool-use-reliability")
            _run_cli_json("benchmark-run", "tool-use-reliability")
            output_dir = Path(temp_dir) / "history"
            exit_code, payload = _run_cli_json(
                "export-history",
                "--benchmark-id",
                "tool-use-reliability",
                "--output-dir",
                str(output_dir),
            )
            summary_text = (output_dir / "history-summary.json").read_text(encoding="utf-8")
            brief_text = (output_dir / "history-brief.md").read_text(encoding="utf-8")
            regressions_json = (output_dir / "history-regressions.json").read_text(encoding="utf-8")
            regressions_md = (output_dir / "history-regressions.md").read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["output_dir"].endswith("history"))
        self.assertIn("tool-use-reliability", summary_text)
        self.assertIn("# History Brief", brief_text)
        self.assertIn("history-regressions.json", " ".join(payload["files"]))
        self.assertIn("history-regressions.md", " ".join(payload["files"]))
        self.assertIn("fp8_policy_regressions", regressions_json)
        self.assertIn("# History Regressions", regressions_md)

    def test_render_history_regressions_md_defaults_to_none(self) -> None:
        md = _render_history_regressions_md(
            {
                "benchmark_id": "matmul-speedup",
                "topic": "matrix multiplication speedup",
                "run_count": 2,
                "fp8_policy_regressions": [],
            }
        )
        self.assertIn("# History Regressions", md)
        self.assertIn("FP8 Policy Regressions", md)
        self.assertIn("- none", md)

    def test_export_run_command_writes_bundle(self) -> None:
        with _temp_workspace() as temp_dir:
            exit_code, payload = _run_cli_json("run", "--topic", "memory routing")
            output_dir = Path(temp_dir) / "exports"
            export_exit_code, exported = _run_cli_json(
                "export-run",
                payload["run_id"],
                "--output-dir",
                str(output_dir),
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(export_exit_code, 0)
        self.assertEqual(exported["run_id"], payload["run_id"])
        self.assertTrue(exported["bundle_dir"].endswith(payload["run_id"]))

    def test_status_command_reports_latest_run_and_artifacts(self) -> None:
        with _temp_workspace(), patch(
            "research_engine.cli._status_workflow_summary",
            return_value={
                "repo": "0sec-labs/noeris",
                "workflows": {"CI": {"queued": 1, "completed": 2, "latest_display_title": "Latest"}},
            },
        ):
            run_exit_code, payload = _run_cli_json("benchmark-run", "tool-use-reliability")
            _run_cli_json(
                "export-history",
                "--benchmark-id",
                "tool-use-reliability",
            )
            status_exit_code, status_payload = _run_cli_json("status")
        self.assertEqual(run_exit_code, 0)
        self.assertEqual(status_exit_code, 0)
        self.assertIn("capabilities", status_payload)
        self.assertGreaterEqual(status_payload["run_count"], 1)
        self.assertGreaterEqual(status_payload["artifact_bundle_count"], 1)
        self.assertEqual(status_payload["latest_run"]["run_id"], payload["run_id"])
        self.assertTrue(status_payload["history_artifacts_present"]["history_summary_json"])
        self.assertTrue(status_payload["history_artifacts_present"]["history_brief_md"])
        self.assertTrue(status_payload["history_artifacts_present"]["history_regressions_json"])
        self.assertTrue(status_payload["history_artifacts_present"]["history_regressions_md"])
        self.assertIn("CI", status_payload["workflow_summary"]["workflows"])

    def test_benchmark_run_command_persists_and_exports_empirical_lane(self) -> None:
        with _temp_workspace():
            exit_code, payload = _run_cli_json("benchmark-run", "tool-use-reliability")
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["benchmark_id"], "tool-use-reliability")
        self.assertTrue(payload["verification"]["passed"])
        self.assertIn(
            "tool-selection-summary.json",
            payload["memo"]["results"][0]["artifact_refs"],
        )
        self.assertTrue(payload["bundle_dir"].endswith(payload["run_id"]))

    def test_iterate_command_runs_multiple_benchmark_iterations(self) -> None:
        with _temp_workspace():
            exit_code, payload = _run_cli_json(
                "iterate",
                "matmul-speedup",
                "--iterations",
                "2",
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["benchmark_id"], "matmul-speedup")
        self.assertEqual(payload["iterations"], 2)
        self.assertEqual(len(payload["runs"]), 2)
        self.assertTrue(payload["best_run_id"])
        self.assertIn(payload["outcome"], {"new_baseline", "improved", "regressed", "plateaued"})
        self.assertIn("previous_frontier", payload)
        self.assertIn("best_frontier", payload)
        self.assertIn("frontier_delta", payload)
        self.assertIn("added_candidates", payload["frontier_delta"])
        self.assertIn("added_pareto_candidates", payload["frontier_delta"])
        self.assertIn("workload_changes", payload["frontier_delta"])

    def test_triton_iterate_parser_accepts_fused_norm_linear(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "triton-iterate",
            "--operator",
            "fused_norm_linear",
        ])
        self.assertEqual(args.command, "triton-iterate")
        self.assertEqual(args.operator, "fused_norm_linear")

    def test_triton_iterate_parser_accepts_attention_v2(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "triton-iterate",
            "--operator",
            "attention_v2",
        ])
        self.assertEqual(args.command, "triton-iterate")
        self.assertEqual(args.operator, "attention_v2")

    def test_parse_operator_shape_supports_fused_norm_linear(self) -> None:
        self.assertEqual(
            _parse_operator_shape("fused_norm_linear", "2048x2560x1536"),
            {"M": 2048, "N": 2560, "K": 1536},
        )

    def test_parse_operator_shape_supports_attention_v2_metadata(self) -> None:
        self.assertEqual(
            _parse_operator_shape(
                "attention_v2",
                "1x32x4096x512",
                {
                    "num_kv_heads": 4,
                    "is_causal": True,
                    "window_size": -1,
                    "use_qk_norm": True,
                    "shared_kv": False,
                },
            ),
            {
                "batch": 1,
                "heads": 32,
                "seq_len": 4096,
                "head_dim": 512,
                "num_kv_heads": 4,
                "is_causal": True,
                "window_size": -1,
                "use_qk_norm": True,
                "shared_kv": False,
            },
        )

    def test_sources_command_aggregates_provider_results(self) -> None:
        with (
            patch("research_engine.cli.UrllibHttpClient", return_value=object()),
            patch("research_engine.cli.ArxivAtomSourceProvider", _FakeArxivProvider),
            patch(
                "research_engine.cli.GitHubRepositorySourceProvider",
                _FakeGitHubProvider,
            ),
        ):
            exit_code, payload = _run_cli_json(
                "sources",
                "--topic",
                "long-context reasoning",
                "--max-results",
                "2",
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(payload), 2)
        self.assertEqual(payload[0]["kind"], "paper")
        self.assertEqual(payload[1]["kind"], "repository")

    def test_run_command_reports_llm_configuration_error(self) -> None:
        with patch(
            "research_engine.cli.build_pipeline",
            side_effect=LlmConfigurationError("missing provider"),
        ):
            exit_code, payload = _run_cli_json(
                "run",
                "--topic",
                "long-context reasoning",
                "--llm",
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(payload["error"], "missing provider")


if __name__ == "__main__":
    unittest.main()
