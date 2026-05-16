"""Run the Noeris autonomous search loop on a free T4 GPU (Kaggle or Colab).

Primary platform: Kaggle (30 hr/week free T4, API-driven via `kaggle kernels push`).
Backup platform: Google Colab (~4-5 hr/day free T4).

This is the CORE of the Noeris system: generate configs → benchmark on GPU →
update the config database → learn → repeat. The bandit selector uses
Thompson sampling with Beta posteriors to explore the config space, and the
cost model (if available) filters candidates before GPU evaluation.

Usage (Kaggle or Colab):
  !git clone https://github.com/0sec-labs/noeris && cd noeris
  !pip install -e . numpy scikit-learn -q
  !python scripts/colab_iterate.py --operator qk_norm_rope --iterations 3 --configs-per-iter 8

Each iteration benchmarks `configs-per-iter` configurations across all shape
buckets for the chosen operator. Results are saved to a local config database
that persists across iterations (but not across Colab sessions unless you
download it).

No Modal needed. Free T4 GPU. This is how Noeris finds better configs.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import torch


# ---------------------------------------------------------------------------
# LLM proposer prompt template
# ---------------------------------------------------------------------------
PROPOSER_PROMPT = """You are a Triton GPU kernel tuning expert. Given the following benchmark results \
for the {operator} operator on {hardware}, propose {n_configs} novel configurations that might \
perform better than the current best.

Current database insights:
{insights_json}

Best configs found so far:
{best_configs}

The config parameters are: {param_space}

Propose {n_configs} new configs as JSON. Each must have all required keys. \
Think about: tile sizes that amortize memory access, warp counts that match \
the GPU's SM count, pipeline stages that hide latency.

Return ONLY a JSON array of config dicts, no explanation."""


def _detect_api_key() -> tuple[str, str] | None:
    """Return (provider, api_key) or None if nothing is available."""
    for env_var, provider in [
        ("ANTHROPIC_API_KEY", "anthropic"),
        ("AZURE_OPENAI_API_KEY", "azure"),
        ("OPENAI_API_KEY", "openai"),
    ]:
        key = os.environ.get(env_var)
        if key:
            return (provider, key)
    return None


def _propose_configs_anthropic(
    *,
    api_key: str,
    operator: str,
    hardware: str,
    insights: list[dict],
    best_configs: list[dict],
    param_space: dict,
    n_configs: int = 4,
) -> list[dict]:
    """Call the Anthropic Messages API to propose novel configs."""
    from urllib.request import Request, urlopen

    prompt = PROPOSER_PROMPT.format(
        operator=operator,
        hardware=hardware,
        n_configs=n_configs,
        insights_json=json.dumps(insights, indent=2),
        best_configs=json.dumps(best_configs, indent=2),
        param_space=json.dumps(param_space, indent=2),
    )
    payload = {
        "model": os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }
    request = Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urlopen(request, timeout=60) as response:
        body = json.loads(response.read().decode("utf-8"))

    text = ""
    for block in body.get("content", []):
        if block.get("type") == "text":
            text += block.get("text", "")

    # Extract JSON array from the response text
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end < 0:
        return []
    return json.loads(text[start : end + 1])


def _propose_configs_responses_api(
    *,
    operator: str,
    hardware: str,
    insights: list[dict],
    best_configs: list[dict],
    param_space: dict,
    n_configs: int = 4,
) -> list[dict]:
    """Use the existing Noeris ResponsesApiClient (OpenAI / Azure)."""
    from research_engine.llm import ResponsesApiClient

    client = ResponsesApiClient.from_environment()
    prompt = PROPOSER_PROMPT.format(
        operator=operator,
        hardware=hardware,
        n_configs=n_configs,
        insights_json=json.dumps(insights, indent=2),
        best_configs=json.dumps(best_configs, indent=2),
        param_space=json.dumps(param_space, indent=2),
    )
    # Use a minimal schema — just an array of config dicts
    param_props = {pname: {"type": "integer"} for pname in param_space.keys()}
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "configs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": param_props,
                    "required": list(param_props.keys()),
                },
            },
        },
        "required": ["configs"],
    }
    payload = client.generate_json(
        schema_name="colab_config_proposals",
        schema=schema,
        instructions="You propose Triton kernel configs. Return only valid configs.",
        prompt=prompt,
        max_output_tokens=1024,
        reasoning_effort="low",
        text_verbosity="low",
    )
    return [c for c in payload.get("configs", []) if isinstance(c, dict)]


def get_llm_proposed_configs(
    *,
    operator: str,
    hardware: str,
    db,
    spec,
    n_configs: int = 4,
    prefer_anthropic: bool = False,
) -> tuple[list[dict], str]:
    """Get LLM-proposed configs. Returns (configs, source_label).

    Falls back gracefully — never raises on missing keys.
    """
    cred = _detect_api_key()
    if cred is None:
        print("  WARNING: --llm requested but no API key found "
              "(ANTHROPIC_API_KEY, AZURE_OPENAI_API_KEY, OPENAI_API_KEY). "
              "Falling back to grid-only.")
        return [], "no_api_key"

    provider, api_key = cred

    # If --anthropic flag was passed, force Anthropic even if other keys exist
    if prefer_anthropic:
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_key:
            provider, api_key = "anthropic", anthropic_key
        else:
            print("  WARNING: --anthropic requested but ANTHROPIC_API_KEY not set. "
                  f"Using {provider} instead.")

    insights = db.get_insights(hardware=hardware, operator=operator) if hasattr(db, "get_insights") else []
    best_configs = []
    if insights:
        for ins in insights[:5]:
            bc = ins.get("best_config")
            if bc:
                best_configs.append(bc)

    param_space = spec.param_space if hasattr(spec, "param_space") else {}

    try:
        if provider == "anthropic":
            configs = _propose_configs_anthropic(
                api_key=api_key,
                operator=operator,
                hardware=hardware,
                insights=insights,
                best_configs=best_configs,
                param_space=param_space,
                n_configs=n_configs,
            )
        else:
            configs = _propose_configs_responses_api(
                operator=operator,
                hardware=hardware,
                insights=insights,
                best_configs=best_configs,
                param_space=param_space,
                n_configs=n_configs,
            )
        # Validate configs against param_space
        valid = []
        for cfg in configs:
            if not isinstance(cfg, dict):
                continue
            ok = True
            clean = {}
            for pname, allowed in param_space.items():
                val = cfg.get(pname)
                if not isinstance(val, int) or val not in allowed:
                    ok = False
                    break
                clean[pname] = val
            if ok:
                valid.append(clean)
        print(f"  LLM proposer ({provider}): {len(valid)} valid configs from {len(configs)} proposals")
        return valid[:n_configs], provider
    except Exception as exc:
        print(f"  WARNING: LLM proposer failed ({type(exc).__name__}: {str(exc)[:120]}). "
              "Falling back to grid-only.")
        return [], f"error_{provider}"


def run_iteration(
    operator: str,
    configs_per_iter: int,
    db_path: str,
    use_bandit: bool = True,
    shapes_mode: str = "standard",
    use_llm: bool = False,
    prefer_anthropic: bool = False,
) -> dict:
    """Run one iteration of the Noeris search loop on the local GPU."""
    from research_engine.triton_operators import REGISTRY
    from research_engine.triton_kernels import ConfigDatabase
    from research_engine.timing_snippet import install_noeris_timing

    spec = REGISTRY.get(operator)
    if spec is None:
        return {"error": f"Unknown operator: {operator}"}

    db = ConfigDatabase(path=db_path)
    hardware = torch.cuda.get_device_name(0)

    # Select shapes
    if shapes_mode == "tiny":
        shapes = spec.shape_buckets[:2]
    elif shapes_mode == "full":
        shapes = spec.shape_buckets
    else:
        shapes = spec.shape_buckets[:6]

    # LLM proposer — generate novel configs outside the grid
    proposed_configs: list[dict] = []
    llm_source = "none"
    if use_llm:
        proposed_configs, llm_source = get_llm_proposed_configs(
            operator=operator,
            hardware=hardware,
            db=db,
            spec=spec,
            n_configs=max(2, configs_per_iter // 2),
            prefer_anthropic=prefer_anthropic,
        )

    # Select configs — use bandit if available, else grid sample
    if use_bandit:
        try:
            from research_engine.bandit_selector import BanditSelector
            bandit = BanditSelector()
            configs = bandit.select_configs(
                spec=spec, database=db, hardware=hardware,
                shapes=shapes, max_configs=configs_per_iter,
                proposed_configs=proposed_configs,
            )
            selector = "bandit"
        except Exception as e:
            print(f"  Bandit failed ({e}), falling back to grid sample")
            configs = proposed_configs + spec.grid_generator_fn(max_configs=configs_per_iter)[:configs_per_iter]
            configs = configs[:configs_per_iter]
            selector = "grid+llm" if proposed_configs else "grid"
    else:
        configs = proposed_configs + spec.grid_generator_fn(max_configs=configs_per_iter)[:configs_per_iter]
        configs = configs[:configs_per_iter]
        selector = "grid+llm" if proposed_configs else "grid"

    print(f"  Selector: {selector}, {len(configs)} configs, {len(shapes)} shapes")

    # Generate and run the benchmark script
    script = spec.benchmark_script_fn(configs, shapes)
    script = install_noeris_timing(script, timer="cuda_event")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name

    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, script_path],
        capture_output=True, text=True, timeout=300,
    )
    elapsed = time.time() - t0

    if proc.returncode != 0:
        return {
            "error": proc.stderr[-500:] if proc.stderr else f"returncode={proc.returncode}",
            "elapsed": elapsed,
        }

    # Parse results
    stdout = proc.stdout
    start = stdout.find("{")
    if start < 0:
        return {"error": "No JSON in stdout", "elapsed": elapsed}

    payload = json.loads(stdout[start:])

    # Update the config database with results
    new_bests = 0
    total_correct = 0
    total_tested = 0
    best_per_shape = {}

    for cfg_result in payload.get("config_results", []):
        cid = cfg_result.get("config_id", "?")
        config = cfg_result.get("config", {})
        for r in cfg_result.get("results", []):
            total_tested += 1
            shape_name = r.get("shape_name", "?")
            correct = r.get("correct", False)
            metric = r.get("gb_per_s") or r.get("tflops") or 0
            ms = r.get("ms") or 0

            if correct:
                total_correct += 1

            # Record in database
            bucket = spec.shape_bucket_fn(
                next((s for s in spec.shape_buckets if s["name"] == shape_name), {})
            ) if spec.shape_bucket_fn else shape_name

            is_new_best = db.record_result(
                shape={"name": shape_name},
                hardware=hardware,
                config=config,
                tflops=float(metric),
                ms=float(ms),
                correct=correct,
                operator=operator,
                bucket=bucket,
                config_id_str=cid,
            )
            if is_new_best:
                new_bests += 1

            # Track best per shape for display
            if correct and (shape_name not in best_per_shape or metric > best_per_shape[shape_name]["metric"]):
                best_per_shape[shape_name] = {
                    "config_id": cid, "metric": metric, "ms": ms,
                    "fusion_speedup": r.get("fusion_speedup"),
                }

    # Save database
    db.save()

    return {
        "operator": operator,
        "hardware": hardware,
        "selector": selector,
        "configs_tested": len(configs),
        "shapes_tested": len(shapes),
        "total_measurements": total_tested,
        "correct": total_correct,
        "new_bests": new_bests,
        "elapsed": round(elapsed, 1),
        "best_per_shape": best_per_shape,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Noeris search loop on Colab T4")
    parser.add_argument("--operator", default="qk_norm_rope",
                       help="Which operator to search (default: qk_norm_rope)")
    parser.add_argument("--iterations", type=int, default=3,
                       help="Number of search iterations (default: 3)")
    parser.add_argument("--configs-per-iter", type=int, default=8,
                       help="Configs to test per iteration (default: 8)")
    parser.add_argument("--db-path", default=".noeris/colab-configs.json",
                       help="Path to config database (default: .noeris/colab-configs.json)")
    parser.add_argument("--shapes", default="standard", choices=["tiny", "standard", "full"],
                       help="Shape set (default: standard)")
    parser.add_argument("--no-bandit", action="store_true",
                       help="Use grid sampling instead of bandit")
    parser.add_argument("--llm", action="store_true",
                       help="Use LLM proposer to generate novel configs outside the grid")
    parser.add_argument("--anthropic", action="store_true",
                       help="Prefer Anthropic Claude API for the LLM proposer (requires ANTHROPIC_API_KEY)")
    parser.add_argument("--all-operators", action="store_true",
                       help="Run on ALL operators (1 iteration each)")
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    Path(args.db_path).parent.mkdir(parents=True, exist_ok=True)

    if args.all_operators:
        from research_engine.triton_operators import REGISTRY
        operators = sorted(REGISTRY.names())
    else:
        operators = [args.operator]

    for op in operators:
        print(f"\n{'='*60}")
        print(f"OPERATOR: {op}")
        print(f"{'='*60}")

        for i in range(args.iterations):
            print(f"\n--- Iteration {i+1}/{args.iterations} ---")
            result = run_iteration(
                operator=op,
                configs_per_iter=args.configs_per_iter,
                db_path=args.db_path,
                use_bandit=not args.no_bandit,
                shapes_mode=args.shapes,
                use_llm=args.llm or args.anthropic,
                prefer_anthropic=args.anthropic,
            )

            if "error" in result:
                print(f"  ERROR: {result['error'][:200]}")
                continue

            print(f"  {result['correct']}/{result['total_measurements']} correct, "
                  f"{result['new_bests']} new bests, {result['elapsed']}s")

            for shape, best in result.get("best_per_shape", {}).items():
                fs = f" fusion={best['fusion_speedup']:.2f}x" if best.get("fusion_speedup") else ""
                print(f"    {shape:30s} {best['metric']:>10.2f} GB/s  {best['config_id']}{fs}")

    # Final summary from DB
    from research_engine.triton_kernels import ConfigDatabase
    db = ConfigDatabase(path=args.db_path)
    insights = db.get_insights()
    if insights:
        print(f"\n{'='*60}")
        print(f"DATABASE INSIGHTS ({len(insights)} buckets)")
        print(f"{'='*60}")
        for ins in insights:
            print(f"  {ins.get('shape_bucket', '?'):40s} best={ins.get('best_tflops', 0):.2f} "
                  f"config={ins.get('best_config_id', '?')} "
                  f"experiments={ins.get('total_experiments', 0)}")


if __name__ == "__main__":
    main()
