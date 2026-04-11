# Research OS

Noeris should be framed as a **Research OS**.

That is more accurate than:

- an agent framework
- a deep research app
- a kernel optimizer

because it explains both the shared substrate and the domain-specific work built on top of it.

## Core Idea

The Research OS owns the common machinery for empirical discovery:

- source ingestion
- research memory
- hypothesis generation
- experiment planning
- execution
- artifact capture
- verification
- cross-run learning

Everything else is built on top of that substrate.

## Better Language Than "Programs"

Use:

- **Tracks** for major research directions
- **Lanes** for executable benchmark/evaluation paths
- **Studies** for focused investigations or ablations
- **Substrates** for shared infrastructure layers

This is better than `programs` because it sounds less like app modules and more like research operations.

## Recommended Vocabulary

### Substrates

Shared layers:

- ingestion substrate
- memory substrate
- execution substrate
- verification substrate
- artifact substrate

### Tracks

Persistent research directions:

- Triton kernel optimization track
- long-context reasoning track
- tool-use reliability track
- evaluation design track

### Lanes

Concrete executable paths:

- offline long-context lane
- offline tool-use lane
- offline matmul lane
- Modal A100 Triton iterate lane
- H100 KernelBench evaluation lane

### Studies

Bounded investigations:

- cross-run learning ablation study
- LayerNorm regression study
- AutoKernel comparison study

## Mapping The Current Repo

### Research OS substrate

Mostly core:

- `models.py`
- `components.py`
- `pipeline.py`
- `store.py`
- `export.py`
- `ingestion.py`
- `benchmarks.py`
- `agenda.py`
- `llm.py`

### Mixed / bridging layer

Connects substrate to executable lanes:

- `defaults.py`
- `executors.py`
- `cli.py`

### Triton / systems track

Clearly domain-specific:

- `triton_kernels.py`
- `triton_operators.py`
- `triton_rmsnorm.py`
- `triton_softmax.py`
- `triton_layernorm.py`
- `triton_cross_entropy.py`
- `triton_attention.py`
- `modal_runner.py`
- `kernelbench.py`
- `ablation.py`

## The Most Honest Reading Of The Repo Today

Noeris is currently:

- a Research OS substrate
- plus one very advanced flagship track: Triton/kernel optimization
- plus smaller benchmark lanes for long-context and tool use

So the repo is not "just Triton," but Triton is clearly the most mature track.

## Recommended Direction

Do **not** collapse Noeris into "the Triton project."

Do **not** pretend Triton is just one tiny plugin either.

The right stance is:

- Noeris = Research OS
- Triton = flagship track
- long-context/tool-use/evals = secondary tracks that preserve the broader ceiling

## Next Structural Move

The next refactor should make the layering more explicit without breaking velocity:

1. clarify docs using substrate/track/lane/study language
2. gradually separate core substrate from Triton-specific code
3. keep workflows aligned to tracks and lanes

