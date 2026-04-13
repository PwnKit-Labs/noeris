from __future__ import annotations

from ..models import ResearchAgendaItem


DEFAULT_RESEARCH_AGENDA: list[ResearchAgendaItem] = [
    ResearchAgendaItem(
        area_id="long-context",
        name="Long-Context Reasoning",
        category="llm",
        recommended_mode="post-training+evals",
        priority="high",
        why_it_matters=(
            "Long-context capability remains a visible failure mode and gives "
            "Noeris a practical evaluation-heavy wedge."
        ),
        benchmark_fit="strong",
        starter_questions=[
            "Which memory or retrieval intervention improves reasoning with fixed cost?",
            "What eval slices reveal genuine reasoning rather than retrieval shortcuts?",
        ],
    ),
    ResearchAgendaItem(
        area_id="tool-use",
        name="Tool-Use Reliability",
        category="agents",
        recommended_mode="post-training+systems+evals",
        priority="high",
        why_it_matters=(
            "Reliable tool use is central to agent products and is easier to "
            "measure than vague 'agenticness'."
        ),
        benchmark_fit="strong",
        starter_questions=[
            "Which planner/reviewer patterns reduce unforced tool errors?",
            "How should memory and recovery loops be evaluated?",
        ],
    ),
    ResearchAgendaItem(
        area_id="evals",
        name="Evaluation Design",
        category="research-infra",
        recommended_mode="evals",
        priority="high",
        why_it_matters=(
            "Weak evals let models appear to improve without actually improving."
        ),
        benchmark_fit="strong",
        starter_questions=[
            "Which benchmark slices are saturated or gameable?",
            "What artifact requirements should each benchmark lane enforce?",
        ],
    ),
    ResearchAgendaItem(
        area_id="post-training-data",
        name="Post-Training Data Selection",
        category="llm",
        recommended_mode="post-training",
        priority="high",
        why_it_matters=(
            "Data quality and targeting often matter more than broad scaling for "
            "small practical improvements."
        ),
        benchmark_fit="medium",
        starter_questions=[
            "Which synthetic or curated data slices move specific behaviors?",
            "How should candidate datasets be filtered and scored before training?",
        ],
    ),
    ResearchAgendaItem(
        area_id="memory-retrieval",
        name="Memory And Retrieval Structures",
        category="llm",
        recommended_mode="post-training+systems+evals",
        priority="high",
        why_it_matters=(
            "Many apparent reasoning failures are memory and context selection failures."
        ),
        benchmark_fit="strong",
        starter_questions=[
            "When does retrieval help or hurt compared with compression or routing?",
            "What memory architectures are robust under long-horizon tool use?",
        ],
    ),
    ResearchAgendaItem(
        area_id="inference-efficiency",
        name="Inference Efficiency",
        category="systems",
        recommended_mode="systems",
        priority="medium",
        why_it_matters=(
            "Lower latency and cost increase the practical ceiling of every other improvement."
        ),
        benchmark_fit="strong",
        starter_questions=[
            "Which decoding or routing tricks preserve quality while reducing cost?",
            "Which tasks benefit most from speculative or hierarchical inference?",
        ],
    ),
    ResearchAgendaItem(
        area_id="kernel-speed",
        name="Kernel And Matmul Speedups",
        category="systems",
        recommended_mode="systems",
        priority="medium",
        why_it_matters=(
            "Kernel-level speedups are concrete, measurable, and a strong proving ground "
            "for literature-to-execution research."
        ),
        benchmark_fit="strong",
        starter_questions=[
            "Which kernel techniques are underexplored on current hardware profiles?",
            "What baseline and artifact contract is sufficient to trust a reported speedup?",
        ],
    ),
    ResearchAgendaItem(
        area_id="failure-analysis",
        name="Failure Analysis And Error Taxonomy",
        category="research-infra",
        recommended_mode="evals",
        priority="medium",
        why_it_matters=(
            "Without a clear error taxonomy, the system cannot learn what to fix next."
        ),
        benchmark_fit="strong",
        starter_questions=[
            "Which benchmark failures are due to planning, memory, tool use, or eval design?",
            "How should error classes be persisted across runs?",
        ],
    ),
    ResearchAgendaItem(
        area_id="reasoning-data",
        name="Reasoning Data Generation",
        category="llm",
        recommended_mode="post-training",
        priority="medium",
        why_it_matters=(
            "Synthetic or targeted reasoning data can be cheaper to iterate on than "
            "broad retraining efforts."
        ),
        benchmark_fit="medium",
        starter_questions=[
            "What synthetic data formats transfer to real benchmark improvements?",
            "How can generated data be filtered for usefulness rather than volume?",
        ],
    ),
    ResearchAgendaItem(
        area_id="base-model-scaling",
        name="Base-Model Pretraining And Scaling",
        category="foundation-model",
        recommended_mode="pretraining",
        priority="low",
        why_it_matters=(
            "Important in the abstract, but too capital intensive for Noeris v1."
        ),
        benchmark_fit="weak",
        starter_questions=[
            "Which questions in this area can be approximated with smaller-scale studies first?",
            "What evidence would justify spending time here later?",
        ],
    ),
]
