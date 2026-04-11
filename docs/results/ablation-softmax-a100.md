# Cross-Run Learning Ablation Report

Operator: `softmax`
Hardware: `A100`
Iterations per condition: 5
95% threshold: 921.13

## Condition Comparison

| Condition | Final best | Iterations to 95% | Trajectory |
|---|---|---|---|
| **with_database** | 864.40 | 6 | [846.38, 864.4, 864.4, 864.4, 864.4] |
| **without_database** | 969.61 | 3 | [866.62, 868.47, 969.61, 969.61, 969.61] |

## Result: 0.50x speedup to convergence

Cross-run learning reaches 95% of the final best in 6 iterations vs 3 without.
