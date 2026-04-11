# Ensemble vs Individual Selectors: matmul on A100

**Date:** 2026-04-11  
**Experiment:** 4-way ablation (baseline, cost_model, bandit, ensemble)  
**Operator:** matmul | **GPU:** A100 | **Iterations:** 5 | **Configs/run:** 6

## Results

| Condition    | Final TFLOPS | vs baseline |
|--------------|-------------|-------------|
| baseline     | 58.62       | —           |
| cost_model   | 85.24       | +45.39%     |
| bandit       | 137.51      | +134.56%    |
| ensemble     | 85.13       | +45.21%     |

## Interpretation

**Ensemble loses to bandit by a large margin.** The ensemble alternates between cost_model and bandit picks, but on matmul the two strategies diverge dramatically: bandit (+134.6%) vastly outperforms cost_model (+45.4%). The alternating strategy averages the two, landing squarely at cost_model performance (~+45%) rather than approaching the bandit optimum.

**Why ensemble fails here:**  
The ensemble's alternation assumes both underlying selectors are "directionally correct" — that each contributes useful signal. When the selectors *diverge strongly*, as on matmul, the ensemble wastes half its budget on cost_model proposals that occupy GPU time without discovering the high-TFLOPS regime the bandit has already found. In effect, the ensemble dilutes the bandit's superior exploration.

**Context from prior three-way comparison:**  
The three-way study (bandit, cost_model, baseline — no ensemble) confirmed bandit's +133% advantage on matmul. This four-way study adds the ensemble condition and shows it *does not close that gap*. Ensemble converges to cost_model performance (both ~85 TFLOPS), while bandit reaches ~137.5 TFLOPS.

**Cross-entropy contrast:**  
On cross_entropy, all three smart selectors tied at ~+37%, suggesting the cost_model and bandit find similar configs on that operator. The ensemble tie there reflects a case where strategies *agree*, not where they diverge. matmul is the harder test, and the ensemble fails it.

## Takeaway

When cost_model and bandit disagree significantly, the alternation strategy degrades to the weaker selector's performance level. The ensemble is not a safe default — it works when strategies converge but destroys value when they diverge. For matmul specifically, **bandit should be the default selector**.
