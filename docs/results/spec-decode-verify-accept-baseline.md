# Spec Decode Verify+Accept Baseline

Generated: 2026-04-21T04:40:21.326847+00:00

| GPU | Shape | ms | tokens/ms | correct |
|---|---|---:|---:|---|
| A100 | draft4_vocab32k | 0.1556 | 25.6990 | true |
| A100 | draft8_vocab32k | 0.1413 | 56.6123 | true |
| A100 | draft16_vocab32k | 0.1536 | 104.1667 | true |
| A100 | draft16_vocab128k | 0.1423 | 112.4101 | true |
| A100 | draft32_vocab128k | 0.1413 | 226.4493 | true |
| H100 | draft4_vocab32k | 0.0859 | 46.5549 | true |
| H100 | draft8_vocab32k | 0.0794 | 100.7252 | true |
| H100 | draft16_vocab32k | 0.0760 | 210.5263 | true |
| H100 | draft16_vocab128k | 0.0789 | 202.7575 | true |
| H100 | draft32_vocab128k | 0.0838 | 381.9710 | true |

Baseline path: argmax(target logits) + draft compare + prefix-accept mask extraction.
