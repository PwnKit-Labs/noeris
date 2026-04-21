# Spec Decode Verify+Accept Runtime Integration

Generated: 2026-04-21T05:28:08.957646+00:00

| GPU | Shape | separated ms | fused ms | auto ms | fused speedup | auto backend |
|---|---|---:|---:|---:|---:|---|
| A100 | draft4_vocab32k | 0.1802 | 0.1290 | 0.1280 | 1.3968x | fused |
| A100 | draft8_vocab32k | 0.1782 | 0.1300 | 0.1290 | 1.3701x | fused |
| A100 | draft16_vocab32k | 0.2355 | 0.1597 | 0.1587 | 1.4744x | fused |
| A100 | draft16_vocab128k | 0.1782 | 0.1290 | 0.1300 | 1.3810x | fused |
| A100 | draft32_vocab128k | 0.2355 | 0.1597 | 0.1597 | 1.4744x | fused |
| H100 | draft4_vocab32k | 0.1215 | 0.0859 | 0.0824 | 1.4147x | fused |
| H100 | draft8_vocab32k | 0.1178 | 0.0796 | 0.0836 | 1.4803x | fused |
| H100 | draft16_vocab32k | 0.1192 | 0.0836 | 0.0827 | 1.4259x | fused |
| H100 | draft16_vocab128k | 0.1171 | 0.0857 | 0.0854 | 1.3663x | fused |
| H100 | draft32_vocab128k | 0.1186 | 0.0802 | 0.0824 | 1.4779x | fused |

Auto backend should select fused in this environment (no flashinfer callback).
