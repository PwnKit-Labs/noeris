# KV Quantize-on-Write Runtime Integration

Generated: 2026-04-21T06:02:01.280550+00:00

| GPU | Shape | separated ms | fused ms | auto ms | fused speedup | auto backend |
|---|---|---:|---:|---:|---:|---|
| A100 | b1_kv16_d256_t1 | 0.4352 | 0.1393 | 0.1372 | 3.1250x | fused |
| A100 | b1_kv4_d512_t1 | 0.4280 | 0.1372 | 0.1372 | 3.1194x | fused |
| A100 | b4_kv16_d256_t1 | 0.4321 | 0.1362 | 0.1352 | 3.1729x | fused |
| A100 | b1_kv16_d256_t8 | 0.4332 | 0.1352 | 0.1352 | 3.2045x | fused |
| H100 | b1_kv16_d256_t1 | 0.1876 | 0.0612 | 0.0631 | 3.0669x | fused |
| H100 | b1_kv4_d512_t1 | 0.1999 | 0.0650 | 0.0613 | 3.0753x | fused |
| H100 | b4_kv16_d256_t1 | 0.1824 | 0.0678 | 0.0669 | 2.6887x | fused |
| H100 | b1_kv16_d256_t8 | 0.1827 | 0.0604 | 0.0696 | 3.0254x | fused |

Auto backend should select fused in this environment (no external callback).
