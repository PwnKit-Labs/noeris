# Spec Decode Verify+Accept Fused v1

Generated: 2026-04-21T05:15:40.819499+00:00

| GPU | Shape | best speedup | fused ms | separated ms | config |
|---|---|---:|---:|---:|---|
| A100 | draft16_vocab128k | 1.5839x | 0.1403 | 0.2222 | bs256_w4_s2 |
| A100 | draft16_vocab32k | 1.6963x | 0.1382 | 0.2345 | bs64_w2_s1 |
| A100 | draft32_vocab128k | 1.6364x | 0.1352 | 0.2212 | bs128_w2_s2 |
| A100 | draft4_vocab32k | 1.6906x | 0.1423 | 0.2406 | bs128_w2_s2 |
| A100 | draft8_vocab32k | 1.6074x | 0.1382 | 0.2222 | bs64_w2_s1 |
| H100 | draft16_vocab128k | 1.6027x | 0.0564 | 0.0904 | bs64_w2_s1 |
| H100 | draft16_vocab32k | 1.4208x | 0.0643 | 0.0913 | bs64_w2_s1 |
| H100 | draft32_vocab128k | 1.5820x | 0.0552 | 0.0873 | bs256_w4_s2 |
| H100 | draft4_vocab32k | 1.4217x | 0.0642 | 0.0913 | bs64_w2_s1 |
| H100 | draft8_vocab32k | 1.4108x | 0.0636 | 0.0898 | bs64_w2_s1 |

Fused path includes token match, first mismatch detection, and accepted-prefix mask write.
Argmax over logits remains outside this v1 kernel.
