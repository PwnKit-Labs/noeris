# FP8 Layout Reuse Policy (H100)

Generated: 2026-04-24T14:50:32.623362+00:00

| Shape | reuse=1 | reuse=2 | reuse=4 | reuse=8 | reuse=16 |
|---|---|---|---|---|---|
| fp8_mm_1024 | kn | nk | nk | nk | nk |
| fp8_mm_2048x1024x2048 | kn | nk | nk | nk | nk |
| fp8_mm_4096x4096x4096 | nk | nk | nk | nk | nk |

Detailed effective latencies (ms):

- fp8_mm_1024 (kn=0.0359, nk=0.0257, prepack=0.0199)
  reuse=1: kn_total=0.0359, nk_total=0.0456 -> kn
  reuse=2: kn_total=0.0718, nk_total=0.0713 -> nk
  reuse=4: kn_total=0.1436, nk_total=0.1227 -> nk
  reuse=8: kn_total=0.2872, nk_total=0.2255 -> nk
  reuse=16: kn_total=0.5744, nk_total=0.4311 -> nk
- fp8_mm_2048x1024x2048 (kn=0.0504, nk=0.0332, prepack=0.0264)
  reuse=1: kn_total=0.0504, nk_total=0.0596 -> kn
  reuse=2: kn_total=0.1008, nk_total=0.0928 -> nk
  reuse=4: kn_total=0.2016, nk_total=0.1592 -> nk
  reuse=8: kn_total=0.4032, nk_total=0.292 -> nk
  reuse=16: kn_total=0.8064, nk_total=0.5576 -> nk
- fp8_mm_4096x4096x4096 (kn=0.4597, nk=0.1866, prepack=0.1257)
  reuse=1: kn_total=0.4597, nk_total=0.3123 -> nk
  reuse=2: kn_total=0.9194, nk_total=0.4989 -> nk
  reuse=4: kn_total=1.8388, nk_total=0.8721 -> nk
  reuse=8: kn_total=3.6776, nk_total=1.6185 -> nk
  reuse=16: kn_total=7.3552, nk_total=3.1113 -> nk
