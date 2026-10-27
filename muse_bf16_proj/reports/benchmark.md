# Official H200 benchmark

Job `86e5f6c3e21d41eb` (run). All 9 preflight tests and all 30 scored cases passed.

Overall: **1.2202×** over SGLang, 95% interval **[1.2019, 1.2385]**. Normalized step time: 3958.89 µs versus 4830.80 µs. This is a model-step score, not the latency of one GEMM.

Each row reports the official paired cold-L2 CUDA-graph measurement. The candidate time is the baseline median divided by the median paired speedup, as specified by the judge. The overall score aggregates weighted projection times within each of six loads, then takes the geometric mean of load speedups; it is not the geometric mean of these 30 individual ratios.

| Layer | M | N | K | SGLang µs | Candidate µs | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| qkv_gate | 1 | 2304 | 8192 | 13.95 | 11.08 | 1.260× |
| o | 1 | 8192 | 1024 | 6.90 | 6.24 | 1.106× |
| pooler | 1 | 4096 | 8192 | 21.63 | 19.11 | 1.132× |
| unpooler | 1 | 1024 | 4096 | 7.04 | 4.05 | 1.739× |
| router | 1 | 256 | 8192 | 10.13 | 4.77 | 2.125× |
| qkv_gate | 16 | 2304 | 8192 | 13.89 | 14.27 | 0.973× |
| o | 16 | 8192 | 1024 | 7.10 | 7.22 | 0.983× |
| pooler | 16 | 4096 | 8192 | 21.71 | 19.22 | 1.130× |
| unpooler | 16 | 8192 | 4096 | 18.43 | 18.03 | 1.022× |
| router | 16 | 256 | 8192 | 15.07 | 6.94 | 2.171× |
| qkv_gate | 32 | 2304 | 8192 | 14.32 | 14.41 | 0.994× |
| o | 32 | 8192 | 1024 | 6.91 | 7.15 | 0.967× |
| pooler | 32 | 4096 | 8192 | 22.21 | 19.53 | 1.137× |
| unpooler | 32 | 8192 | 4096 | 19.14 | 18.82 | 1.017× |
| router | 32 | 256 | 8192 | 15.63 | 7.51 | 2.080× |
| qkv_gate | 8 | 2304 | 8192 | 14.06 | 13.95 | 1.008× |
| o | 8 | 8192 | 1024 | 6.53 | 6.46 | 1.010× |
| pooler | 8 | 4096 | 8192 | 21.52 | 19.30 | 1.115× |
| unpooler | 8 | 1024 | 4096 | 6.82 | 6.50 | 1.048× |
| router | 8 | 256 | 8192 | 26.27 | 6.43 | 4.083× |
| qkv_gate | 128 | 2304 | 8192 | 17.47 | 16.08 | 1.087× |
| o | 128 | 8192 | 1024 | 7.42 | 7.23 | 1.027× |
| pooler | 128 | 4096 | 8192 | 24.77 | 21.37 | 1.159× |
| unpooler | 128 | 8192 | 4096 | 19.98 | 19.07 | 1.048× |
| router | 128 | 256 | 8192 | 16.38 | 8.43 | 1.943× |
| qkv_gate | 256 | 2304 | 8192 | 19.92 | 20.04 | 0.994× |
| o | 256 | 8192 | 1024 | 9.50 | 9.38 | 1.014× |
| pooler | 256 | 4096 | 8192 | 31.02 | 29.78 | 1.042× |
| unpooler | 256 | 8192 | 4096 | 24.16 | 23.82 | 1.014× |
| router | 256 | 256 | 8192 | 17.41 | 10.35 | 1.682× |

Artifact SHA-256: `9f87ee5344c4507e9161cbe3784972f8cdfd0f9c90ace6d692cdfdd5deffe269`. Full timings, samples, correctness messages, evaluator image digest and machine details are retained in the result JSON.
