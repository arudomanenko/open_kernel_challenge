# Official H200 benchmark

Job `cbbd6faa6ca74248` (check). All 9 preflight tests and all 30 scored cases passed.

Overall: **1.2250×** over SGLang, 95% interval **[1.2067, 1.2434]**. Normalized step time: 3943.41 µs versus 4830.80 µs. This is a model-step score, not the latency of one GEMM.

Each row reports the official paired cold-L2 CUDA-graph measurement. The candidate time is the baseline median divided by the median paired speedup, as specified by the judge. The overall score aggregates weighted projection times within each of six loads, then takes the geometric mean of load speedups; it is not the geometric mean of these 30 individual ratios.

| Layer | M | N | K | SGLang µs | Candidate µs | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| qkv_gate | 1 | 2304 | 8192 | 14.06 | 10.94 | 1.286× |
| o | 1 | 8192 | 1024 | 6.51 | 5.87 | 1.108× |
| pooler | 1 | 4096 | 8192 | 21.18 | 19.16 | 1.106× |
| unpooler | 1 | 1024 | 4096 | 7.04 | 3.99 | 1.765× |
| router | 1 | 256 | 8192 | 10.53 | 4.95 | 2.126× |
| qkv_gate | 16 | 2304 | 8192 | 14.13 | 14.37 | 0.983× |
| o | 16 | 8192 | 1024 | 6.58 | 6.68 | 0.984× |
| pooler | 16 | 4096 | 8192 | 21.25 | 18.91 | 1.124× |
| unpooler | 16 | 8192 | 4096 | 18.45 | 17.84 | 1.034× |
| router | 16 | 256 | 8192 | 14.78 | 6.70 | 2.208× |
| qkv_gate | 32 | 2304 | 8192 | 13.87 | 14.04 | 0.988× |
| o | 32 | 8192 | 1024 | 6.88 | 7.01 | 0.982× |
| pooler | 32 | 4096 | 8192 | 21.86 | 19.16 | 1.141× |
| unpooler | 32 | 8192 | 4096 | 18.85 | 18.30 | 1.030× |
| router | 32 | 256 | 8192 | 15.42 | 7.30 | 2.112× |
| qkv_gate | 8 | 2304 | 8192 | 14.16 | 13.79 | 1.027× |
| o | 8 | 8192 | 1024 | 7.06 | 6.93 | 1.019× |
| pooler | 8 | 4096 | 8192 | 21.34 | 19.12 | 1.116× |
| unpooler | 8 | 1024 | 4096 | 6.82 | 6.36 | 1.072× |
| router | 8 | 256 | 8192 | 26.02 | 6.36 | 4.092× |
| qkv_gate | 128 | 2304 | 8192 | 16.91 | 15.84 | 1.068× |
| o | 128 | 8192 | 1024 | 7.07 | 7.07 | 1.000× |
| pooler | 128 | 4096 | 8192 | 25.25 | 21.74 | 1.161× |
| unpooler | 128 | 8192 | 4096 | 20.00 | 18.76 | 1.066× |
| router | 128 | 256 | 8192 | 16.10 | 7.92 | 2.031× |
| qkv_gate | 256 | 2304 | 8192 | 19.74 | 19.84 | 0.995× |
| o | 256 | 8192 | 1024 | 9.82 | 9.63 | 1.020× |
| pooler | 256 | 4096 | 8192 | 30.59 | 29.42 | 1.040× |
| unpooler | 256 | 8192 | 4096 | 23.94 | 23.50 | 1.019× |
| router | 256 | 256 | 8192 | 16.83 | 9.81 | 1.716× |

Artifact SHA-256: `98c913426e8eaa9c81cac7b3daa18be19d303185d5cd52f89860400810f66d4b`. Full timings, samples, correctness messages, evaluator image digest and machine details are retained in the result JSON.
