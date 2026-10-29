# Official H200 benchmark

Job `c014ae47ef1e478b` (run). All 9 preflight tests and all 30 scored cases passed.

Overall: **1.2322×** over SGLang, 95% interval **[1.2138, 1.2507]**. Normalized step time: 3920.31 µs versus 4830.80 µs. This is a model-step score, not the latency of one GEMM.

Each row reports the official paired cold-L2 CUDA-graph measurement. The candidate time is the baseline median divided by the median paired speedup, as specified by the judge. The overall score aggregates weighted projection times within each of six loads, then takes the geometric mean of load speedups; it is not the geometric mean of these 30 individual ratios.

| Layer | M | N | K | SGLang µs | Candidate µs | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| qkv_gate | 1 | 2304 | 8192 | 14.22 | 9.87 | 1.442× |
| o | 1 | 8192 | 1024 | 6.82 | 5.98 | 1.140× |
| pooler | 1 | 4096 | 8192 | 21.38 | 19.34 | 1.105× |
| unpooler | 1 | 1024 | 4096 | 6.78 | 3.59 | 1.889× |
| router | 1 | 256 | 8192 | 10.24 | 3.56 | 2.878× |
| qkv_gate | 16 | 2304 | 8192 | 13.94 | 13.22 | 1.054× |
| o | 16 | 8192 | 1024 | 6.53 | 6.52 | 1.001× |
| pooler | 16 | 4096 | 8192 | 21.15 | 18.66 | 1.133× |
| unpooler | 16 | 8192 | 4096 | 18.46 | 18.05 | 1.023× |
| router | 16 | 256 | 8192 | 14.58 | 6.90 | 2.113× |
| qkv_gate | 32 | 2304 | 8192 | 14.35 | 14.28 | 1.005× |
| o | 32 | 8192 | 1024 | 6.62 | 6.90 | 0.960× |
| pooler | 32 | 4096 | 8192 | 22.08 | 19.43 | 1.137× |
| unpooler | 32 | 8192 | 4096 | 18.64 | 18.31 | 1.018× |
| router | 32 | 256 | 8192 | 15.23 | 7.25 | 2.100× |
| qkv_gate | 8 | 2304 | 8192 | 14.37 | 13.13 | 1.095× |
| o | 8 | 8192 | 1024 | 6.53 | 6.40 | 1.020× |
| pooler | 8 | 4096 | 8192 | 21.42 | 20.15 | 1.063× |
| unpooler | 8 | 1024 | 4096 | 6.88 | 6.03 | 1.141× |
| router | 8 | 256 | 8192 | 26.59 | 6.31 | 4.214× |
| qkv_gate | 128 | 2304 | 8192 | 16.93 | 15.65 | 1.082× |
| o | 128 | 8192 | 1024 | 7.50 | 7.45 | 1.008× |
| pooler | 128 | 4096 | 8192 | 25.06 | 21.82 | 1.148× |
| unpooler | 128 | 8192 | 4096 | 19.73 | 18.83 | 1.048× |
| router | 128 | 256 | 8192 | 16.14 | 8.83 | 1.829× |
| qkv_gate | 256 | 2304 | 8192 | 19.38 | 19.87 | 0.975× |
| o | 256 | 8192 | 1024 | 9.02 | 9.14 | 0.988× |
| pooler | 256 | 4096 | 8192 | 30.53 | 30.19 | 1.011× |
| unpooler | 256 | 8192 | 4096 | 24.14 | 24.65 | 0.980× |
| router | 256 | 256 | 8192 | 17.39 | 10.77 | 1.616× |

Artifact SHA-256: `67982fe239b900bdaac86823ee95e4af888968e38158d6a7565ae75cc5716686`. Full timings, samples, correctness messages, evaluator image digest and machine details are retained in the result JSON.
