# Reproducible serving experiments

This document reports synthetic, analytical simulator experiments. None of the
serving goodput or millisecond SLO values is an end-to-end GPU-server
measurement. The optional CUDA measurements below concern one isolated decode
attention kernel.

## Reproduce

Build and test the simulator as described in the [README](../README.md), then
run:

```bash
python3 scripts/run_sensitivity.py \
  --executable build/serving_sim \
  --output results/sensitivity \
  --requests 500 \
  --seeds 7 42 123 2026 4096
```

The script generates `manifest.json`, `load_sweep.csv`,
`multi_seed_summary.csv`, `slo_sweep.csv`, `policy_ablation.csv`,
`kv_capacity_sweep.csv`, `prefix_length_sweep.csv`, and
`arrival_process_check.csv`. These generated files are ignored by Git. The
tables here give representative aggregates; use the script for all per-seed
rows and alternate settings.

The load and capacity comparisons use the same `mixed_poisson` trace within
each seed, 128 tokens of scheduling budget per iteration, 64-token bounded
prefill chunks, and up to 64 active requests. The analytical latency model is
not fitted to a complete transformer iteration.

## Request information and KV policy

The trace contains `actual_output_tokens` as completion ground truth, but the
runtime request visible to admission, scheduling, and KV allocation contains
only the prompt and `max_new_tokens`. The synthetic workload applies one
caller-visible cap per kind, independently of each realized output length:
96 tokens for short, 384 for prefix-heavy, and 768 for mixed workloads.

Contiguous allocation reserves a run for prompt plus `max_new_tokens` at
admission. Paged allocation reserves prompt blocks only and obtains a new KV
page at a decode block boundary. If growth cannot obtain a page, the request
is rejected after partial generation and its KV is released. This model does
not retry, preempt, recompute, or swap; `kv_growth_rejected_requests` must be
reported alongside goodput under memory pressure.

## Metrics and SLO semantics

Goodput is the count of completed requests meeting TTFT and the configured
inter-token SLO, divided by simulated wall time from first arrival to last
terminal request. `tpot_slo_mode=mean` checks each request's mean token gap;
`p95` and `max` check its p95 and worst gap. The latter two are stricter than
conventional per-request mean TPOT. `generated_tokens` counts completed
requests; `all_generated_tokens` also includes partial output from rejected
requests. `peak_internal_waste_ratio` is unused space inside allocated blocks,
while contiguous `peak_external_fragmentation_ratio` uses the largest free
physical run.

## Offered-load comparison

Five-seed mean goodput in requests/s with 6,144 KV blocks and a 0.2 ms
analytical token-gap threshold:

| Offered load | SLO mode | Contiguous + prefill-first | Paged + decode-first | Ratio of means |
| ---: | --- | ---: | ---: | ---: |
| 500 req/s | mean TPOT | 469.8 | 469.7 | 1.00x |
| 500 req/s | max ITL | 285.1 | 469.7 | 1.65x |
| 2,000 req/s | mean TPOT | 902.1 | 1,157.6 | 1.28x |
| 5,000 req/s | mean TPOT | 848.0 | 1,172.2 | 1.38x |

The combined comparison changes both layout and scheduling. A fixed-policy
ablation at 500 req/s with the strict max-ITL SLO produces 285.1 req/s for
both layouts under prefill-first and 469.7 req/s for both under decode-first;
at ample capacity, this difference is a scheduling-priority effect. At
5,000 req/s, completed output-token throughput is lower for decode-first
(about 238k versus 282k tokens/s). Goodput measures SLO compliance, not raw
compute speed.

Under the strict max-ITL threshold at 5,000 req/s, the prefill-first baseline
averages only 6/500 compliant requests while decode-first completes 500/500.
The resulting large ratio is dominated by a near-zero denominator and should
not be interpreted as hardware acceleration.

## KV capacity with scheduling held fixed

The following five-seed means use decode-first for both layouts, no prefix
cache, and a 1.0 ms mean-TPOT SLO:

| KV blocks | Layout | Goodput (req/s) | Completed / 500 | Mid-decode KV rejects | Maximum queue wait (ms) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1,536 | Contiguous | 469.6 | 500.0 | 0 | 14.5 |
| 1,536 | Paged | 469.7 | 500.0 | 0 | 0.1 |
| 1,024 | Contiguous | 463.1 | 500.0 | 0 | 40.8 |
| 1,024 | Paged | 469.3 | 499.6 | 0.4 | 1.5 |
| 768 | Contiguous | 441.1 | 500.0 | 0 | 112.1 |
| 768 | Paged | 466.1 | 496.2 | 3.8 | 4.4 |
| 512 | Contiguous | 345.6 | 500.0 | 0 | 413.4 |
| 512 | Paged | 449.1 | 476.6 | 23.4 | 12.4 |

At 512 blocks, paging yields 1.30x mean goodput but 4.68% of requests fail
mid-decode in the reject-on-exhaustion model. It is a queueing/completion-rate
trade-off, not an unconditional speedup. Because rejected requests are not
retried, the goodput metric alone cannot price their user-visible cost.

## Prefix-heavy workload

For a separate synthetic `prefix_heavy` trace at 5,000 req/s and 6,144 KV
blocks, 512 shared prompt tokens reduce processed prefill work from 253,438
to about 37,246 tokens (about 85%) and raise mean-SLO goodput 1.56x versus
the same paged/decode policy with prefix caching off. Ordinary mixed-Poisson
traces show no prefix-cache benefit at the evaluated operating point. Prefix
results are therefore workload-specific.

## CUDA kernel evidence and limits

On a GeForce RTX 3060 Laptop GPU, the standalone FP16 GQA PagedAttention
kernel passed an irregular-page CPU FP32 check with maximum absolute error
`5.02e-5`. A PyTorch `softmax` reference also passed for 32 query heads,
8 KV heads, head dimension 128, and an irregular 1,024-token context with
maximum absolute error `1.5e-5`; this check uses about 4.4 MiB of KV. The
validation is reproducible with `python3 cuda/validate_pytorch.py` in an
environment with CUDA-enabled PyTorch and `nvcc`.

Five 20-warmup/100-measurement event-timed runs of the isolated kernel gave
representative medians of 0.0945 ms (batch 1, context 128), 0.3798 ms
(batch 4, context 512), and 3.4739 ms (batch 32, context 1,024). The first
batch-1/128 run was 0.1300 ms, showing why repeated measurements matter.
The profile uses a small shared physical KV cache; it excludes transformer
projections, prefill, sampling, host scheduling, and client latency. See the
[CUDA optimization note](cuda_optimization.md) for the `__expf` comparison.

The 8B-like analytical config reserves 6,144 KV blocks of 2 MiB each
(12 GiB of KV alone), which cannot fit on a 6 GiB laptop GPU. Even supplying
`--decode-profile` replaces only an isolated decode-attention latency lookup;
prefill remains analytical. Hardware-feasible serving calibration requires a
model and KV budget that fit, full prefill/decode iteration timing, a measured
application SLO, and end-to-end validation on the same trace.
