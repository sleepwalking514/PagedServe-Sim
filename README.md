# LLM Serving Simulator

A C++ simulator for exploring a few practical LLM-serving decisions:

- contiguous and paged KV-cache allocation;
- `prefill-first` and `decode-first` scheduling under the same bounded prefill chunking;
- block-level prefix caching;
- SLO goodput with explicit mean-TPOT, p95-ITL, or max-ITL semantics;
- TTFT, token-level ITL, request-level TPOT, realized arrival rate, and KV-cache use;
- an optional FP16 GQA PagedAttention CUDA benchmark for decode-latency calibration.

The simulator is single-threaded and event-driven. It models scheduling and resource usage; networking and RPC handling are outside its scope.

## Architecture

```text
workload generator -> deterministic CSV trace
                         |
                         v
                 serving simulator
                  |-- admission control
                  |-- KV-cache allocator
                  |-- prefill/decode scheduler
                  |-- prefix-cache LRU
                         |
                         v
              request CSV + comparison CSV

CUDA PagedAttention benchmark -> decode profile CSV -^
```

All three scenarios use the same generated trace and seed:

| Scenario | KV layout | Scheduler | Prefix cache |
| --- | --- | --- | --- |
| `baseline` | contiguous reservation | prefill-first | off |
| `paged` | paged allocation | decode-first | off |
| `prefix` | paged allocation | decode-first | on |

The ablation-only scenarios `continuous_prefill`, `paged_prefill`,
`continuous_decode`, and `paged_decode` allow layout and scheduling to be held
fixed independently. All policies use the configured `prefill_chunk_tokens`;
chunking is not unique to `decode-first`.

## Build

The core simulator needs a C++20 compiler and CMake 3.24+. CUDA is optional: if CMake cannot find `nvcc`, it builds the simulator and unit tests without the benchmark.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

A successful build produces `build/serving_sim` and `build/serving_tests`. If the same checkout was previously
configured with a different operating system, compiler, or CMake generator, remove `build/` and configure again.

## Run experiments

```bash
./build/serving_sim --config configs/saturated_demo.ini --scenario all --output results/mixed
```

The command writes:

- `workload.csv`: trace metadata with a fixed seed, including the caller-visible
  `max_new_tokens` and the simulator-only `actual_output_tokens`;
- `{scenario}_summary.csv`: one-row experiment summary;
- `{scenario}_requests.csv`: request-level latency and cache-hit data;
- `comparison.csv`: the three scenario comparison.

`saturated_demo.ini` uses a high arrival rate, a tight token budget, and an
explicit max-ITL SLO to expose decode stalls. `llama3_8b_like.ini` is a more
moderate mean-TPOT starting point. Both use the analytical model; calibrate
latency and choose application-specific SLOs before making hardware claims.

To create a quick chart, install `matplotlib` in your Python environment and run:

```bash
python3 scripts/plot_results.py results/mixed/comparison.csv results/mixed/comparison.png
```

To reproduce the experiment matrix:

```bash
python3 scripts/run_sensitivity.py \
  --executable build/serving_sim \
  --output results/sensitivity
```

The script writes:

- `arrival_process_check.csv`: configured versus realized arrival rate;
- `load_sweep.csv` and `multi_seed_summary.csv`: five-seed offered-load curves;
- `slo_sweep.csv`: mean-TPOT, p95-ITL, and max-ITL SLO sensitivity;
- `policy_ablation.csv`: layout, scheduler, and prefix-cache attribution;
- `kv_capacity_sweep.csv`: five-seed contiguous versus paged allocation at fixed
  scheduling, with admission and mid-decode KV-growth rejection counts;
- `prefix_length_sweep.csv`: prefix reuse under multiple loads and capacities;
- `manifest.json`: experiment assumptions and seeds.

The generated `results/` directory is intentionally ignored by Git. The
[experiment note](docs/experiments.md) contains representative aggregates;
run the script above to obtain all CSV rows on your checkout.

The clean load sweep uses `mixed_poisson`. `mixed_burst` is evaluated
separately with monotonic arrival timestamps and a mixture of short and normal
gaps whose expected rate matches the configured rate.

Common switches:

```text
--scenario all|baseline|paged|prefix|continuous_prefill|continuous_decode|paged_prefill|paged_decode
--workload short|mixed_poisson|mixed_burst|prefix_heavy
--requests N --seed N --output DIRECTORY
--decode-profile PROFILE.csv
--cuda-profile PROFILE.csv
```

## Metric definitions

- `realized_arrival_rate_per_second` is `(request_count - 1) / arrival_span`; it
  catches malformed or mislabeled traces.
- `simulated_time_ms` is wall-clock time from the first arrival through the last
  terminal request. `busy_time_ms` excludes idle jumps between arrivals.
- `mean_tpot_*` is a percentile across per-request mean inter-token latency;
  `itl_*` pools individual token gaps; `max_itl_*` is a percentile across each
  request's worst gap.
- `tpot_slo_mode = mean|p95|max` selects the per-request statistic used for
  SLO-goodput. A max-ITL SLO is intentionally much stricter than conventional
  mean TPOT and must be labeled as such.
- `peak_internal_waste_ratio` measures unused slots inside allocated blocks.
  `peak_external_fragmentation_ratio` measures
  `1 - largest_free_run / total_free_blocks` for contiguous allocation; paging
  has no contiguous-run requirement.

## Analytical experiment highlights

The following results use 500 requests, five seeds, the analytical latency
model, and `mixed_poisson`. They are simulator results, not GPU throughput
claims.

| Arrival rate | SLO mode at 0.2 ms | Contiguous + prefill-first | Paged + decode-first | Ratio of mean goodputs |
| ---: | --- | ---: | ---: | ---: |
| 500 req/s | mean TPOT | 469.8 req/s | 469.7 req/s | 1.00x |
| 500 req/s | max ITL | 285.1 req/s | 469.7 req/s | 1.65x |
| 2,000 req/s | mean TPOT | 902.1 req/s | 1,157.6 req/s | 1.28x |
| 5,000 req/s | mean TPOT | 848.0 req/s | 1,172.2 req/s | 1.38x |

The combined high-load comparison changes both layout and scheduler; it does
not measure a GPU compute speedup. A fixed-policy ablation at 500 req/s and
the strict max-ITL SLO attributes its goodput difference to scheduling at
ample KV capacity. Output-token throughput is lower for decode-first at
5,000 req/s (about 238k versus 282k tokens/s).

With decode-first fixed at 512 KV blocks, paged allocation averages 449.1
compliant req/s versus 345.6 for contiguous allocation, but 23.4/500 paged
requests fail on mid-decode KV growth. Goodput and failure rate must be read
together. On a separate saturated prefix-heavy workload, 512 shared prompt
tokens reduce processed prefill work by about 85% and improve modeled goodput
1.56x. See [the experiment note](docs/experiments.md) for setup, capacity
sweep, and limitations.

## CUDA calibration

When CUDA is available, CMake also builds `cuda/paged_attention.cu`. The kernel uses a paged `block_table`, FP16 KV cache, GQA head mapping, online softmax, and a shared-memory query tile. It is a benchmark kernel rather than a production replacement for FlashAttention or vLLM.

The CUDA code supports CUDA 12.x and keeps host-side validation numerics separate from device-only CUDA intrinsics.
Build the simulator target directly when iterating on the benchmark:

```bash
cmake --build build --target serving_sim -j
```

Generate a profile on the target GPU:

```bash
./build/serving_sim --cuda-profile results/decode_profile.csv
./build/serving_sim --cuda-validate
./build/serving_sim --config configs/llama3_8b_like.ini --decode-profile results/decode_profile.csv --output results/decode_kernel_assisted
```

The simulator uses the nearest measured `(batch_size, sequence_length)` point
for decode attention only. Prefill and other model/runtime work remain
analytical, so this is not a full serving calibration. Record the hardware,
CUDA version, model shape, warmup count, and benchmark settings with each
result. The 8B-like config's 12 GiB KV capacity is analytical and does not
fit on a 6 GiB laptop GPU.

`--cuda-validate` checks a small irregular block table against a CPU FP32
reference. For independent PyTorch comparison, use a CUDA-enabled PyTorch
environment with `nvcc` and run:

```bash
python3 cuda/validate_pytorch.py
```

The script JIT-builds `cuda/paged_attention_torch.cu`, which includes the same kernel as the standalone benchmark, then compares FP16 output with a `torch.softmax` reference on an irregular physical block table.
It also checks the 32-query-head/8-KV-head/128-dimensional GQA attention
shape at both 130-token and 1,024-token contexts, using at most 70 physical
KV blocks. This validates the kernel at relevant
head dimensions without loading an 8B model; it is not full serving
calibration. The [experiment note](docs/experiments.md#cuda-kernel-evidence-and-limits)
includes the measured errors and repeated kernel timings.

### Measured CUDA result

On an RTX 3060 Laptop GPU, replacing the online softmax `expf` calls with CUDA's `__expf` intrinsic delivered a 1.016x geometric-mean speedup across 15 tested batch/sequence shapes and up to 6.1% on an individual shape. Six baseline and optimized runs were interleaved; each reported point contains 20 warmup and 100 measured iterations. The CPU and PyTorch reference checks still pass.

Nsight Compute measured a 3.5% reduction in duration for the captured batch-1, sequence-128 launch. It also showed that this small launch uses only 0.1 full waves across the GPU, so increasing parallelism for small batches remains the larger optimization opportunity. See [`docs/cuda_optimization.md`](docs/cuda_optimization.md) for the setup, full results, and profiling commands.

## Model details and assumptions

- KV block bytes are `2 * layers * kv_heads * head_dim * block_tokens * dtype_bytes`.
- The trace keeps `actual_output_tokens` only as completion ground truth.
  Admission, scheduling, and KV allocation receive a different runtime type
  with `prompt_tokens` and `max_new_tokens`, not the final length.
- Contiguous allocation reserves one physical run for prompt plus
  `max_new_tokens` at admission. Paged allocation reserves prompt pages only,
  then allocates a new page at a decode block boundary. If KV growth cannot
  succeed, this simplified model rejects the partially generated request and
  releases its KV; it does not preempt, recompute, or swap.
- Synthetic requests use a workload-wide, caller-visible generation cap
  independent of each realized length: 96 tokens for short, 384 for
  prefix-heavy, and 768 for mixed workloads.
- Only complete prompt blocks enter the prefix cache. Appending tokens always allocates a new block; mutable shared tail blocks and Copy-on-Write are not modeled.
- Prefix cache entries are evicted by LRU only when no active request references the page.
- Both schedulers use bounded prompt chunks. `prefill-first` spends the token budget on chunks before decode; `decode-first` schedules one token per active decode sequence before using the remainder for prefill.
- Without `--decode-profile`, the simulator uses an analytical latency model so it can run on a CPU-only machine.

## Experiments to try

1. Re-run the five-seed matrix after changing the analytical model or workload distribution.
2. Vary only `prefill_chunk_tokens`; plot the TTFT versus mean/max-ITL trade-off.
3. Add preemption, recomputation, or swapping and a cap-sensitivity study before
   claiming paging's full production-serving benefit.
4. Calibrate both prefill and decode latency on target hardware before quoting real-time SLO values.

## Project layout

```text
include/serving/   Public types and simulator interfaces
src/               Simulator, cache models, workloads, metrics, CLI
cuda/              Optional PagedAttention decode benchmark
tests/             Dependency-free unit tests
configs/           Reproducible experiment settings
scripts/           CSV plotting helper
```

## Not modeled

The project does not include an RPC server, multi-GPU placement, swapping,
preemption, output-length prediction, speculative decoding, beam-search
Copy-on-Write, or kernel autotuning. The analytical prefill/decode equations are
not calibrated to the RTX 3060 kernel benchmark. The goal is to keep the
experiments focused on cache allocation, scheduling, and prefix reuse.
