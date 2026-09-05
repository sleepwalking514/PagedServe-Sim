# LLM Serving Simulator

A C++ simulator for exploring a few practical LLM-serving decisions:

- contiguous and paged KV-cache allocation;
- `prefill-first` and `decode-first + chunked prefill` scheduling;
- block-level prefix caching;
- SLO goodput, TTFT, TPOT, and physical KV-cache use;
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
| `paged` | paged allocation | decode-first + chunked prefill | off |
| `prefix` | paged allocation | decode-first + chunked prefill | on |

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

- `workload.csv`: trace metadata with a fixed seed;
- `{scenario}_summary.csv`: one-row experiment summary;
- `{scenario}_requests.csv`: request-level latency and cache-hit data;
- `comparison.csv`: the three scenario comparison.

`saturated_demo.ini` uses a high arrival rate, a tight token budget, and a strict analytical TPOT SLO to make policy differences visible. `llama3_8b_like.ini` is a more moderate starting point. Update its latency profile and SLO thresholds before using it for hardware-specific conclusions.

To create a quick chart, install `matplotlib` in your Python environment and run:

```bash
python3 scripts/plot_results.py results/mixed/comparison.csv results/mixed/comparison.png
```

Common switches:

```text
--scenario all|baseline|paged|prefix
--workload short|mixed_burst|prefix_heavy
--requests N --seed N --output DIRECTORY
--decode-profile PROFILE.csv
--cuda-profile PROFILE.csv
```

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
./build/serving_sim --config configs/llama3_8b_like.ini --decode-profile results/decode_profile.csv --output results/calibrated
```

The simulator uses the nearest measured `(batch_size, sequence_length)` point. Record the hardware, CUDA version, model shape, warmup count, and benchmark settings with each result. The analytical fallback is not a hardware measurement.

`--cuda-validate` checks a small irregular block table against a CPU FP32 reference. For a fuller check against PyTorch, run the script below in a CUDA-enabled PyTorch environment.

For the full PyTorch comparison required for a kernel report, use a CUDA-enabled PyTorch environment:

```bash
python3 cuda/validate_pytorch.py
```

The script JIT-builds `cuda/paged_attention_torch.cu`, which includes the same kernel as the standalone benchmark, then compares FP16 output with a `torch.softmax` reference on an irregular physical block table.

## Model details and assumptions

- KV block bytes are `2 * layers * kv_heads * head_dim * block_tokens * dtype_bytes`.
- A paged request reserves logical blocks for its known output budget. Physical pages are created as tokens are processed, so the simulator can capture allocation and fragmentation without simulated OOM.
- Only complete prompt blocks enter the prefix cache. Appending tokens always allocates a new block; mutable shared tail blocks and Copy-on-Write are not modeled.
- Prefix cache entries are evicted by LRU only when no active request references the page.
- `prefill-first` is a simple baseline. The main policy schedules one decode token per active sequence, then spends the remaining token budget on bounded prompt chunks.
- Without `--decode-profile`, the simulator uses an analytical latency model so it can run on a CPU-only machine.

## Experiments to try

1. Run all scenarios on `mixed_burst`; compare SLO-goodput and TTFT/TPOT p95.
2. Run `prefix_heavy` and compare `paged` with `prefix`; inspect `prefix_hit_tokens`, cache evictions, and physical utilization.
3. Vary only `prefill_chunk_tokens` in the config. Plot the TTFT–TPOT trade-off.
4. Reduce `model.total_blocks` until pressure appears. Compare the contiguous baseline's allocation failures with paged KV-cache behavior.

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

The project does not include an RPC server, multi-GPU placement, swapping, speculative decoding, beam-search Copy-on-Write, or kernel autotuning. The goal is to keep the experiments focused on cache allocation, scheduling, and prefix reuse.
