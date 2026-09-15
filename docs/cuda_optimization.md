# CUDA optimization notes

## Test system

- GPU: NVIDIA GeForce RTX 3060 Laptop GPU (compute capability 8.6)
- Driver: 560.94
- CUDA compiler: 12.4.99
- Host compiler: MSVC 19.44
- Nsight Compute: 2024.3.1
- PyTorch reference: 2.11.0+cu126 under WSL

The Windows binary was built from an x64 Visual Studio Developer Command Prompt:

```powershell
cmake -S . -B build-win-cuda -G "NMake Makefiles" `
  -DSERVING_BUILD_CUDA=ON `
  -DCMAKE_CUDA_ARCHITECTURES=86 `
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-win-cuda
```

If CMake does not locate `nvcc`, pass its full path with
`-DCMAKE_CUDA_COMPILER=<path-to-nvcc>` on the configure command.

## Change

The kernel originally used `expf` twice per token for its online softmax update. The optimized version uses CUDA's faster `__expf` intrinsic. Because it is an approximation, correctness was checked again after the change:

- standalone CPU FP32 reference: max absolute error `5.02e-5`, mean absolute error `7.59e-6`;
- PyTorch `torch.softmax` reference: the original tiny irregular case reports
  less than `1e-6` maximum absolute error; a later 32-Q/8-KV-head,
  1,024-token irregular check reports `1.5e-5`;
- validation tolerance: `3e-3` for the PyTorch test and `5e-3` for the standalone test.

Two other changes were tested and discarded because they were slower: computing the softmax state only in lane 0, and replacing the shared-memory query tile with a per-thread register array.

## Latency results

Each CSV point averages 100 measured kernel launches after 20 warmup launches. The table reports the median of six baseline runs and six optimized runs. Baseline and optimized executables were run in alternating order to reduce clock and temperature bias.

| Batch | Sequence | Baseline (ms) | `__expf` (ms) | Change |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 0.095257 | 0.094124 | +1.19% |
| 1 | 512 | 0.370748 | 0.366643 | +1.11% |
| 1 | 1024 | 1.153310 | 1.130235 | +2.00% |
| 4 | 128 | 0.097633 | 0.097741 | -0.11% |
| 4 | 512 | 0.379729 | 0.379877 | -0.04% |
| 4 | 1024 | 1.146465 | 1.141655 | +0.42% |
| 8 | 128 | 0.101478 | 0.102331 | -0.84% |
| 8 | 512 | 0.395740 | 0.397460 | -0.43% |
| 8 | 1024 | 1.145055 | 1.148295 | -0.28% |
| 16 | 128 | 0.207002 | 0.194448 | +6.06% |
| 16 | 512 | 0.811638 | 0.763350 | +5.95% |
| 16 | 1024 | 2.289635 | 2.287410 | +0.10% |
| 32 | 128 | 0.308971 | 0.297809 | +3.61% |
| 32 | 512 | 1.219380 | 1.173500 | +3.76% |
| 32 | 1024 | 3.460015 | 3.451375 | +0.25% |

The geometric-mean speedup over all 15 shapes is 1.016x (1.57%). Small negative changes are within roughly 1% and are treated as measurement noise rather than improvements.

## Nsight Compute comparison

Both reports captured the first batch-1, sequence-128 kernel launch with the Basic section set.

| Metric | Baseline | Optimized | Change |
| --- | ---: | ---: | ---: |
| Duration | 262.88 us | 253.73 us | -3.48% |
| Elapsed cycles | 214,837 | 207,358 | -3.48% |
| Registers per thread | 40 | 40 | unchanged |
| Achieved occupancy | 2.22% | 2.22% | unchanged |

Nsight reports only 0.07 waves per SM for this launch: 32 one-warp blocks are distributed across 30 SMs. The low occupancy is caused mainly by the small grid, not register or shared-memory pressure. A larger follow-up optimization would split a head across multiple warps or batch more work per launch; that requires a different reduction scheme and is outside this small change.

Example profiling command, run from an Administrator PowerShell:

```powershell
& "ncu.exe" `
  --set basic `
  --target-processes all `
  --kernel-name "regex:paged_attention_decode_kernel" `
  --launch-count 1 `
  --export ".\results\paged_attention_windows" `
  --force-overwrite `
  ".\build-win-cuda\serving_sim.exe" `
  --cuda-profile ".\results\decode_profile_windows.csv"
```

The generated profiles and Nsight reports live under `results/`, which is intentionally excluded from Git.
