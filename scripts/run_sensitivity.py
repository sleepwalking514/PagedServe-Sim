#!/usr/bin/env python3
"""Run reproducible serving-policy sensitivity and attribution experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable


CORE_SCENARIOS = ("continuous_prefill", "paged_prefill", "continuous_decode", "paged_decode", "prefix")
SUMMARY_KEYS = (
    "realized_arrival_rate_per_second",
    "busy_time_ms",
    "busy_fraction",
    "completed_requests",
    "admitted_requests",
    "rejected_requests",
    "capacity_rejected_requests",
    "kv_growth_rejected_requests",
    "ttft_expired_requests",
    "slo_compliant_requests",
    "all_generated_tokens",
    "processed_prefill_tokens",
    "slo_goodput_requests_per_second",
    "throughput_tokens_per_second",
    "ttft_p95_ms",
    "itl_p95_ms",
    "mean_tpot_p95_ms",
    "max_itl_p95_ms",
    "max_queue_wait_ms",
    "average_block_utilization",
    "peak_internal_waste_ratio",
    "peak_external_fragmentation_ratio",
    "peak_allocated_blocks",
    "peak_active_requests",
    "total_prefix_hit_tokens",
    "prefix_hit_token_ratio",
    "cache_evictions",
)


def make_config(
    *,
    rate: float,
    tpot_slo: float,
    tpot_mode: str,
    requests: int,
    seed: int,
    workload: str,
    total_blocks: int = 6144,
    shared_prefix_tokens: int = 128,
) -> str:
    return f"""[model]
num_layers = 32
kv_heads = 8
head_dim = 128
block_tokens = 16
total_blocks = {total_blocks}
dtype_bytes = 2

[service]
token_budget = 128
prefill_chunk_tokens = 64
max_active_requests = 64
ttft_slo_ms = 1000
tpot_slo_ms = {tpot_slo}
tpot_slo_mode = {tpot_mode}

[workload]
kind = {workload}
request_count = {requests}
seed = {seed}
arrival_rate_per_second = {rate}
shared_prefix_tokens = {shared_prefix_tokens}
"""


def run_one(
    executable: Path,
    temporary_root: Path,
    *,
    label: str,
    scenario: str,
    rate: float,
    tpot_slo: float,
    tpot_mode: str,
    requests: int,
    seed: int,
    workload: str,
    total_blocks: int = 6144,
    shared_prefix_tokens: int = 128,
) -> dict[str, str]:
    work_dir = temporary_root / label
    work_dir.mkdir(parents=True, exist_ok=True)
    config = work_dir / "config.ini"
    output = work_dir / "output"
    config.write_text(
        make_config(
            rate=rate,
            tpot_slo=tpot_slo,
            tpot_mode=tpot_mode,
            requests=requests,
            seed=seed,
            workload=workload,
            total_blocks=total_blocks,
            shared_prefix_tokens=shared_prefix_tokens,
        ),
        encoding="utf-8",
    )
    command = [str(executable), "--config", str(config), "--scenario", scenario, "--output", str(output)]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        raise RuntimeError(
            f"{scenario} failed ({label}):\n{completed.stdout}\n{completed.stderr}"
        )
    with (output / f"{scenario}_summary.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    if row["tpot_slo_mode"] != tpot_mode:
        raise RuntimeError(f"TPOT SLO mode mismatch for {label}: {row['tpot_slo_mode']} != {tpot_mode}")
    return row


def number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def ratio(numerator: float, denominator: float) -> float | str:
    return numerator / denominator if denominator > 0.0 else "inf"


def select_metrics(row: dict[str, str]) -> dict[str, object]:
    selected: dict[str, object] = {}
    for key in SUMMARY_KEYS:
        value = number(row, key)
        selected[key] = int(value) if key.endswith("requests") or key.endswith("tokens") or key in {
            "peak_allocated_blocks", "peak_active_requests", "cache_evictions"
        } else value
    return selected


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def assert_trace_consistency(rows: Iterable[dict[str, str]], label: str) -> None:
    rates = {round(number(row, "realized_arrival_rate_per_second"), 6) for row in rows}
    if len(rates) != 1:
        raise RuntimeError(f"scenarios did not receive the same trace for {label}: {sorted(rates)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=Path("build/serving_sim"))
    parser.add_argument("--output", type=Path, default=Path("results/sensitivity"))
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--seeds", type=int, nargs="+", default=(7, 42, 123, 2026, 4096))
    args = parser.parse_args()

    executable = args.executable.resolve()
    if not executable.exists():
        raise SystemExit(f"Simulator executable not found: {executable}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    manifest = {
        "executable": str(executable),
        "request_count": args.requests,
        "seeds": args.seeds,
        "latency_model": "analytical",
        "notes": [
            "Load sweeps use mixed_poisson; mixed_burst is evaluated separately.",
            "mean and max TPOT/ITL SLO semantics are reported separately.",
            "All layout comparisons hold scheduler and prefix caching fixed.",
            "The trace's actual_output_tokens is completion ground truth only; runtime policies see max_new_tokens.",
            "Mixed workloads use a caller-visible 768-token generation cap; prefix-heavy uses 384.",
            "Paged admission reserves prompt KV only and rejects on decode growth exhaustion; contiguous reserves prompt plus max_new_tokens.",
            "Capacity sweeps use all selected seeds and report mid-decode rejection separately.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="pagedserve-sensitivity-") as temporary:
        temporary_root = Path(temporary)

        load_rows: list[dict[str, object]] = []
        for seed in args.seeds:
            for mode in ("mean", "max"):
                for rate in (24.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0):
                    scenario_rows: dict[str, dict[str, str]] = {}
                    for scenario in ("continuous_prefill", "paged_decode", "prefix"):
                        label = f"load-s{seed}-{mode}-{rate:g}-{scenario}"
                        scenario_rows[scenario] = run_one(
                            executable, temporary_root, label=label, scenario=scenario, rate=rate,
                            tpot_slo=0.2, tpot_mode=mode, requests=args.requests, seed=seed,
                            workload="mixed_poisson"
                        )
                    assert_trace_consistency(scenario_rows.values(), label)
                    baseline = number(scenario_rows["continuous_prefill"], "slo_goodput_requests_per_second")
                    for scenario, result in scenario_rows.items():
                        load_rows.append({
                            "seed": seed,
                            "workload": "mixed_poisson",
                            "configured_arrival_rate_per_second": rate,
                            "tpot_slo_mode": mode,
                            "tpot_slo_ms": 0.2,
                            "scenario": scenario,
                            **select_metrics(result),
                            "goodput_ratio_vs_continuous_prefill": ratio(
                                number(result, "slo_goodput_requests_per_second"), baseline
                            ),
                        })
        write_rows(output / "load_sweep.csv", load_rows)

        slo_rows: list[dict[str, object]] = []
        for mode in ("mean", "p95", "max"):
            for threshold in (0.1, 0.15, 0.2, 0.3, 0.5, 1.0):
                scenario_rows = {}
                for scenario in ("continuous_prefill", "paged_decode", "prefix"):
                    label = f"slo-{mode}-{threshold:g}-{scenario}"
                    scenario_rows[scenario] = run_one(
                        executable, temporary_root, label=label, scenario=scenario, rate=500.0,
                        tpot_slo=threshold, tpot_mode=mode, requests=args.requests, seed=42,
                        workload="mixed_poisson"
                    )
                assert_trace_consistency(scenario_rows.values(), label)
                baseline = number(scenario_rows["continuous_prefill"], "slo_goodput_requests_per_second")
                for scenario, result in scenario_rows.items():
                    slo_rows.append({
                        "seed": 42,
                        "workload": "mixed_poisson",
                        "configured_arrival_rate_per_second": 500.0,
                        "tpot_slo_mode": mode,
                        "tpot_slo_ms": threshold,
                        "scenario": scenario,
                        **select_metrics(result),
                        "goodput_ratio_vs_continuous_prefill": ratio(
                            number(result, "slo_goodput_requests_per_second"), baseline
                        ),
                    })
        write_rows(output / "slo_sweep.csv", slo_rows)

        ablation_rows: list[dict[str, object]] = []
        for seed in args.seeds:
            scenario_rows = {}
            for scenario in CORE_SCENARIOS:
                label = f"ablation-s{seed}-{scenario}"
                scenario_rows[scenario] = run_one(
                    executable, temporary_root, label=label, scenario=scenario, rate=500.0,
                    tpot_slo=0.2, tpot_mode="max", requests=args.requests, seed=seed,
                    workload="mixed_poisson"
                )
            assert_trace_consistency(scenario_rows.values(), label)
            baseline = number(scenario_rows["continuous_prefill"], "slo_goodput_requests_per_second")
            for scenario, result in scenario_rows.items():
                ablation_rows.append({
                    "seed": seed,
                    "scenario": scenario,
                    "configured_arrival_rate_per_second": 500.0,
                    "tpot_slo_mode": "max",
                    "tpot_slo_ms": 0.2,
                    **select_metrics(result),
                    "goodput_ratio_vs_continuous_prefill": ratio(
                        number(result, "slo_goodput_requests_per_second"), baseline
                    ),
                })
        write_rows(output / "policy_ablation.csv", ablation_rows)

        capacity_rows: list[dict[str, object]] = []
        for seed in args.seeds:
            for blocks in (6144, 4096, 3072, 2048, 1536, 1024, 768, 512):
                for scenario in ("continuous_decode", "paged_decode"):
                    label = f"capacity-s{seed}-{blocks}-{scenario}"
                    result = run_one(
                        executable, temporary_root, label=label, scenario=scenario, rate=500.0,
                        tpot_slo=1.0, tpot_mode="mean", requests=args.requests, seed=seed,
                        workload="mixed_poisson", total_blocks=blocks
                    )
                    capacity_rows.append({
                        "seed": seed,
                        "total_blocks": blocks,
                        "scenario": scenario,
                        **select_metrics(result),
                    })
        write_rows(output / "kv_capacity_sweep.csv", capacity_rows)

        prefix_rows: list[dict[str, object]] = []
        for rate, total_blocks in ((500.0, 6144), (1000.0, 6144), (5000.0, 6144),
                                   (1000.0, 16384), (5000.0, 16384)):
            for shared_prefix in (0, 64, 128, 256, 512):
                scenario_rows = {}
                for scenario in ("paged_decode", "prefix"):
                    label = f"prefix-r{rate:g}-b{total_blocks}-p{shared_prefix}-{scenario}"
                    scenario_rows[scenario] = run_one(
                        executable, temporary_root, label=label, scenario=scenario, rate=rate,
                        tpot_slo=1.0, tpot_mode="mean", requests=args.requests, seed=42,
                        workload="prefix_heavy", total_blocks=total_blocks,
                        shared_prefix_tokens=shared_prefix
                    )
                no_cache_goodput = number(scenario_rows["paged_decode"], "slo_goodput_requests_per_second")
                for scenario, result in scenario_rows.items():
                    prefix_rows.append({
                        "seed": 42,
                        "configured_arrival_rate_per_second": rate,
                        "total_blocks": total_blocks,
                        "shared_prefix_tokens": shared_prefix,
                        "scenario": scenario,
                        **select_metrics(result),
                        "goodput_ratio_vs_no_cache": ratio(
                            number(result, "slo_goodput_requests_per_second"), no_cache_goodput
                        ),
                    })
        write_rows(output / "prefix_length_sweep.csv", prefix_rows)

        arrival_rows: list[dict[str, object]] = []
        for workload in ("mixed_poisson", "mixed_burst"):
            for rate in (100.0, 500.0, 1000.0):
                label = f"arrival-{workload}-{rate:g}"
                result = run_one(
                    executable, temporary_root, label=label, scenario="paged_decode", rate=rate,
                    tpot_slo=1.0, tpot_mode="mean", requests=args.requests, seed=42,
                    workload=workload
                )
                arrival_rows.append({
                    "workload": workload,
                    "configured_arrival_rate_per_second": rate,
                    **select_metrics(result),
                    "realized_to_configured_ratio":
                        number(result, "realized_arrival_rate_per_second") / rate,
                })
        write_rows(output / "arrival_process_check.csv", arrival_rows)

    aggregate_rows: list[dict[str, object]] = []
    for mode in ("mean", "max"):
        for rate in (100.0, 500.0, 1000.0, 5000.0):
            for scenario in ("continuous_prefill", "paged_decode", "prefix"):
                matching = [
                    row for row in load_rows
                    if row["tpot_slo_mode"] == mode
                    and row["configured_arrival_rate_per_second"] == rate
                    and row["scenario"] == scenario
                ]
                goodputs = [float(row["slo_goodput_requests_per_second"]) for row in matching]
                throughputs = [float(row["throughput_tokens_per_second"]) for row in matching]
                aggregate_rows.append({
                    "tpot_slo_mode": mode,
                    "configured_arrival_rate_per_second": rate,
                    "scenario": scenario,
                    "seed_count": len(matching),
                    "goodput_mean_req_per_s": statistics.mean(goodputs),
                    "goodput_stdev_req_per_s": statistics.stdev(goodputs) if len(goodputs) > 1 else 0.0,
                    "throughput_mean_tok_per_s": statistics.mean(throughputs),
                    "throughput_stdev_tok_per_s": statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0,
                })
    write_rows(output / "multi_seed_summary.csv", aggregate_rows)

    for csv_path in output.glob("*.csv"):
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                for value in row.values():
                    if value and value.lower() not in {"inf", "nan"}:
                        parsed = float(value) if value.replace(".", "", 1).replace("-", "", 1).isdigit() else None
                        if parsed is not None and not math.isfinite(parsed):
                            raise RuntimeError(f"non-finite metric in {csv_path}: {value}")

    print(f"Wrote validated sensitivity results to {output}")


if __name__ == "__main__":
    main()
