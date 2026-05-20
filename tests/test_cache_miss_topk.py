"""
Cache miss topk correctness and performance benchmark.
Compares PyTorch reference vs AscendC implementation.
"""

import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch

import ascend_kernel


FIXED_TOPK = 2048
MAX_NUM_REQS = 128

DEFAULT_NUM_REQS_LIST = [1, 2, 4, 8, 16, 32, 64, 128]
DEFAULT_MAX_VALUES = [10000, 32768, 131072]
DEFAULT_HIT_RATES = [0.10, 0.30, 0.60, 0.70, 0.80, 0.90, 0.95]

KEY_NUM_REQS_FOR_BAR = [1, 2, 4, 8, 16, 32, 64, 128]

# Set this to the target card id, e.g. 1..15 on a 16-card machine.
NPU_DEVICE_ID = 15
DEVICE = f"npu:{NPU_DEVICE_ID}"

WARMUP_ITERS = 10
BENCH_ITERS = 100

OUTPUT_DIR = "output"
CORRECTNESS_CSV = f"{OUTPUT_DIR}/correctness_results.csv"
BENCHMARK_CSV = f"{OUTPUT_DIR}/benchmark_results.csv"
CHART_DIR = f"{OUTPUT_DIR}/charts"

RUN_CORRECTNESS = True
RUN_BENCHMARK = True
RUN_PLOT = True

STOP_ON_CORRECTNESS_FAIL = True

CORRECTNESS_FIELDS = [
    "num_reqs",
    "max_value",
    "hit_rate",
    "topk",
    "passed",
    "result_diff_count",
    "last_step_diff_count",
]
BENCHMARK_FIELDS = [
    "num_reqs",
    "max_value",
    "hit_rate",
    "topk",
    "pytorch_ms",
    "ascend_ms",
    "speedup",
    "warmup_iters",
    "bench_iters",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache miss topk correctness and performance benchmark"
    )
    parser.add_argument(
        "--num-reqs",
        default=None,
        help="Single value or comma-separated list, e.g. 32 or 1,8,32,128",
    )
    parser.add_argument(
        "--max-values",
        default=None,
        help="Single value or comma-separated list, e.g. 32768 or 10000,32768",
    )
    parser.add_argument(
        "--hit-rates",
        default=None,
        help="Single value or comma-separated list; supports 0.9 and 90%",
    )
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    return parser.parse_args()


def parse_int_list(value: Optional[str], default: List[int]) -> List[int]:
    if value is None:
        return list(default)

    result = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parsed = int(item)
        if parsed <= 0:
            raise ValueError(f"Integer values must be positive: {item}")
        result.append(parsed)

    if not result:
        raise ValueError("Expected at least one integer value")
    return result


def parse_hit_rate_list(value: Optional[str], default: List[float]) -> List[float]:
    if value is None:
        return list(default)

    result = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item.endswith("%"):
            parsed = float(item[:-1]) / 100.0
        else:
            parsed = float(item)
        if parsed < 0.0 or parsed > 1.0:
            raise ValueError(f"hit_rate must be in [0, 1] or written as percent: {item}")
        result.append(parsed)

    if not result:
        raise ValueError("Expected at least one hit_rate value")
    return result


def ensure_output_dirs() -> None:
    """Create output directories for CSV files and charts."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHART_DIR, exist_ok=True)


def append_csv_row(path: str, fieldnames: List[str], row: Dict[str, object]) -> None:
    """Append one row to a CSV file, writing the header when needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0

    with open(path, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        csv_file.flush()


def synchronize_device() -> None:
    """Synchronize configured accelerator device after measured work."""
    if DEVICE.startswith("npu"):
        if DEVICE == "npu":
            torch.npu.synchronize()
            return
        try:
            torch.npu.synchronize(DEVICE)
        except TypeError:
            torch.npu.synchronize()


def configure_device() -> None:
    """Set the configured NPU device when a concrete device id is provided."""
    if DEVICE.startswith("npu:") and hasattr(torch, "npu"):
        torch.npu.set_device(DEVICE)
        current_device = torch.npu.current_device()
        print(f"Using NPU device: {DEVICE}, current_device={current_device}")


def _format_hit_rate_for_label(hit_rate: float) -> str:
    return f"{hit_rate:.0%}"


def _same_hit_rate(left: float, right: float) -> bool:
    return abs(left - right) < 1e-9


def _validate_case_config(num_reqs: int, topk: int, max_value: int) -> None:
    if num_reqs > MAX_NUM_REQS:
        raise ValueError(
            f"num_reqs={num_reqs} exceeds MAX_NUM_REQS={MAX_NUM_REQS}"
        )
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    if max_value <= 0:
        raise ValueError(f"max_value must be positive, got {max_value}")


def generate_topk_pair_with_hit_rate(
    batch_size: int,
    topk: int,
    max_value: int,
    hit_rate: float,
    device: str,
    seed: Optional[int] = None,
    num_neg_ones: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate new topk and old cache state with a controlled hit rate."""
    _validate_case_config(batch_size, topk, max_value)
    if hit_rate < 0.0 or hit_rate > 1.0:
        raise ValueError(f"hit_rate must be in [0, 1], got {hit_rate}")
    if seed is not None:
        torch.manual_seed(seed)

    actual_topk = max(0, topk - num_neg_ones)
    hit_count = int(round(actual_topk * hit_rate))
    hit_count = max(0, min(hit_count, actual_topk))
    miss_count = actual_topk - hit_count
    required_unique = actual_topk + miss_count

    if required_unique > max_value:
        raise ValueError(
            "max_value is too small for requested hit_rate: "
            f"need at least {required_unique}, got {max_value}"
        )

    topk_indices = torch.full(
        (batch_size, topk), -1, dtype=torch.int32, device=device
    )
    last_step_topk_indices = torch.full(
        (MAX_NUM_REQS, topk), -1, dtype=torch.int64, device=device
    )

    for req_id in range(batch_size):
        perm = torch.randperm(max_value)
        raw_new_tokens = perm[:actual_topk].to(torch.int64)
        hit_tokens = raw_new_tokens[:hit_count]
        old_extra_tokens = perm[actual_topk:required_unique].to(torch.int64)
        raw_old_tokens = torch.cat((hit_tokens, old_extra_tokens))

        if actual_topk > 0:
            raw_new_tokens = raw_new_tokens[torch.randperm(actual_topk)]
            raw_old_tokens = raw_old_tokens[torch.randperm(actual_topk)]

        topk_indices[req_id, :actual_topk] = raw_new_tokens.to(
            device=device, dtype=torch.int32
        )

        req_id_offset = req_id * (1 << 16)
        old_tokens_with_offset = raw_old_tokens + req_id_offset
        last_step_topk_indices[req_id, :actual_topk] = old_tokens_with_offset.to(
            device=device, dtype=torch.int64
        )

    return topk_indices, last_step_topk_indices


def pytorch_reference(
    topk_indices: torch.Tensor,
    last_step_topk_indices: torch.Tensor,
    req_ids_tensor: torch.Tensor,
) -> torch.Tensor:
    """PyTorch reference implementation."""

    def get_set_diff_mask(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute set difference mask: elements in a but not in b."""
        assert a.shape == b.shape
        assert a.ndim == 2
        comparison_mask = a.unsqueeze(-1) == b.unsqueeze(1)
        intersect_mask = comparison_mask.any(-1)
        return ~intersect_mask

    num_reqs = topk_indices.shape[0]

    req_ids_offset = (req_ids_tensor[:num_reqs] * (1 << 16)).unsqueeze(-1)
    topk_indices_new = torch.where(topk_indices >= 0, topk_indices + req_ids_offset, -1)
    topk_indices_old = last_step_topk_indices[:num_reqs]

    cache_miss_token_mask = get_set_diff_mask(topk_indices_new, topk_indices_old)
    available_slot_mask = get_set_diff_mask(topk_indices_old, topk_indices_new)
    num_tokens_to_load = cache_miss_token_mask.sum(dim=1)
    num_available_slot = available_slot_mask.sum(dim=1)
    num_shortage_slot = num_tokens_to_load - num_available_slot

    num_shortage_slot = num_shortage_slot.unsqueeze(1)
    empty_slot_mask = topk_indices_old == -1
    empty_slot_cumsum = torch.cumsum(empty_slot_mask, dim=1)
    selected_empty_slot_mask = (empty_slot_cumsum <= num_shortage_slot) & empty_slot_mask
    available_slot_mask = torch.where(selected_empty_slot_mask, True, available_slot_mask)

    topk_indices_to_load_flattened = topk_indices_new[cache_miss_token_mask]
    topk_indices_new.fill_(-1)
    topk_indices_new[available_slot_mask] = topk_indices_to_load_flattened

    last_step_topk_indices[:num_reqs] = torch.where(
        available_slot_mask, topk_indices_new, last_step_topk_indices[:num_reqs]
    )

    topk_indices_new = torch.where(
        topk_indices_new >= 0, topk_indices_new - req_ids_offset, -1
    )

    return topk_indices_new.to(torch.int32)


def ascend_impl(
    topk_indices: torch.Tensor,
    last_step_topk_indices: torch.Tensor,
    req_ids_tensor: torch.Tensor,
) -> torch.Tensor:
    """AscendC implementation; topk_indices is updated in place."""
    torch.ops.npu.get_cache_miss_topk_indices(
        topk_indices, last_step_topk_indices, req_ids_tensor
    )
    return topk_indices


def _print_mismatch_examples(
    name: str,
    expected: torch.Tensor,
    actual: torch.Tensor,
    max_examples: int = 5,
) -> int:
    expected_cpu = expected.cpu()
    actual_cpu = actual.cpu()
    diff_mask = expected_cpu != actual_cpu
    diff_count = int(diff_mask.sum().item())

    if diff_count == 0:
        return 0

    print(f"  [{name} mismatch]")
    print(f"    diff_count={diff_count}")
    mismatch_positions = diff_mask.nonzero(as_tuple=False)[:max_examples]
    for pos in mismatch_positions:
        index = tuple(int(v) for v in pos.tolist())
        print(
            f"    pos={index}, pytorch={expected_cpu[index].item()}, "
            f"ascend={actual_cpu[index].item()}"
        )
    return diff_count


def test_correctness(
    num_reqs: int, topk: int, max_value: int, hit_rate: float
) -> bool:
    """Compare PyTorch and AscendC result tensors for one configuration."""
    topk_indices, last_step_topk_indices = generate_topk_pair_with_hit_rate(
        num_reqs,
        topk,
        max_value=max_value,
        hit_rate=hit_rate,
        device=DEVICE,
        seed=42,
    )
    req_ids_tensor = torch.arange(MAX_NUM_REQS, dtype=torch.int64, device=DEVICE)

    topk_indices_pt = topk_indices.clone()
    last_step_topk_indices_pt = last_step_topk_indices.clone()
    topk_indices_asc = topk_indices.clone()
    last_step_topk_indices_asc = last_step_topk_indices.clone()

    pytorch_result = pytorch_reference(
        topk_indices_pt, last_step_topk_indices_pt, req_ids_tensor
    )
    ascend_result = ascend_impl(topk_indices_asc, last_step_topk_indices_asc, req_ids_tensor)
    synchronize_device()

    result_match = torch.allclose(pytorch_result.cpu(), ascend_result.cpu(), rtol=0, atol=0)
    last_step_match = torch.allclose(
        last_step_topk_indices_pt.cpu(), last_step_topk_indices_asc.cpu(), rtol=0, atol=0
    )

    result_diff_count = 0
    last_step_diff_count = 0
    passed = result_match and last_step_match

    if not passed:
        print("\nMismatch detected!")
        print(f"  num_reqs={num_reqs}")
        print(f"  max_value={max_value}")
        print(f"  hit_rate={hit_rate:.2f}")
        result_diff_count = _print_mismatch_examples(
            "topk_indices", pytorch_result, ascend_result
        )
        last_step_diff_count = _print_mismatch_examples(
            "last_step_topk_indices",
            last_step_topk_indices_pt,
            last_step_topk_indices_asc,
        )
        print(f"  result_diff_count={result_diff_count}")
        print(f"  last_step_diff_count={last_step_diff_count}")

    append_csv_row(
        CORRECTNESS_CSV,
        CORRECTNESS_FIELDS,
        {
            "num_reqs": num_reqs,
            "max_value": max_value,
            "hit_rate": f"{hit_rate:.6g}",
            "topk": topk,
            "passed": passed,
            "result_diff_count": result_diff_count,
            "last_step_diff_count": last_step_diff_count,
        },
    )

    return passed


def run_correctness_matrix(
    num_reqs_list: List[int], max_values: List[int], hit_rates: List[float]
) -> bool:
    """Run the full correctness matrix and return True if every case passes."""
    if os.path.exists(CORRECTNESS_CSV):
        os.remove(CORRECTNESS_CSV)

    print("\n" + "=" * 60)
    print("Running Result Comparison Matrix")
    print("=" * 60)

    all_passed = True
    total_cases = len(max_values) * len(hit_rates) * len(num_reqs_list)
    case_idx = 0

    for max_value in max_values:
        for hit_rate in hit_rates:
            for num_reqs in num_reqs_list:
                case_idx += 1
                passed = test_correctness(num_reqs, FIXED_TOPK, max_value, hit_rate)
                all_passed = all_passed and passed
                status = "PASS" if passed else "FAIL"
                print(
                    f"[correctness {case_idx}/{total_cases}] "
                    f"max_value={max_value}, hit_rate={hit_rate:.2f}, "
                    f"num_reqs={num_reqs}, topk={FIXED_TOPK}: {status}"
                )
                if not passed and STOP_ON_CORRECTNESS_FAIL:
                    return False

    return all_passed


def _make_benchmark_inputs(
    num_reqs: int,
    topk: int,
    max_value: int,
    hit_rate: float,
    device: str,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    inputs = []
    for iter_idx in range(WARMUP_ITERS + BENCH_ITERS):
        num_neg_ones = max(0, 256 - iter_idx)
        inputs.append(
            generate_topk_pair_with_hit_rate(
                num_reqs,
                topk,
                max_value=max_value,
                hit_rate=hit_rate,
                device=device,
                seed=1000 + iter_idx,
                num_neg_ones=num_neg_ones,
            )
        )
    return inputs


def benchmark(
    num_reqs: int, topk: int, max_value: int, hit_rate: float, provider: str
) -> float:
    """Benchmark one provider for a single configuration."""
    req_ids_tensor = torch.arange(MAX_NUM_REQS, dtype=torch.int64, device=DEVICE)
    inputs = _make_benchmark_inputs(num_reqs, topk, max_value, hit_rate, DEVICE)
    provider_inputs = [(x.clone(), y.clone()) for x, y in inputs]
    synchronize_device()

    if provider == "ascend":
        impl = ascend_impl
    elif provider == "pytorch":
        impl = pytorch_reference
    else:
        raise ValueError(f"Unknown provider: {provider}")

    for iter_idx in range(WARMUP_ITERS):
        topk_indices, last_step_topk_indices = provider_inputs[iter_idx]
        impl(topk_indices, last_step_topk_indices, req_ids_tensor)
    synchronize_device()

    start_time = time.time()
    for iter_idx in range(BENCH_ITERS):
        input_idx = WARMUP_ITERS + iter_idx
        topk_indices, last_step_topk_indices = provider_inputs[input_idx]
        impl(topk_indices, last_step_topk_indices, req_ids_tensor)
    synchronize_device()

    return (time.time() - start_time) * 1000 / BENCH_ITERS


def run_benchmark_matrix(
    num_reqs_list: List[int], max_values: List[int], hit_rates: List[float]
) -> None:
    """Run the full benchmark matrix and append every result immediately."""
    if os.path.exists(BENCHMARK_CSV):
        os.remove(BENCHMARK_CSV)

    print("\n" + "=" * 60)
    print("Running Benchmark Matrix")
    print("=" * 60)

    total_cases = len(max_values) * len(hit_rates) * len(num_reqs_list)
    case_idx = 0

    for max_value in max_values:
        for hit_rate in hit_rates:
            for num_reqs in num_reqs_list:
                case_idx += 1
                pytorch_ms = benchmark(
                    num_reqs, FIXED_TOPK, max_value, hit_rate, "pytorch"
                )
                ascend_ms = benchmark(
                    num_reqs, FIXED_TOPK, max_value, hit_rate, "ascend"
                )
                speedup = pytorch_ms / ascend_ms if ascend_ms > 0 else float("inf")

                append_csv_row(
                    BENCHMARK_CSV,
                    BENCHMARK_FIELDS,
                    {
                        "num_reqs": num_reqs,
                        "max_value": max_value,
                        "hit_rate": f"{hit_rate:.6g}",
                        "topk": FIXED_TOPK,
                        "pytorch_ms": pytorch_ms,
                        "ascend_ms": ascend_ms,
                        "speedup": speedup,
                        "warmup_iters": WARMUP_ITERS,
                        "bench_iters": BENCH_ITERS,
                    },
                )

                print(
                    f"[benchmark {case_idx}/{total_cases}] "
                    f"max_value={max_value}, hit_rate={hit_rate:.2f}, "
                    f"num_reqs={num_reqs}, pytorch={pytorch_ms:.3f} ms, "
                    f"ascend={ascend_ms:.3f} ms, speedup={speedup:.2f}x"
                )


def load_benchmark_results() -> List[Dict[str, object]]:
    """Load benchmark CSV rows with numeric values converted."""
    if not os.path.exists(BENCHMARK_CSV):
        raise FileNotFoundError(f"Benchmark CSV not found: {BENCHMARK_CSV}")

    rows = []
    with open(BENCHMARK_CSV, newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            rows.append(
                {
                    "num_reqs": int(row["num_reqs"]),
                    "max_value": int(row["max_value"]),
                    "hit_rate": float(row["hit_rate"]),
                    "topk": int(row["topk"]),
                    "pytorch_ms": float(row["pytorch_ms"]),
                    "ascend_ms": float(row["ascend_ms"]),
                    "speedup": float(row["speedup"]),
                    "warmup_iters": int(row["warmup_iters"]),
                    "bench_iters": int(row["bench_iters"]),
                }
            )
    return rows


def _rows_for_max_value(
    rows: List[Dict[str, object]], max_value: int
) -> List[Dict[str, object]]:
    return sorted(
        (row for row in rows if row["max_value"] == max_value),
        key=lambda row: (row["hit_rate"], row["num_reqs"]),
    )


def _rows_for_max_value_and_hit_rate(
    rows: List[Dict[str, object]], max_value: int, hit_rate: float
) -> List[Dict[str, object]]:
    return sorted(
        (
            row
            for row in rows
            if row["max_value"] == max_value
            and _same_hit_rate(float(row["hit_rate"]), hit_rate)
        ),
        key=lambda row: row["num_reqs"],
    )


def _rows_for_max_value_and_num_reqs(
    rows: List[Dict[str, object]], max_value: int, num_reqs: int
) -> List[Dict[str, object]]:
    return sorted(
        (
            row
            for row in rows
            if row["max_value"] == max_value and row["num_reqs"] == num_reqs
        ),
        key=lambda row: row["hit_rate"],
    )


def plot_num_reqs_curves_by_hit_rate(
    rows: List[Dict[str, object]],
    max_values: List[int],
    hit_rates: List[float],
    metric: str,
    ylabel: str,
    filename_prefix: str,
) -> None:
    """Plot metric vs num_reqs, one chart per max_value, one line per hit_rate."""
    import matplotlib.pyplot as plt

    for max_value in max_values:
        plt.figure(figsize=(10, 6))
        for hit_rate in hit_rates:
            series = _rows_for_max_value_and_hit_rate(rows, max_value, hit_rate)
            if not series:
                continue
            plt.plot(
                [row["num_reqs"] for row in series],
                [row[metric] for row in series],
                marker="o",
                markersize=4,
                linewidth=1.5,
                label=f"hit_rate={_format_hit_rate_for_label(hit_rate)}",
            )

        plt.xlabel("num_reqs")
        plt.ylabel(ylabel)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(CHART_DIR, f"{filename_prefix}_max_value_{max_value}.png"),
            dpi=160,
        )
        plt.close()


def plot_speedup_vs_hit_rate(
    rows: List[Dict[str, object]], max_values: List[int]
) -> None:
    """Plot speedup vs hit_rate for selected num_reqs and each max_value."""
    import matplotlib.pyplot as plt

    num_reqs_for_hit_rate = [1, 8, 32, 128]
    for max_value in max_values:
        for num_reqs in num_reqs_for_hit_rate:
            series = _rows_for_max_value_and_num_reqs(rows, max_value, num_reqs)
            if not series:
                continue

            plt.figure(figsize=(8, 5))
            plt.plot(
                [row["hit_rate"] for row in series],
                [row["speedup"] for row in series],
                marker="o",
                linewidth=1.5,
            )
            plt.xlabel("hit_rate")
            plt.ylabel("speedup")
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    CHART_DIR,
                    f"speedup_vs_hit_rate_num_reqs_{num_reqs}_"
                    f"max_value_{max_value}.png",
                ),
                dpi=160,
            )
            plt.close()


def plot_results(max_values: List[int], hit_rates: List[float]) -> None:
    """Generate all benchmark charts from BENCHMARK_CSV."""
    print("\n" + "=" * 60)
    print("Generating Charts")
    print("=" * 60)

    rows = load_benchmark_results()
    plot_num_reqs_curves_by_hit_rate(
        rows,
        max_values,
        hit_rates,
        "speedup",
        "speedup",
        "speedup_vs_num_reqs_by_hit_rate",
    )
    plot_num_reqs_curves_by_hit_rate(
        rows,
        max_values,
        hit_rates,
        "ascend_ms",
        "AscendC latency_ms",
        "ascend_latency_vs_num_reqs_by_hit_rate",
    )
    plot_num_reqs_curves_by_hit_rate(
        rows,
        max_values,
        hit_rates,
        "pytorch_ms",
        "PyTorch latency_ms",
        "pytorch_latency_vs_num_reqs_by_hit_rate",
    )
    plot_speedup_vs_hit_rate(rows, max_values)

    print(f"Charts saved to: {CHART_DIR}")


def print_environment_info(
    num_reqs_list: List[int], max_values: List[int], hit_rates: List[float]
) -> None:
    print("=" * 60)
    print("Cache Miss TopK Correctness and Performance Benchmark")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"NPU available: {hasattr(torch, 'npu') and torch.npu.is_available()}")
    print(f"FIXED_TOPK: {FIXED_TOPK}")
    print(f"MAX_NUM_REQS: {MAX_NUM_REQS}")
    print(f"DEFAULT_NUM_REQS_LIST: {DEFAULT_NUM_REQS_LIST}")
    print(f"DEFAULT_MAX_VALUES: {DEFAULT_MAX_VALUES}")
    print(f"DEFAULT_HIT_RATES: {DEFAULT_HIT_RATES}")
    print(f"num_reqs_list: {num_reqs_list}")
    print(f"max_values: {max_values}")
    print(f"hit_rates: {hit_rates}")
    print(f"NPU_DEVICE_ID: {NPU_DEVICE_ID}")
    print(f"DEVICE: {DEVICE}")
    print(f"WARMUP_ITERS: {WARMUP_ITERS}")
    print(f"BENCH_ITERS: {BENCH_ITERS}")
    print("=" * 60)


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    configure_device()

    try:
        num_reqs_list = parse_int_list(args.num_reqs, DEFAULT_NUM_REQS_LIST)
        max_values = parse_int_list(args.max_values, DEFAULT_MAX_VALUES)
        hit_rates = parse_hit_rate_list(args.hit_rates, DEFAULT_HIT_RATES)
    except ValueError as exc:
        raise SystemExit(f"Argument error: {exc}") from exc

    run_correctness = RUN_CORRECTNESS and not args.skip_correctness
    run_benchmark = RUN_BENCHMARK and not args.skip_benchmark
    run_plot = RUN_PLOT and not args.skip_plot

    print_environment_info(num_reqs_list, max_values, hit_rates)

    correctness_passed = True
    if run_correctness:
        correctness_passed = run_correctness_matrix(
            num_reqs_list, max_values, hit_rates
        )
        if correctness_passed:
            print("All correctness cases passed.")
        else:
            print("Correctness failed.")

    if run_benchmark and (correctness_passed or not STOP_ON_CORRECTNESS_FAIL):
        run_benchmark_matrix(num_reqs_list, max_values, hit_rates)
    elif run_benchmark:
        print("Skipping benchmark because correctness failed.")

    if run_plot:
        plot_results(max_values, hit_rates)

    print("\nDone!")
    print(f"Correctness CSV: {CORRECTNESS_CSV}")
    print(f"Benchmark CSV: {BENCHMARK_CSV}")
    print(f"Charts: {CHART_DIR}")


if __name__ == "__main__":
    main()
