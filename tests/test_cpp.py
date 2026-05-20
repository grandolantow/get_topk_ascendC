import argparse
import csv
import itertools
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, List


# Runtime modules are imported after ASCEND_RT_VISIBLE_DEVICES is configured.
torch = None
torch_npu = None
load = None


DEFAULT_NPU_ID = 15
DEFAULT_NUM_REQS = 4
DEFAULT_THREAD_NUM = 16
DEFAULT_SEQ_LEN = 32768
DEFAULT_TOPK = 2048
DEFAULT_PAGE_SIZE = 128
DEFAULT_LOAD_RATIO = 0.5
DEFAULT_CONTIGUITY = 0.0
DEFAULT_REPEAT = 100
DEFAULT_WARMUP = 5
DEFAULT_SEED = 0
DEFAULT_CSV = "output_compact/benchmark_results.csv"

NOPE_DIM = 512
PE_DIM = 64
HEAD_KV = 1


@dataclass(frozen=True)
class ExperimentConfig:
    npu_id: int
    num_reqs: int
    thread_num: int
    seq_len: int
    topk: int
    page_size: int
    load_ratio: float
    contiguity: float
    repeat: int
    warmup: int
    seed: int

    @property
    def max_num_pages(self) -> int:
        return math.ceil(self.seq_len / self.page_size)


def parse_list(value: object, cast_fn: Callable[[str], object]) -> List[object]:
    return [cast_fn(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parameterized CPU sparse attention get_kv_topk benchmark"
    )
    parser.add_argument("--npu-id", type=int, default=DEFAULT_NPU_ID)
    parser.add_argument("--num-reqs", default=str(DEFAULT_NUM_REQS))
    parser.add_argument("--thread-num", default=str(DEFAULT_THREAD_NUM))
    parser.add_argument("--seq-len", default=str(DEFAULT_SEQ_LEN))
    parser.add_argument("--topk", default=str(DEFAULT_TOPK))
    parser.add_argument("--page-size", default=str(DEFAULT_PAGE_SIZE))
    parser.add_argument("--load-ratio", default=str(DEFAULT_LOAD_RATIO))
    parser.add_argument("--contiguity", default=str(DEFAULT_CONTIGUITY))
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    return parser.parse_args()


def initialize_runtime(npu_id: int) -> None:
    global torch
    global torch_npu
    global load

    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(npu_id)
    os.environ["TORCH_EXTENSIONS_ALWAYS_BUILD"] = "1"
    os.environ["CXX"] = "clang++"
    os.environ["CC"] = "clang"

    import torch as torch_module
    import torch_npu as torch_npu_module
    from torch.utils.cpp_extension import load as cpp_load

    torch = torch_module
    torch_npu = torch_npu_module
    load = cpp_load

    torch.npu.set_device("npu:0")
    print(
        "Using NPU: "
        f"physical={npu_id}, visible={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')}, "
        f"logical=npu:0, current_device={torch.npu.current_device()}"
    )


def load_sparse_attn_extension():
    ascend_home = os.environ.get(
        "ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"
    )
    npu_include_path = os.path.join(ascend_home, "include")
    npu_lib_path = os.path.join(ascend_home, "lib64")
    if not os.path.exists(npu_lib_path):
        npu_lib_path = os.path.join(ascend_home, "lib")

    torch_npu_path = os.path.dirname(torch_npu.__file__)
    torch_npu_include = os.path.join(torch_npu_path, "include")
    torch_npu_lib_path = os.path.join(torch_npu_path, "lib")
    source_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "cpu_sparse_attn.cpp"
    )

    return load(
        name="cpu_sparse_attn",
        sources=[source_path],
        extra_cflags=[
            "-O3",
            "-std=c++20",
            "-fopenmp",
            "-march=armv8.2-a+sve+fp16+bf16",
            "-fPIC",
            f"-I{npu_include_path}",
            f"-I{torch_npu_include}",
        ],
        extra_ldflags=[
            "-fopenmp",
            f"-L{npu_lib_path}",
            "-lascendcl",
            f"-L{torch_npu_lib_path}",
            "-ltorch_npu",
        ],
        verbose=True,
    )


def validate_config(cfg: ExperimentConfig) -> None:
    assert cfg.num_reqs > 0
    assert cfg.thread_num > 0
    assert cfg.seq_len > 0
    assert cfg.topk > 0
    assert cfg.topk <= cfg.seq_len
    assert cfg.page_size > 0
    assert 0.0 <= cfg.load_ratio <= 1.0
    assert 0.0 <= cfg.contiguity <= 1.0
    assert cfg.repeat > 0
    assert cfg.warmup >= 0


def make_configs(args: argparse.Namespace) -> List[ExperimentConfig]:
    num_reqs_values = parse_list(args.num_reqs, int)
    thread_num_values = parse_list(args.thread_num, int)
    seq_len_values = parse_list(args.seq_len, int)
    topk_values = parse_list(args.topk, int)
    page_size_values = parse_list(args.page_size, int)
    load_ratio_values = parse_list(args.load_ratio, float)
    contiguity_values = parse_list(args.contiguity, float)

    configs = []
    for (
        num_reqs,
        thread_num,
        seq_len,
        topk,
        page_size,
        load_ratio,
        contiguity,
    ) in itertools.product(
        num_reqs_values,
        thread_num_values,
        seq_len_values,
        topk_values,
        page_size_values,
        load_ratio_values,
        contiguity_values,
    ):
        cfg = ExperimentConfig(
            npu_id=args.npu_id,
            num_reqs=num_reqs,
            thread_num=thread_num,
            seq_len=seq_len,
            topk=topk,
            page_size=page_size,
            load_ratio=load_ratio,
            contiguity=contiguity,
            repeat=args.repeat,
            warmup=args.warmup,
            seed=args.seed,
        )
        validate_config(cfg)
        configs.append(cfg)

    return configs


def generate_valid_indices(
    seq_len: int, valid_count: int, contiguity: float, rng: random.Random
) -> List[int]:
    if valid_count == 0:
        return []

    num_contiguous = int(valid_count * contiguity)
    num_contiguous = min(num_contiguous, valid_count)
    indices = set()

    if num_contiguous > 0:
        span_start = rng.randint(0, seq_len - num_contiguous)
        indices.update(range(span_start, span_start + num_contiguous))

    while len(indices) < valid_count:
        indices.add(rng.randrange(seq_len))

    return sorted(indices)


def generate_one_topk_indices(cfg: ExperimentConfig, rng: random.Random) -> List[int]:
    valid_count = int(round(cfg.topk * cfg.load_ratio))
    valid_count = max(0, min(valid_count, cfg.topk))
    valid_indices = generate_valid_indices(
        cfg.seq_len, valid_count, cfg.contiguity, rng
    )

    row = [-1] * cfg.topk
    if valid_count == 0:
        return row

    valid_positions = sorted(rng.sample(range(cfg.topk), valid_count))
    for position, token_idx in zip(valid_positions, valid_indices):
        row[position] = token_idx

    return row


def generate_topk_idx(cfg: ExperimentConfig) -> object:
    rng = random.Random(cfg.seed)
    rows = [generate_one_topk_indices(cfg, rng) for _ in range(cfg.num_reqs)]
    return torch.tensor(rows, dtype=torch.int32).unsqueeze(1)


def make_get_kv_topk_args(
    k_nope: object,
    k_pe: object,
    topk_idx: object,
    actual_seq_qlen: object,
    block_table: object,
    k_nope_topk: object,
    k_pe_topk: object,
    thread_num: int,
) -> tuple:
    return (
        k_nope.data_ptr(),
        k_pe.data_ptr(),
        topk_idx.data_ptr(),
        actual_seq_qlen.data_ptr(),
        block_table.data_ptr(),
        k_nope_topk.data_ptr(),
        k_pe_topk.data_ptr(),
        k_nope.shape,
        k_pe.shape,
        topk_idx.shape,
        actual_seq_qlen.shape,
        block_table.shape,
        k_nope_topk.shape,
        k_pe_topk.shape,
        thread_num,
    )


def run_one_experiment(cfg: ExperimentConfig, sparse_attn) -> Dict[str, object]:
    validate_config(cfg)
    torch.manual_seed(cfg.seed)
    rng_seed = cfg.seed + cfg.num_reqs * 1009 + cfg.seq_len * 17 + cfg.topk

    _q_nope = torch.rand([cfg.num_reqs, 1, NOPE_DIM], dtype=torch.bfloat16)
    _q_pe = torch.rand([cfg.num_reqs, 1, PE_DIM], dtype=torch.bfloat16)
    k_nope = torch.rand(
        [cfg.num_reqs * cfg.max_num_pages, cfg.page_size, HEAD_KV, NOPE_DIM],
        dtype=torch.bfloat16,
        pin_memory=True,
    )
    k_pe = torch.rand(
        [cfg.num_reqs * cfg.max_num_pages, cfg.page_size, HEAD_KV, PE_DIM],
        dtype=torch.bfloat16,
        pin_memory=True,
    )

    topk_idx = generate_topk_idx(
        ExperimentConfig(
            npu_id=cfg.npu_id,
            num_reqs=cfg.num_reqs,
            thread_num=cfg.thread_num,
            seq_len=cfg.seq_len,
            topk=cfg.topk,
            page_size=cfg.page_size,
            load_ratio=cfg.load_ratio,
            contiguity=cfg.contiguity,
            repeat=cfg.repeat,
            warmup=cfg.warmup,
            seed=rng_seed,
        )
    )
    actual_seq_qlen = torch.arange(cfg.num_reqs, dtype=torch.int32) + 1
    block_table = torch.arange(
        cfg.num_reqs * cfg.max_num_pages, dtype=torch.int32
    ).reshape([cfg.num_reqs, cfg.max_num_pages])

    topk_idx_flat = topk_idx.squeeze(1)
    block_indices = torch.clamp(topk_idx_flat // cfg.page_size, min=0)
    block_ids = torch.gather(block_table, 1, block_indices)
    offset_in_block = topk_idx_flat % cfg.page_size
    valid_mask = topk_idx_flat != -1

    k_nope_topk_gold = k_nope[block_ids, offset_in_block]
    k_pe_topk_gold = k_pe[block_ids, offset_in_block]

    k_nope_topk = torch.empty(
        [cfg.num_reqs, cfg.topk, HEAD_KV, NOPE_DIM],
        dtype=torch.bfloat16,
        device="cpu",
        pin_memory=True,
    )
    k_pe_topk = torch.empty(
        [cfg.num_reqs, cfg.topk, HEAD_KV, PE_DIM],
        dtype=torch.bfloat16,
        device="cpu",
        pin_memory=True,
    )
    call_args = make_get_kv_topk_args(
        k_nope,
        k_pe,
        topk_idx,
        actual_seq_qlen,
        block_table,
        k_nope_topk,
        k_pe_topk,
        cfg.thread_num,
    )

    sparse_attn.get_kv_topk(*call_args)

    k_nope_topk_gold[~valid_mask] = 0
    k_pe_topk_gold[~valid_mask] = 0
    k_nope_topk[~valid_mask] = 0
    k_pe_topk[~valid_mask] = 0
    correct = torch.equal(k_nope_topk_gold, k_nope_topk) and torch.equal(
        k_pe_topk_gold, k_pe_topk
    )

    if not correct:
        print(f"[ERROR] incorrect result for config: {cfg}")
        raise AssertionError(f"get_kv_topk correctness failed: {cfg}")

    for _ in range(cfg.warmup):
        sparse_attn.get_kv_topk(*call_args)

    time_start = time.perf_counter()
    for _ in range(cfg.repeat):
        sparse_attn.get_kv_topk(*call_args)
    time_end = time.perf_counter()

    avg_ms = (time_end - time_start) * 1000.0 / cfg.repeat
    tokens_to_load = int((topk_idx >= 0).sum().item())
    result = {
        "num_reqs": cfg.num_reqs,
        "thread_num": cfg.thread_num,
        "seq_len": cfg.seq_len,
        "topk": cfg.topk,
        "page_size": cfg.page_size,
        "max_num_pages": cfg.max_num_pages,
        "load_ratio": cfg.load_ratio,
        "contiguity": cfg.contiguity,
        "tokens_to_load": tokens_to_load,
        "avg_ms": avg_ms,
        "correct": correct,
    }
    print_result(result)
    return result


def print_result(result: Dict[str, object]) -> None:
    print(
        "[RESULT] "
        f"num_reqs={result['num_reqs']} "
        f"thread_num={result['thread_num']} "
        f"seq_len={result['seq_len']} "
        f"topk={result['topk']} "
        f"page_size={result['page_size']} "
        f"max_num_pages={result['max_num_pages']} "
        f"load_ratio={result['load_ratio']} "
        f"contiguity={result['contiguity']} "
        f"tokens_to_load={result['tokens_to_load']} "
        f"avg_ms={result['avg_ms']:.6f} "
        f"correct={result['correct']}"
    )


def print_summary(results: List[Dict[str, object]]) -> None:
    print("\nSummary:")
    headers = [
        "num_reqs",
        "thread_num",
        "seq_len",
        "topk",
        "page_size",
        "load_ratio",
        "contiguity",
        "tokens_to_load",
        "avg_ms",
        "correct",
    ]
    print(" | ".join(headers))
    for result in results:
        print(
            " | ".join(
                [
                    str(result["num_reqs"]),
                    str(result["thread_num"]),
                    str(result["seq_len"]),
                    str(result["topk"]),
                    str(result["page_size"]),
                    str(result["load_ratio"]),
                    str(result["contiguity"]),
                    str(result["tokens_to_load"]),
                    f"{result['avg_ms']:.6f}",
                    str(result["correct"]),
                ]
            )
        )


def write_csv(path: str, results: List[Dict[str, object]]) -> None:
    if not results:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV saved to: {path}")


def main() -> None:
    args = parse_args()
    initialize_runtime(args.npu_id)
    configs = make_configs(args)
    print(f"Total experiments: {len(configs)}")

    sparse_attn = load_sparse_attn_extension()
    results = [run_one_experiment(cfg, sparse_attn) for cfg in configs]
    print_summary(results)

    write_csv(args.csv, results)


if __name__ == "__main__":
    main()