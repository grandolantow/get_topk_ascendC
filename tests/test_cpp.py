import random
import time

import torch
import torch_npu
import numpy as np
import os
import shutil
from torch.utils.cpp_extension import load
import torch.nn.functional as F

# apt install clang
# apt-get install libomp-dev

os.environ["TORCH_EXTENSIONS_ALWAYS_BUILD"] = "1"

# 清理缓存目录
cache_dir = "/root/.cache/torch_extensions/py311_cpu/cpu_sparse_attn"
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print(f"已清理缓存目录: {cache_dir}")

ascend_home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
npu_include_path = os.path.join(ascend_home, "include")
npu_lib_path = os.path.join(ascend_home, "lib64")

if not os.path.exists(npu_lib_path):
    npu_lib_path = os.path.join(ascend_home, "lib")

torch_npu_path = os.path.dirname(torch_npu.__file__)
torch_npu_include = os.path.join(torch_npu_path, "include")
torch_npu_lib_path = os.path.join(torch_npu_path, "lib")

os.environ["TORCH_EXTENSIONS_ALWAYS_BUILD"] = "1"
os.environ['CXX'] = 'clang++'
os.environ['CC'] = 'clang'

cpu_sparse_attn = load(
    name="cpu_sparse_attn",
    sources=["/home/s886374/kvoffload/ascend-kernel/tests/cpu_sparse_attn.cpp"],
    extra_cflags=[
        "-O3",
        "-std=c++20",
        "-fopenmp",
        "-march=armv8.2-a+sve+fp16+bf16",
        # "-march=native",
        "-fPIC",
        # f"-I{hgemm_path}",  # 添加包含路径
        f"-I{npu_include_path}",
        f"-I{torch_npu_include}",
    ],
    # extra_ldflags=["-fopenmp"],
    extra_ldflags=[
        "-fopenmp",
        f"-L{npu_lib_path}",
        "-lascendcl",
        f"-L{torch_npu_lib_path}",
        "-ltorch_npu",
    ],
    verbose=True,  # 添加 verbose 查看编译过程
)

# os._exit(0)

# const at::Tensor& q_nope_,    // 查询向量的非位置编码部分 [token_num, head_q, nope_dim]
# const at::Tensor& q_pe_,      // 查询向量的位置编码部分 [token_num, head_q, pe_dim]
# const at::Tensor& k_nope_,    // 键向量的非位置编码部分 [num_page, page_size, head_kv, value_dim]
# const at::Tensor& k_pe_,      // 键向量的位置编码部分 [num_page, page_size, head_kv, value_dim]
# const at::Tensor& topk_indices, // TopK索引 [N, 1, S]
# const at::Tensor& actual_seq_lengths_query, // 实际序列长度 [num_requests]
# const at::Tensor& block_table, // 块表 [num_requests, max_num_pages]
# int64_t tnum,                // 线程数
# // //at::Tensor& k_nope_npu,      // 输出的非位置编码键向量
# // //at::Tensor& k_pe_npu,
# at::Tensor& k_nope_cpu_pinned,      // 输出的非位置编码键向量
# at::Tensor& k_pe_cpu_pinned) {      // 输出的位置编码键向量

bs = 4
thread_num = 16
sparse_ratio = 0.5

sparse_attn = cpu_sparse_attn
q_nope = torch.rand([bs, 1, 512], dtype=torch.bfloat16)
q_pe = torch.rand([bs, 1, 64], dtype=torch.bfloat16)
k_nope = torch.rand([bs * 256, 128, 1, 512], dtype=torch.bfloat16, pin_memory=True)
k_pe = torch.rand([bs * 256, 128, 1, 64], dtype=torch.bfloat16, pin_memory=True)
topk_idx_list = []
for _ in range(bs):
    topk_idx = random.sample(list(range(32768)), 2048); topk_idx.sort()
    topk_idx = torch.tensor(topk_idx, dtype=torch.int32)
    mask = torch.rand([2048]) < sparse_ratio # simulate cache hit and no need to load
    topk_idx[mask] = -1
    topk_idx = topk_idx.unsqueeze(0).unsqueeze(0) # [bs, 1, 2048]
    topk_idx_list.append(topk_idx)

topk_idx = torch.cat(topk_idx_list, dim=0)
actual_seq_qlen = torch.arange(bs, dtype=torch.int32) + 1 # [bs]
block_table = torch.arange(bs * 256, dtype=torch.int32).reshape([bs, 256]) # [bs, max_page_num]
topk_idx_flat = topk_idx.squeeze(1)
block_indices = torch.clamp(topk_idx_flat // 128, min=0)
block_ids = torch.gather(block_table, 1, block_indices)
offset_in_block = topk_idx_flat % 128
valid_mask = topk_idx_flat != -1
k_nope_topk_gold = k_nope[block_ids, offset_in_block]
k_pe_topk_gold = k_pe[block_ids, offset_in_block]
print(f'>>>>> py get_kv_topk start, num tokens to load: {(topk_idx >= 0).sum().item()}')

# k_nope_topk, k_pe_topk = sparse_attn.get_kv_topk(k_nope, k_pe, topk_idx, actual_seq_qlen, block_table, thread_num)
k_nope_topk = torch.empty([bs, 2048, 1, 512], dtype=torch.bfloat16, device='cpu', pin_memory=True)
k_pe_topk = torch.empty([bs, 2048, 1, 64], dtype=torch.bfloat16, device='cpu', pin_memory=True)
args = (
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
# sparse_attn.get_kv_topk(k_nope, k_pe, topk_idx, actual_seq_qlen, block_table, thread_num, k_nope_topk, k_pe_topk)
sparse_attn.get_kv_topk(*args)

k_nope_topk_gold[~valid_mask] = 0; k_pe_topk_gold[~valid_mask] = 0
k_nope_topk[~valid_mask] = 0; k_pe_topk[~valid_mask] = 0
print(
    f'>>>>> py get_kv_topk done, '
    f'k_nope = {k_nope_topk.shape}, equal = {torch.equal(k_nope_topk_gold, k_nope_topk)}; '
    f'k_pe = {k_pe_topk.shape}, equal = {torch.equal(k_pe_topk_gold, k_pe_topk)}'
)

repeat = 100
time_start = time.time()
for _ in range(repeat):
    sparse_attn.get_kv_topk(*args)
time_end = time.time()
time_avg = (time_end - time_start) / repeat
print(f'>>>>> bs={bs}, thread_num={thread_num}, sparse_ratio={sparse_ratio}, avg time: {time_avg * 1000} ms')

# bs1: 0.21-0.25 ms, 10GB/s
#   bs1 sparse 50%, 0.1 ms
#   bs1 sparse 90%, 0.02 ms
#   bs1 2 thread, 0.09 ms, 25 GB/s
#   bs1 4 thread, 0.05 ms, 45 GB/s
#   bs1 8 thread, 0.03 ms, 75 GB/s
#   bs1 16 thread, 0.02 ms
# bs2: 0.5 ms
# bs4: 1.2 ms
#   bs4 2 thread, 0.6 ms
#   bs4 4 thread, 0.3 ms
#   bs4 4 thread 50%, 0.15 ms
#   bs4 8 thread, 0.15 ms
#   bs4 16 thread, 0.09 ms

# time_start = time.time()
# time_end = time.time()
# (time_end - time_start) * 1000


# 90.90.97.29
# >>>>> bs=4, thread_num=1, sparse_ratio=0.0, avg time: 1.5247583389282227 ms
# >>>>> bs=4, thread_num=8, sparse_ratio=0.0, avg time: 0.26349782943725586 ms
# >>>>> bs=4, thread_num=16, sparse_ratio=0.0, avg time: 0.15278100967407227 ms
# >>>>> bs=4, thread_num=16, sparse_ratio=0.9, avg time: 0.030124187469482422 ms
