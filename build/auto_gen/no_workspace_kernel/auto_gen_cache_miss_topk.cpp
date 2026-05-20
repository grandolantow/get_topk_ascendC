#ifndef __CACHE_MISS_TOPK__KERNEL_FUN_H__
#define __CACHE_MISS_TOPK__KERNEL_FUN_H__

#undef __global__
#define __global__ inline
#define cache_miss_topk cache_miss_topk_origin
#include "/home/s886374/kvoffload/ascend-kernel/csrc/ops/cache_miss_topk/op_kernel/cache_miss_topk.cpp"

#undef cache_miss_topk
#undef __global__
#if ASCENDC_CPU_DEBUG
#define __global__
#else
#define __global__ __attribute__((cce_kernel))
#endif

#ifndef ONE_CORE_DUMP_SIZE
#define ONE_CORE_DUMP_SIZE 1048576 * 1
#endif

extern "C" __global__ [aicore] void auto_gen_cache_miss_topk_kernel(
__attribute__((cce_global)) uint8_t* cache_miss_token_mask, __attribute__((cce_global)) uint8_t* available_slot_mask, __attribute__((cce_global)) uint8_t* topk_indices_old, uint32_t batchSize, uint32_t formerNum, uint32_t formerLength, uint32_t tailNum, uint32_t tailLength, GM_ADDR overflow_status) {
#if defined(HAVE_WORKSPACE)
    GM_ADDR workspace_param;
    GM_ADDR workspace_usr;
#if defined(HAVE_TILING)
    workspace_param = tailNum;
#else
    workspace_param = tailLength;
#endif
    AscendC::SetSysWorkspaceForce(workspace_param);
    workspace_usr = AscendC::GetUserWorkspace(workspace_param);
#if defined(HAVE_TILING)
    tailNum = workspace_usr;
#else
    tailLength = workspace_usr;
#endif
#endif
    cache_miss_topk_origin(cache_miss_token_mask, available_slot_mask, topk_indices_old, batchSize, formerNum, formerLength, tailNum, tailLength);
#if defined(ASCENDC_DUMP) && defined(ASCENDC_DEBUG)
    AscendC::WriteBackOverflow(overflow_status);
#endif
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
    pipe_barrier(PIPE_ALL);
    dsb(mem_dsb_t::DSB_ALL);
    dci();
#endif
}

#endif
