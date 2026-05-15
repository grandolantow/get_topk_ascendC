#ifndef __KERNEL_HELLOWORLD__KERNEL_FUN_H__
#define __KERNEL_HELLOWORLD__KERNEL_FUN_H__

#undef __global__
#define __global__ inline
#define helloworld helloworld_origin
#include "/home/s886374/kvoffload/ascend-kernel/csrc/ops/helloworld/op_kernel/kernel_helloworld.cpp"

#undef helloworld
#undef __global__
#if ASCENDC_CPU_DEBUG
#define __global__
#else
#define __global__ __attribute__((cce_kernel))
#endif

#ifndef ONE_CORE_DUMP_SIZE
#define ONE_CORE_DUMP_SIZE 1048576 * 1
#endif

extern "C" __global__ [aicore] void auto_gen_helloworld_kernel(
__attribute__((cce_global)) uint8_t* x, __attribute__((cce_global)) uint8_t* y, __attribute__((cce_global)) uint8_t* z, uint32_t totalLength, GM_ADDR overflow_status) {
#if defined(HAVE_WORKSPACE)
    GM_ADDR workspace_param;
    GM_ADDR workspace_usr;
#if defined(HAVE_TILING)
    workspace_param = z;
#else
    workspace_param = totalLength;
#endif
    AscendC::SetSysWorkspaceForce(workspace_param);
    workspace_usr = AscendC::GetUserWorkspace(workspace_param);
#if defined(HAVE_TILING)
    z = workspace_usr;
#else
    totalLength = workspace_usr;
#endif
#endif
    helloworld_origin(x, y, z, totalLength);
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
