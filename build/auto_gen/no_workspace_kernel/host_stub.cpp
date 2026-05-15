#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>
#include <securec.h>

#ifndef ASCENDC_DUMP
#define ASCENDC_DUMP 1
#endif

#if defined(ASCENDC_DUMP) && (ASCENDC_DUMP == 0)
    #undef ASCENDC_DUMP
#endif

static char ascendcErrMsg[1024] = {0};

static void *g_kernel_handle_aiv = nullptr;

struct ascend_kernels {
    uint32_t version;
    uint32_t type_cnt;
    uint32_t aiv_type;
    uint32_t aiv_len;
    uint32_t aiv_file_len;
    uint8_t aiv_buf[8336];
} __ascend_kernel_ascend910_9382_no_workspace_kernel __attribute__ ((section (".ascend.kernel.ascend910_9382.no_workspace_kernel"))) = {1,1,1,8336,8336,{0}};

extern "C" {
uint32_t RegisterAscendBinary(const char *fileBuf, size_t fileSize, uint32_t type, void **handle);
uint32_t LaunchAscendKernel(void *handle, const uint64_t key, const uint32_t blockDim, void **args,
                            uint32_t size, const void *stream);
uint32_t GetAscendCoreSyncAddr(void **addr);
int UnregisterAscendBinary(void *hdl);
void StartAscendProf(const char *name, uint64_t *startTime);
void ReportAscendProf(const char *name, uint32_t blockDim, uint32_t taskType, const uint64_t startTime);
bool GetAscendProfStatus();
uint32_t AllocAscendMemDevice(void **devMem, uint64_t size);
uint32_t FreeAscendMemDevice(void *devMem);
bool AscendCheckSoCVersion(const char *socVersion, char* errMsg);
void AscendProfRegister();
uint32_t GetCoreNumForMixVectorCore(uint32_t *aiCoreNum, uint32_t *vectorCoreNum);
uint32_t LaunchAscendKernelForVectorCore(const char *opType, void *handle, const uint64_t key, void **args, uint32_t size,
    const void *stream, bool enbaleProf, uint32_t aicBlockDim, uint32_t aivBlockDim, uint32_t aivBlockDimOffset);
}

namespace Adx {
    void AdumpPrintWorkSpace(const void *workSpaceAddr, const size_t dumpWorkSpaceSize,
                            void *stream, const char *opType);
}

    class KernelHandleGradUnregister {
    private:
        KernelHandleGradUnregister() {}

    public:
        KernelHandleGradUnregister(const KernelHandleGradUnregister&) = delete;
        KernelHandleGradUnregister& operator=(const KernelHandleGradUnregister&) = delete;

        static KernelHandleGradUnregister& GetInstance() {
            static KernelHandleGradUnregister instance;
            return instance;
        }
        ~KernelHandleGradUnregister(){
            if (g_kernel_handle_aiv) {
                UnregisterAscendBinary(g_kernel_handle_aiv);
                g_kernel_handle_aiv = nullptr;
            }
        }
    };

static void __register_kernels(void) __attribute__((constructor));
void __register_kernels(void)
{
    const char* compileSocVersion = "ascend910_9382";
    uint32_t ret;

    bool checkSocVersion = AscendCheckSoCVersion(compileSocVersion, ascendcErrMsg);
    if (!checkSocVersion) {
        return;
    }
    ret = RegisterAscendBinary(
        (const char *)__ascend_kernel_ascend910_9382_no_workspace_kernel.aiv_buf,
        __ascend_kernel_ascend910_9382_no_workspace_kernel.aiv_file_len,
        1,
        &g_kernel_handle_aiv);
    if (ret != 0) {
        printf("RegisterAscendBinary aiv ret %u \n", ret);
    }

    AscendProfRegister();
}





uint32_t launch_and_profiling_cache_miss_topk(uint64_t func_key, uint32_t blockDim, void* stream, void **args, uint32_t size)
{
    uint64_t startTime;
    const char *name = "cache_miss_topk";
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        StartAscendProf(name, &startTime);
    }
    if (g_kernel_handle_aiv == nullptr) {
        printf("[ERROR] %s\n", ascendcErrMsg);
        return 0;
    }
    uint32_t ret = LaunchAscendKernel(g_kernel_handle_aiv, func_key, blockDim, args, size, stream);
    if (ret != 0) {
        printf("LaunchAscendKernel ret %u\n", ret);
    }
    if (profStatus) {
        ReportAscendProf(name, blockDim, 1, startTime);
    }
    return ret;
}

extern "C" uint32_t aclrtlaunch_cache_miss_topk(uint32_t blockDim, void* stream, void* cache_miss_token_mask, void* available_slot_mask, void* topk_indices_old, uint32_t batchSize, uint32_t formerNum, uint32_t formerLength, uint32_t tailNum, uint32_t tailLength)
{
    struct {
        alignas(((alignof(void*) + 3) >> 2) << 2) void* cache_miss_token_mask;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* available_slot_mask;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* topk_indices_old;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t batchSize;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t formerNum;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t formerLength;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t tailNum;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t tailLength;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* __ascendc_overflow;
    } __ascendc_args;

    uint32_t __ascendc_ret;
    constexpr uint32_t __ascendc_overflow_status_size = 8;
    AllocAscendMemDevice(&(__ascendc_args.__ascendc_overflow), __ascendc_overflow_status_size);
    __ascendc_args.cache_miss_token_mask = cache_miss_token_mask;
    __ascendc_args.available_slot_mask = available_slot_mask;
    __ascendc_args.topk_indices_old = topk_indices_old;
    __ascendc_args.batchSize = batchSize;
    __ascendc_args.formerNum = formerNum;
    __ascendc_args.formerLength = formerLength;
    __ascendc_args.tailNum = tailNum;
    __ascendc_args.tailLength = tailLength;

    __ascendc_ret = launch_and_profiling_cache_miss_topk(0, blockDim, stream, (void **)&__ascendc_args, sizeof(__ascendc_args));
    KernelHandleGradUnregister::GetInstance();
    FreeAscendMemDevice(__ascendc_args.__ascendc_overflow);
    return __ascendc_ret;
}


uint32_t launch_and_profiling_helloworld(uint64_t func_key, uint32_t blockDim, void* stream, void **args, uint32_t size)
{
    uint64_t startTime;
    const char *name = "helloworld";
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        StartAscendProf(name, &startTime);
    }
    if (g_kernel_handle_aiv == nullptr) {
        printf("[ERROR] %s\n", ascendcErrMsg);
        return 0;
    }
    uint32_t ret = LaunchAscendKernel(g_kernel_handle_aiv, func_key, blockDim, args, size, stream);
    if (ret != 0) {
        printf("LaunchAscendKernel ret %u\n", ret);
    }
    if (profStatus) {
        ReportAscendProf(name, blockDim, 1, startTime);
    }
    return ret;
}

extern "C" uint32_t aclrtlaunch_helloworld(uint32_t blockDim, void* stream, void* x, void* y, void* z, uint32_t totalLength)
{
    struct {
        alignas(((alignof(void*) + 3) >> 2) << 2) void* x;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* y;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* z;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t totalLength;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* __ascendc_overflow;
    } __ascendc_args;

    uint32_t __ascendc_ret;
    constexpr uint32_t __ascendc_overflow_status_size = 8;
    AllocAscendMemDevice(&(__ascendc_args.__ascendc_overflow), __ascendc_overflow_status_size);
    __ascendc_args.x = x;
    __ascendc_args.y = y;
    __ascendc_args.z = z;
    __ascendc_args.totalLength = totalLength;

    __ascendc_ret = launch_and_profiling_helloworld(1, blockDim, stream, (void **)&__ascendc_args, sizeof(__ascendc_args));
    KernelHandleGradUnregister::GetInstance();
    FreeAscendMemDevice(__ascendc_args.__ascendc_overflow);
    return __ascendc_ret;
}
