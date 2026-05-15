#ifndef HEADER_ACLRTLAUNCH_CACHE_MISS_TOPK_H
#define HEADER_ACLRTLAUNCH_CACHE_MISS_TOPK_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_cache_miss_topk(uint32_t blockDim, aclrtStream stream, void* cache_miss_token_mask, void* available_slot_mask, void* topk_indices_old, uint32_t batchSize, uint32_t formerNum, uint32_t formerLength, uint32_t tailNum, uint32_t tailLength);
#endif
