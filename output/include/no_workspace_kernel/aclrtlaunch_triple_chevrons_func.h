
#ifndef HEADER_ACLRTLAUNCH_CACHE_MISS_TOPK_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_CACHE_MISS_TOPK_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_cache_miss_topk(uint32_t blockDim, void* stream, void* cache_miss_token_mask, void* available_slot_mask, void* topk_indices_old, uint32_t batchSize, uint32_t formerNum, uint32_t formerLength, uint32_t tailNum, uint32_t tailLength);

inline uint32_t cache_miss_topk(uint32_t blockDim, void* hold, void* stream, void* cache_miss_token_mask, void* available_slot_mask, void* topk_indices_old, uint32_t batchSize, uint32_t formerNum, uint32_t formerLength, uint32_t tailNum, uint32_t tailLength)
{
    (void)hold;
    return aclrtlaunch_cache_miss_topk(blockDim, stream, cache_miss_token_mask, available_slot_mask, topk_indices_old, batchSize, formerNum, formerLength, tailNum, tailLength);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_HELLOWORLD_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_HELLOWORLD_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_helloworld(uint32_t blockDim, void* stream, void* x, void* y, void* z, uint32_t totalLength);

inline uint32_t helloworld(uint32_t blockDim, void* hold, void* stream, void* x, void* y, void* z, uint32_t totalLength)
{
    (void)hold;
    return aclrtlaunch_helloworld(blockDim, stream, x, y, z, totalLength);
}

#endif
