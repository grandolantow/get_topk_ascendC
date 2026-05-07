// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/* include file of ascendc */
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t SEQ_LEN = 2048;

struct CacheMissTopkTilingData {
    uint32_t batchSize;
    uint32_t formerNum;
    uint32_t formerLength;
    uint32_t tailNum;
    uint32_t tailLength;
};

class KernelCacheMissTopk
{
public:
    __aicore__ inline KernelCacheMissTopk() {}

    __aicore__ inline void Init(GM_ADDR cache_miss_token_mask,
                                GM_ADDR available_slot_mask,
                                GM_ADDR topk_indices_old,
                                const CacheMissTopkTilingData& tilingData,
                                AscendC::TPipe* pipe)
    {
        this->batchSize = tilingData.batchSize;
        this->formerNum = tilingData.formerNum;
        this->formerLength = tilingData.formerLength;
        this->tailNum = tilingData.tailNum;
        this->tailLength = tilingData.tailLength;
        this->usedCoreNum = tilingData.formerNum + tilingData.tailNum;

        missMaskGm.SetGlobalBuffer((__gm__ uint8_t *)cache_miss_token_mask);
        availMaskGm.SetGlobalBuffer((__gm__ uint8_t *)available_slot_mask);
        idxGm.SetGlobalBuffer((__gm__ int64_t *)topk_indices_old);

        pipe->InitBuffer(missMaskQueue, BUFFER_NUM, SEQ_LEN * sizeof(uint8_t));
        pipe->InitBuffer(availMaskQueue, BUFFER_NUM, SEQ_LEN * sizeof(uint8_t));
        pipe->InitBuffer(idxQueue, BUFFER_NUM, SEQ_LEN * sizeof(int64_t));
        pipe->InitBuffer(resultQueue, BUFFER_NUM, SEQ_LEN * sizeof(uint8_t));

        pipe->InitBuffer(calcBuf, SEQ_LEN * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx >= this->usedCoreNum) {
            return;
        }

        uint32_t blockBatches = (blockIdx == this->usedCoreNum - 1) ? this->tailLength : this->formerLength;
        uint32_t startBatch = blockIdx * this->formerLength;

        for (uint32_t b = 0; b < blockBatches; ++b) {
            uint32_t batchIdx = startBatch + b;
            if (batchIdx >= this->batchSize) {
                break;
            }

            ProcessBatch(batchIdx);
        }
    }

private:
    __aicore__ inline void ProcessBatch(uint32_t batchIdx)
    {
        uint32_t offset = batchIdx * SEQ_LEN;

        // Load cache_miss_token_mask
        AscendC::LocalTensor<uint8_t> missLocal = missMaskQueue.AllocTensor<uint8_t>();
        AscendC::DataCopy(missLocal, missMaskGm[offset], SEQ_LEN);
        missMaskQueue.EnQue(missLocal);
        missLocal = missMaskQueue.DeQue<uint8_t>();

        // Load available_slot_mask
        AscendC::LocalTensor<uint8_t> availLocal = availMaskQueue.AllocTensor<uint8_t>();
        AscendC::DataCopy(availLocal, availMaskGm[offset], SEQ_LEN);
        availMaskQueue.EnQue(availLocal);
        availLocal = availMaskQueue.DeQue<uint8_t>();

        // Load topk_indices_old
        AscendC::LocalTensor<int64_t> idxLocal = idxQueue.AllocTensor<int64_t>();
        AscendC::DataCopy(idxLocal, idxGm[offset], SEQ_LEN);
        idxQueue.EnQue(idxLocal);
        idxLocal = idxQueue.DeQue<int64_t>();

        // Compute num_tokens_to_load and num_available_slot (sum)
        AscendC::SumParams params;
        params.outter = 1;
        params.inner = SEQ_LEN;
        params.n = SEQ_LEN;

        AscendC::LocalTensor<half> calcRet = calcBuf.Get<half>();

        AscendC::Cast(calcRet, missLocal, AscendC::RoundMode::CAST_NONE, SEQ_LEN);
        AscendC::Sum(calcRet, calcRet, params);
        int32_t numTokensToLoad = calcRet.GetValue(0);

        AscendC::Cast(calcRet, availLocal, AscendC::RoundMode::CAST_NONE, SEQ_LEN);
        AscendC::Sum(calcRet, calcRet, params);
        int32_t numAvailableSlot = calcRet.GetValue(0);

        int32_t numShortage = numTokensToLoad - numAvailableSlot;

        // Compute empty_slot_cumsum and select
        // empty_slot_mask = (topk_indices_old == -1)
        // selected_empty_slot_mask = (cumsum <= numShortage) & empty_slot_mask
        // result = available_slot_mask | selected_empty_slot_mask
        int32_t cumsum = 0;
        AscendC::LocalTensor<uint8_t> resultLocal = resultQueue.AllocTensor<uint8_t>();

        for (int32_t i = 0; i < SEQ_LEN; ++i) {
            int64_t idxVal = idxLocal.GetValue(i);
            bool isEmpty = (idxVal == -1);
            if (isEmpty) {
                cumsum++;
            }
            bool selected = (cumsum <= numShortage) && isEmpty;
            bool availVal = availLocal.GetValue(i);
            bool result = availVal || selected;
            resultLocal.SetValue(i, result ? 1 : 0);
        }

        // Write back to available_slot_mask (inplace)
        resultQueue.EnQue<uint8_t>(resultLocal);
        resultLocal = resultQueue.DeQue<uint8_t>();
        AscendC::DataCopy(availMaskGm[offset], resultLocal, SEQ_LEN);
        resultQueue.FreeTensor(resultLocal);

        idxQueue.FreeTensor(idxLocal);
        availMaskQueue.FreeTensor(availLocal);
        missMaskQueue.FreeTensor(missLocal);
    }

private:
    AscendC::TPipe* pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> missMaskQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> availMaskQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> idxQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> resultQueue;

    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;

    AscendC::GlobalTensor<uint8_t> missMaskGm;
    AscendC::GlobalTensor<uint8_t> availMaskGm;
    AscendC::GlobalTensor<int64_t> idxGm;

    uint32_t batchSize;
    uint32_t formerNum;
    uint32_t formerLength;
    uint32_t tailNum;
    uint32_t tailLength;
    uint32_t usedCoreNum;
};

extern "C" __global__ __aicore__ void cache_miss_topk(GM_ADDR cache_miss_token_mask,
                                                        GM_ADDR available_slot_mask,
                                                        GM_ADDR topk_indices_old,
                                                        uint32_t batchSize,
                                                        uint32_t formerNum,
                                                        uint32_t formerLength,
                                                        uint32_t tailNum,
                                                        uint32_t tailLength)
{
    CacheMissTopkTilingData tilingData;
    tilingData.batchSize = batchSize;
    tilingData.formerNum = formerNum;
    tilingData.formerLength = formerLength;
    tilingData.tailNum = tailNum;
    tilingData.tailLength = tailLength;

    AscendC::TPipe pipe;
    KernelCacheMissTopk op;
    op.Init(cache_miss_token_mask, available_slot_mask, topk_indices_old, tilingData, &pipe);
    op.Process();
}