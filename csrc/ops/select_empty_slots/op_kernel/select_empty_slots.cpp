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

constexpr int32_t BUFFER_NUM = 1;
constexpr int64_t SEQ_LEN = 2048;

class KernelSelectEmptySlots
{
public:
    __aicore__ inline KernelSelectEmptySlots() {}

    __aicore__ inline void Init(GM_ADDR cache_miss_token_mask,
                                GM_ADDR available_slot_mask,
                                GM_ADDR topk_indices_old,
                                GM_ADDR out,
                                int64_t batchSize,
                                int64_t formerNum,
                                int64_t formerLength,
                                int64_t tailNum,
                                int64_t tailLength)
    {
        this->batchSize = batchSize;
        this->formerNum = formerNum;
        this->formerLength = formerLength;
        this->tailNum = tailNum;
        this->tailLength = tailLength;
        this->usedCoreNum = formerNum + tailNum;

        missMaskGm.SetGlobalBuffer((__gm__ uint8_t *)cache_miss_token_mask);
        availMaskGm.SetGlobalBuffer((__gm__ uint8_t *)available_slot_mask);
        idxGm.SetGlobalBuffer((__gm__ int32_t *)topk_indices_old);
        outGm.SetGlobalBuffer((__gm__ uint8_t *)out);

        pipe.InitBuffer(missMaskQueue, BUFFER_NUM, SEQ_LEN * sizeof(uint8_t));
        pipe.InitBuffer(availMaskQueue, BUFFER_NUM, SEQ_LEN * sizeof(uint8_t));
        pipe.InitBuffer(idxQueue, BUFFER_NUM, SEQ_LEN * sizeof(int32_t));
        pipe.InitBuffer(resultQueue, BUFFER_NUM, SEQ_LEN * sizeof(uint8_t));
    }

    __aicore__ inline void Process()
    {
        int64_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx >= this->usedCoreNum) {
            return;
        }

        int64_t blockBatches = (blockIdx == this->usedCoreNum - 1) ? this->tailLength : this->formerLength;
        int64_t startBatch = blockIdx * this->formerLength;

        for (int64_t b = 0; b < blockBatches; ++b) {
            int64_t batchIdx = startBatch + b;
            if (batchIdx >= this->batchSize) {
                break;
            }

            ProcessBatch(batchIdx);
        }
    }

private:
    __aicore__ inline void ProcessBatch(int64_t batchIdx)
    {
        int64_t offset = batchIdx * SEQ_LEN;

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
        AscendC::LocalTensor<int32_t> idxLocal = idxQueue.AllocTensor<int32_t>();
        AscendC::DataCopy(idxLocal, idxGm[offset], SEQ_LEN);
        idxQueue.EnQue(idxLocal);
        idxLocal = idxQueue.DeQue<int32_t>();

        // Compute num_tokens_to_load and num_available_slot (sum)
        int32_t numTokensToLoad = 0;
        int32_t numAvailableSlot = 0;
        for (int64_t i = 0; i < SEQ_LEN; ++i) {
            numTokensToLoad += missLocal.GetValue(i);
            numAvailableSlot += availLocal.GetValue(i);
        }

        int32_t numShortage = numTokensToLoad - numAvailableSlot;

        // Compute empty_slot_cumsum and select
        // empty_slot_mask = (topk_indices_old == -1)
        // selected_empty_slot_mask = (cumsum <= numShortage) & empty_slot_mask
        // result = available_slot_mask | selected_empty_slot_mask
        int32_t cumsum = 0;
        AscendC::LocalTensor<uint8_t> resultLocal = resultQueue.AllocTensor<uint8_t>();

        for (int64_t i = 0; i < SEQ_LEN; ++i) {
            int32_t idxVal = idxLocal.GetValue(i);
            bool isEmpty = (idxVal == -1);
            if (isEmpty) {
                cumsum++;
            }
            bool selected = (cumsum <= numShortage) && isEmpty;
            bool availVal = availLocal.GetValue(i);
            bool result = availVal || selected;
            resultLocal.SetValue(i, result ? 1 : 0);
        }

        // Write back
        resultQueue.EnQue<uint8_t>(resultLocal);
        resultLocal = resultQueue.DeQue<uint8_t>();
        AscendC::DataCopy(outGm[offset], resultLocal, SEQ_LEN);
        resultQueue.FreeTensor(resultLocal);

        idxQueue.FreeTensor(idxLocal);
        availMaskQueue.FreeTensor(availLocal);
        missMaskQueue.FreeTensor(missLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> missMaskQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> availMaskQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> idxQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> resultQueue;

    AscendC::GlobalTensor<uint8_t> missMaskGm;
    AscendC::GlobalTensor<uint8_t> availMaskGm;
    AscendC::GlobalTensor<int32_t> idxGm;
    AscendC::GlobalTensor<uint8_t> outGm;

    int64_t batchSize;
    int64_t formerNum;
    int64_t formerLength;
    int64_t tailNum;
    int64_t tailLength;
    int64_t usedCoreNum;
};

extern "C" __global__ __aicore__ void select_empty_slots(GM_ADDR cache_miss_token_mask,
                                                           GM_ADDR available_slot_mask,
                                                           GM_ADDR topk_indices_old,
                                                           GM_ADDR out,
                                                           int64_t batchSize,
                                                           int64_t formerNum,
                                                           int64_t formerLength,
                                                           int64_t tailNum,
                                                           int64_t tailLength)
{
    KernelSelectEmptySlots op;
    op.Init(cache_miss_token_mask, available_slot_mask, topk_indices_old, out,
            batchSize, formerNum, formerLength, tailNum, tailLength);
    op.Process();
}
