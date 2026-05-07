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

class KernelSetDiffExchange
{
public:
    __aicore__ inline KernelSetDiffExchange() {}

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR out, GM_ADDR workspace,
                                int64_t batchSize, int64_t topK, int64_t totalElements,
                                int64_t formerNum, int64_t formerLength,
                                int64_t tailNum, int64_t tailLength,
                                int64_t tileLength)
    {
        this->batchSize = batchSize;
        this->topK = topK;
        this->totalElements = totalElements;
        this->formerNum = formerNum;
        this->formerLength = formerLength;
        this->tailNum = tailNum;
        this->tailLength = tailLength;
        this->tileLength = tileLength;
        this->usedCoreNum = formerNum + tailNum;

        // ���� GM ָ��
        aGm.SetGlobalBuffer((__gm__ int32_t *)a);
        bGm.SetGlobalBuffer((__gm__ int32_t *)b);
        outGm.SetGlobalBuffer((__gm__ int32_t *)out);

        // ��ʼ�� queue
        // A buffer����ǰtile��int32����
        pipe.InitBuffer(inQueueA, 1, this->tileLength * sizeof(int32_t));
        // B buffer���洢2048��int32��һ��batch��B���ݣ�
        pipe.InitBuffer(inQueueB, 1, this->topK * sizeof(int32_t));
        // ���buffer���޸ĺ��B���ݣ�int32��
        pipe.InitBuffer(outQueueB, 1, this->topK * sizeof(int32_t));

        // CompareScalar ���: 256 uint8 = 2048 bits, ��Ӧ2048���ȽϽ��
        pipe.InitBuffer(cmpResultBuffer, 256);
        // Cast buffer: ��uint8ת��Ϊhalf��256 bytes -> 512 bytes (256 * 2)
        pipe.InitBuffer(castBuffer, 512);
        // ReduceMax dst buffer: 32��half
        pipe.InitBuffer(reduceDstBuffer, 64);
        // ReduceMax tmp buffer: half���ͣ�����ReduceMax����
        pipe.InitBuffer(reduceTmpBuffer, 2048);
    }

    __aicore__ inline void Process()
    {
        int64_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx >= this->usedCoreNum) {
            return;
        }

        // ���㵱ǰblock�����Ԫ�ط�Χ��һάչ����
        int64_t blockLength = (blockIdx == this->usedCoreNum - 1) ? this->tailLength : this->formerLength;
        int64_t startElement = blockIdx * this->formerLength;
        int64_t endElement = startElement + blockLength;

        if (startElement >= this->totalElements) {
            return;
        }

        // ���㵱ǰblock�����batch��Χ
        int64_t startBatch = startElement / this->topK;
        int64_t endBatch = (endElement - 1) / this->topK;

        // ��batch����ÿ��batch 2048��Ԫ�أ�
        for (int64_t batchIdx = startBatch; batchIdx <= endBatch; ++batchIdx) {
            int64_t batchStart = batchIdx * this->topK;
            int64_t batchEnd = batchStart + this->topK;

            // ��鵱ǰbatch�Ƿ���block�Ĵ���Χ��
            if (batchEnd <= startElement || batchStart >= endElement) {
                continue;
            }

            // �������batch
            ProcessBatch(batchIdx);
        }
    }

private:
    __aicore__ inline void ProcessBatch(int64_t batchIdx)
    {
        int64_t bOffset = batchIdx * this->topK;
        int64_t aOffset = batchIdx * this->topK;

        // ����B���ݵ�UB
        AscendC::LocalTensor<int32_t> bLocal = inQueueB.AllocTensor<int32_t>();
        AscendC::DataCopy(bLocal, bGm[bOffset], this->topK);
        inQueueB.EnQue(bLocal);
        bLocal = inQueueB.DeQue<int32_t>();

        // ����exchange buffer���ڴ洢A���е�Ԫ�أ���A�е�����B�У�
        // ʹ��UB�е���ʱ������
        int32_t exchangeBuffer[2048];
        int32_t exchangeCount = 0;

        // �׶�1: �ռ�A���е�Ԫ�ص�exchange buffer
        CollectOnlyInA(aOffset, bLocal, exchangeBuffer, exchangeCount);

        // �׶�2: ��A���е�Ԫ��д��B���е�λ��
        WriteToBOnlyPositions(bLocal, exchangeBuffer, exchangeCount, bOffset);

        // �ͷ���Դ
        inQueueB.FreeTensor(bLocal);
    }

    __aicore__ inline void CollectOnlyInA(int64_t aOffset,
                                          AscendC::LocalTensor<int32_t>& bLocal,
                                          int32_t* exchangeBuffer,
                                          int32_t& exchangeCount)
    {
        AscendC::LocalTensor<uint8_t> cmpResult = cmpResultBuffer.Get<uint8_t>();
        AscendC::LocalTensor<half> castHalf = castBuffer.Get<half>();
        AscendC::LocalTensor<half> reduceDst = reduceDstBuffer.Get<half>();
        AscendC::LocalTensor<half> reduceTmp = reduceTmpBuffer.Get<half>();

        int64_t cmpResultBytes = this->topK / 8;  // 256 bytes
        exchangeCount = 0;

        // ��tile����A����
        for (int64_t tileStart = 0; tileStart < this->topK; tileStart += this->tileLength) {
            int64_t currentTileLength = this->topK - tileStart;
            if (currentTileLength > this->tileLength) {
                currentTileLength = this->tileLength;
            }

            // ���ص�ǰtile��A����
            AscendC::LocalTensor<int32_t> aLocal = inQueueA.AllocTensor<int32_t>();
            AscendC::DataCopy(aLocal, aGm[aOffset + tileStart], currentTileLength);
            inQueueA.EnQue(aLocal);
            aLocal = inQueueA.DeQue<int32_t>();

            // �������tile��ÿ��Ԫ��
            for (int64_t i = 0; i < currentTileLength; ++i) {
                int32_t aVal = aLocal.GetValue(i);

                // ���aVal�Ƿ���B��
                AscendC::CompareScalar(cmpResult, bLocal, aVal, AscendC::CMPMODE::EQ, this->topK);
                AscendC::Cast(castHalf, cmpResult, AscendC::RoundMode::CAST_NONE, cmpResultBytes);
                AscendC::ReduceMax<half>(reduceDst, castHalf, reduceTmp, cmpResultBytes, true);

                float maxVal = static_cast<float>(reduceDst.GetValue(0));
                bool foundInB = (maxVal > 0.0f);

                // �������B�У�����exchange buffer
                if (!foundInB && exchangeCount < this->topK) {
                    exchangeBuffer[exchangeCount] = aVal;
                    exchangeCount++;
                }
            }

            inQueueA.FreeTensor(aLocal);
        }
    }

    __aicore__ inline void WriteToBOnlyPositions(AscendC::LocalTensor<int32_t>& bLocal,
                                                 int32_t* exchangeBuffer,
                                                 int32_t exchangeCount,
                                                 int64_t bOffset)
    {
        AscendC::LocalTensor<uint8_t> cmpResult = cmpResultBuffer.Get<uint8_t>();
        AscendC::LocalTensor<half> castHalf = castBuffer.Get<half>();
        AscendC::LocalTensor<half> reduceDst = reduceDstBuffer.Get<half>();
        AscendC::LocalTensor<half> reduceTmp = reduceTmpBuffer.Get<half>();

        int64_t cmpResultBytes = this->topK / 8;  // 256 bytes
        int32_t exchangeIdx = 0;

        // �������buffer
        AscendC::LocalTensor<int32_t> outLocal = outQueueB.AllocTensor<int32_t>();

        // �Ƚ�B���ݸ��Ƶ����buffer
        for (int64_t i = 0; i < this->topK; ++i) {
            outLocal.SetValue(i, bLocal.GetValue(i));
        }

        // ����A���ݣ������ж�B�е�Ԫ���Ƿ���A�У�
        // ������Ҫ��UB����һ��������A���ݸ���
        // ����A��B��С��ͬ�����ǿ��Ը���inQueueA������Ҫ��ʱ�洢
        int32_t aBuffer[2048];
        for (int64_t tileStart = 0; tileStart < this->topK; tileStart += this->tileLength) {
            int64_t currentTileLength = this->topK - tileStart;
            if (currentTileLength > this->tileLength) {
                currentTileLength = this->tileLength;
            }

            AscendC::LocalTensor<int32_t> aLocal = inQueueA.AllocTensor<int32_t>();
            AscendC::DataCopy(aLocal, aGm[bOffset + tileStart], currentTileLength);
            inQueueA.EnQue(aLocal);
            aLocal = inQueueA.DeQue<int32_t>();

            for (int64_t i = 0; i < currentTileLength; ++i) {
                aBuffer[tileStart + i] = aLocal.GetValue(i);
            }

            inQueueA.FreeTensor(aLocal);
        }

        // ����B��ÿ��λ�ã������λ�õ�Ԫ�ز���A�У����滻ΪexchangeBuffer�е�Ԫ��
        for (int64_t i = 0; i < this->topK && exchangeIdx < exchangeCount; ++i) {
            int32_t bVal = bLocal.GetValue(i);

            // ���bVal�Ƿ���A�У�ͨ��������������ΪA��GM�У�
            // �Ż���ʹ��CompareScalarһ���Լ��
            // ����������Ҫ��bVal��A�е�����Ԫ�رȽ�
            // ����A̫������ʹ���������Ƚ�
            bool foundInA = false;
            
            // ��A���ݼ��ص�UB���������Ƚ�
            for (int64_t tileStart = 0; tileStart < this->topK && !foundInA; tileStart += this->tileLength) {
                int64_t currentTileLength = this->topK - tileStart;
                if (currentTileLength > this->tileLength) {
                    currentTileLength = this->tileLength;
                }

                AscendC::LocalTensor<int32_t> aLocal = inQueueA.AllocTensor<int32_t>();
                AscendC::DataCopy(aLocal, aGm[bOffset + tileStart], currentTileLength);
                inQueueA.EnQue(aLocal);
                aLocal = inQueueA.DeQue<int32_t>();

                // ʹ��CompareScalar���bVal�Ƿ������tile��A��
                AscendC::CompareScalar(cmpResult, aLocal, bVal, AscendC::CMPMODE::EQ, currentTileLength);
                AscendC::Cast(castHalf, cmpResult, AscendC::RoundMode::CAST_NONE, (currentTileLength + 7) / 8);
                AscendC::ReduceMax<half>(reduceDst, castHalf, reduceTmp, (currentTileLength + 7) / 8, true);

                float maxVal = static_cast<float>(reduceDst.GetValue(0));
                if (maxVal > 0.0f) {
                    foundInA = true;
                }

                inQueueA.FreeTensor(aLocal);
            }

            // �������A�У��滻ΪA���е�Ԫ��
            if (!foundInA) {
                int32_t newVal = exchangeBuffer[exchangeIdx];
                outLocal.SetValue(i, newVal);
                exchangeIdx++;
            }
        }

        // ��������GM
        outQueueB.EnQue(outLocal);
        outLocal = outQueueB.DeQue<int32_t>();
        AscendC::DataCopy(outGm[bOffset], outLocal, this->topK);
        outQueueB.FreeTensor(outLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueA;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueB;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueB;
    AscendC::TBuf<AscendC::TPosition::VECCALC> cmpResultBuffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> castBuffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceDstBuffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceTmpBuffer;

    AscendC::GlobalTensor<int32_t> aGm;
    AscendC::GlobalTensor<int32_t> bGm;
    AscendC::GlobalTensor<int32_t> outGm;

    int64_t batchSize;
    int64_t topK;
    int64_t totalElements;
    int64_t formerNum;
    int64_t formerLength;
    int64_t tailNum;
    int64_t tailLength;
    int64_t tileLength;
    int64_t usedCoreNum;
};

extern "C" __global__ __aicore__ void set_diff_exchange(GM_ADDR a, GM_ADDR b, GM_ADDR out, GM_ADDR workspace,
                                                         int64_t batchSize, int64_t topK, int64_t totalElements,
                                                         int64_t formerNum, int64_t formerLength,
                                                         int64_t tailNum, int64_t tailLength,
                                                         int64_t tileLength)
{
    KernelSetDiffExchange op;
    op.Init(a, b, out, workspace, batchSize, topK, totalElements,
            formerNum, formerLength, tailNum, tailLength, tileLength);
    op.Process();
}

