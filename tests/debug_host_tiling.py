import torch

def debug_tiling(batchSize, topK, coreNum=48, ubSize=192*1024):
    totalElements = batchSize * topK
    dtypeSize = 4  # int32

    # CompareScalar对齐要求 (int32=4字节, 256字节对齐 -> count必须是64的倍数)
    alignedTopK = ((topK + 63) // 64) * 64

    # Block级Tiling
    elementsPerCore = (totalElements + coreNum - 1) // coreNum
    batchesPerCore = (elementsPerCore + topK - 1) // topK
    alignedElementsPerCore = batchesPerCore * topK

    # Cache Line对齐 (512 bytes)
    alignElements = 512 // dtypeSize
    alignedElementsPerCore = ((alignedElementsPerCore + alignElements - 1) // alignElements) * alignElements

    usedCoreNum = (totalElements + alignedElementsPerCore - 1) // alignedElementsPerCore
    formerNum = usedCoreNum - 1
    tailNum = 1
    formerLength = alignedElementsPerCore
    tailLength = totalElements - formerNum * formerLength

    # UB级Tiling (B buffer按alignedTopK分配)
    bufferCoefficient = 10
    maxTileElements = (ubSize - alignedTopK * dtypeSize - 256) // dtypeSize // bufferCoefficient
    ubAlignElements = 32 // dtypeSize
    tileLength = (maxTileElements // ubAlignElements) * ubAlignElements

    if tileLength > topK:
        tileLength = topK
    if tileLength < ubAlignElements:
        tileLength = ubAlignElements

    print(f"=== batchSize={batchSize}, topK={topK} ===")
    print(f"totalElements={totalElements}")
    print(f"elementsPerCore={elementsPerCore}, batchesPerCore={batchesPerCore}")
    print(f"alignedElementsPerCore={alignedElementsPerCore}")
    print(f"usedCoreNum={usedCoreNum}, formerNum={formerNum}, tailNum={tailNum}")
    print(f"formerLength={formerLength}, tailLength={tailLength}")
    print(f"tileLength={tileLength}")
    print()

    # Simulate what each core does
    for blockIdx in range(usedCoreNum):
        blockLength = tailLength if blockIdx == usedCoreNum - 1 else formerLength
        startElement = blockIdx * formerLength
        endElement = startElement + blockLength

        if startElement >= totalElements:
            print(f"  blockIdx={blockIdx}: SKIPPED (startElement={startElement} >= totalElements)")
            continue

        startBatch = startElement // topK
        endBatch = (endElement - 1) // topK

        print(f"  blockIdx={blockIdx}: startElement={startElement}, endElement={endElement}")
        print(f"    startBatch={startBatch}, endBatch={endBatch}")

        for batch in range(startBatch, endBatch + 1):
            bOffset = batch * topK
            aStart = (startElement % topK) if batch == startBatch else 0
            aEnd = ((endElement - 1) % topK + 1) if batch == endBatch else topK
            print(f"    batch={batch}: bOffset={bOffset}, aStart={aStart}, aEnd={aEnd}")
    print()

debug_tiling(2, 3)  # Small case that fails
debug_tiling(4, 2048)  # Large case that passes
