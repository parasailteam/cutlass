#include <assert.h>

#ifndef __OVERLAP_HANDLE__
#define __OVERLAP_HANDLE__

#define HOST_FUNC __host__
#define DEVICE_FUNC __device__

// #define CUDA_CHECK(cmd) do {                        \
//   cudaError_t e = cmd;                              \
//   if( e != cudaSuccess ) {                          \
//     printf("Failed: Cuda error %s:%d '%s'\n",       \
//         __FILE__,__LINE__,cudaGetErrorString(e));   \
//     exit(EXIT_FAILURE);                             \
//   }                                                 \
// } while(0);

template<typename T>
T divup(T x, T y) {
  return (x + y - 1)/y;
}

struct OverlapHandle {
  uint xSize;
  uint ySize;
  uint zSize;

  uint xTile;
  uint yTile;
  uint zTile;

  uint xGridDim;
  uint yGridDim;
  uint zGridDim;

  uint* tileStatusMap;
  uint expectedInputStatusVal;
  uint iter;
  bool enable_;

  int* isBlockRemaining;
  int* numProducerTBs;
  int* numConsumerTBs;
  int* blockIndexOrder;
  int* consumerBlockIndexOrder;
  //True for producer and false for consumer
  bool producerOrConsumer_;

  DEVICE_FUNC HOST_FUNC OverlapHandle() : enable_(false), iter(0), xGridDim(0), yGridDim(0), zGridDim(0) {}

  DEVICE_FUNC HOST_FUNC OverlapHandle(uint xSize_, uint ySize_, uint zSize_, 
                                      uint expectedInputStatusVal_) : 
    xSize(xSize_), ySize(ySize_), zSize(zSize_), 
    expectedInputStatusVal(expectedInputStatusVal_),
    enable_(true),
    producerOrConsumer_(false), iter(0)
  {}

  void setGridDims(uint xGridDim_, uint yGridDim_, uint zGridDim_) {
    xGridDim = xGridDim_;
    yGridDim = yGridDim_;
    zGridDim = zGridDim_;
  }

  HOST_FUNC bool validGridDims() {
    return xGridDim > 0 && yGridDim > 0 && zGridDim > 0;
  }

  DEVICE_FUNC HOST_FUNC bool enable() {return enable_;}

  HOST_FUNC cudaError_t clearTileStatusMap() {
    if (tileStatusMap == NULL) return cudaErrorInvalidValue;
    //TODO: a single function to get total size
    cudaError_t error = cudaMemset(tileStatusMap, 0, sizeof(int) * divup(xSize, xTile) * divup(ySize, yTile) * divup(zSize, zTile));
    return error;
  }

  HOST_FUNC cudaError_t allocTileStatusMap(uint xTile_, uint yTile_, uint zTile_) {
    xTile = xTile_;
    yTile = yTile_;
    zTile = zTile_;
    
    cudaError_t error;

    error = cudaMalloc(&tileStatusMap, sizeof(int) * divup(xSize, xTile) * divup(ySize, yTile) * divup(zSize, zTile));
    if (error != cudaSuccess) return error;

    return clearTileStatusMap();
  }

  DEVICE_FUNC int getLinearTileIdx(uint xTileIdx, uint yTileIdx, uint zTileIdx) {
    int xMaxTiles = xSize/xTile;
    int yMaxTiles = ySize/yTile;
    int zMaxTiles = zSize/zTile;
    
    assert(xTileIdx < xMaxTiles);
    assert(yTileIdx < yMaxTiles);
    assert(zTileIdx < zMaxTiles);

    int linearTileIdx = xTileIdx * yMaxTiles * zMaxTiles + yTileIdx * zMaxTiles;       

    return linearTileIdx;
  }

  DEVICE_FUNC void waitOnTile(uint xTileIdx, uint yTileIdx, uint zTileIdx, uint expectedInputStatusVal) {
    if (threadIdx.x == 0) {
      volatile uint* waitBuffer = tileStatusMap;
      uint linearTileIdx = getLinearTileIdx(xTileIdx, yTileIdx, zTileIdx);
      // printf("waitBuffer[%d] %d iter %d expectedInputStatusVal\n", linearTileIdx, waitBuffer[linearTileIdx], iter, expectedInputStatusVal);
      while(waitBuffer[linearTileIdx] < iter * expectedInputStatusVal);
    }

    __syncthreads();
  }

  DEVICE_FUNC void waitOnTiles(uint xTileIdx, uint yTileIdx, uint zTileIdx, uint numTiles, uint expectedInputStatusVal) {
    if (threadIdx.x < numTiles) {
      volatile uint* waitBuffer = tileStatusMap;

      uint linearTileIdx = getLinearTileIdx(xTileIdx, yTileIdx, zTileIdx);
      // printf("waitBuffer[linearTileIdx] %d iter %d expectedInputStatusVal\n", waitBuffer[linearTileIdx], iter, expectedInputStatusVal);
      while(waitBuffer[linearTileIdx + threadIdx.x] < iter * expectedInputStatusVal);
    }

    __syncthreads();
  }

  DEVICE_FUNC void setRowStatus(uint xTileIdx, uint yTileIdx, uint zTileIdx, uint tileStatus) {
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
      // uint linearTileIdx = xTileIdx*yMaxTiles + yTileIdx;
      uint linearTileIdx = getLinearTileIdx(xTileIdx, yTileIdx, zTileIdx);
      // printf("tileStatusMap[%d] %d xTileIdx %d yTileIdx %d\n", linearTileIdx, tileStatusMap[linearTileIdx], xTileIdx, yTileIdx);
      atomicAdd(&tileStatusMap[linearTileIdx], 1);
      // tileStatusMap[linearTileIdx] = iter;
    }

    __syncwarp();
  }

  DEVICE_FUNC void setTileStatus(uint xTileIdx, uint yTileIdx, uint zTileIdx, uint tileStatus) {
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
      int xMaxTiles = xSize/xTile;
      int yMaxTiles = ySize/yTile;
      int zMaxTiles = zSize/zTile;
      uint linearTileIdx = xTileIdx*yMaxTiles + yTileIdx;
      // uint linearTileIdx = getLinearTileIdx(xTileIdx, yTileIdx, zTileIdx);
      // printf("tileStatusMap[%d] %d xTileIdx %d yTileIdx %d\n", linearTileIdx, tileStatusMap[linearTileIdx], xTileIdx, yTileIdx);
      // atomicAdd(&tileStatusMap[linearTileIdx], 1);
      tileStatusMap[linearTileIdx] = iter;
    }

    __syncwarp();
  }

  DEVICE_FUNC HOST_FUNC bool isProducer() {return producerOrConsumer_;}
  DEVICE_FUNC HOST_FUNC bool isConsumer() {return !producerOrConsumer_;}
};

#endif