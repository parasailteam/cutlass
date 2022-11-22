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

struct OverlapHandle {
  int xSize;
  int ySize;
  int zSize;

  int xTile;
  int yTile;
  int zTile;

  int* tileStatusMap;
  int expectedInputStatusVal;
  int iter;
  bool enable_;
  //True for producer and false for consumer
  bool producerOrConsumer_;

  DEVICE_FUNC HOST_FUNC OverlapHandle() : enable_(false), iter(0) {}

  DEVICE_FUNC HOST_FUNC OverlapHandle(int xSize_, int ySize_, int zSize_, 
                                      int expectedInputStatusVal_) : 
    xSize(xSize_), ySize(ySize_), zSize(zSize_), 
    expectedInputStatusVal(expectedInputStatusVal_),
    enable_(true),
    producerOrConsumer_(false), iter(0)
  {}

  DEVICE_FUNC bool enable() {return enable_;}

  HOST_FUNC cudaError_t clearTileStatusMap() {
    if (tileStatusMap == NULL) return cudaErrorInvalidValue;

    cudaError_t error = cudaMemset(tileStatusMap, 0, sizeof(int) * (xSize/xTile) * (ySize/yTile) * (zSize/zTile));
    return error;
  }

  HOST_FUNC cudaError_t allocTileStatusMap(int xTile_, int yTile_, int zTile_) {
    xTile = xTile_;
    yTile = yTile_;
    zTile = zTile_;
    
    cudaError_t error;

    error = cudaMalloc(&tileStatusMap, sizeof(int) * (xSize/xTile) * (ySize/yTile) * (zSize/zTile));
    if (error != cudaSuccess) return error;

    return clearTileStatusMap();
  }

  DEVICE_FUNC int getLinearTileIdx(int xTileIdx, int yTileIdx, int zTileIdx) {
    int xMaxTiles = xSize/xTile;
    int yMaxTiles = ySize/yTile;
    int zMaxTiles = zSize/zTile;
    
    assert(xTileIdx < xMaxTiles);
    assert(yTileIdx < yMaxTiles);
    assert(zTileIdx < zMaxTiles);

    int linearTileIdx = xTileIdx * yMaxTiles * zMaxTiles + yTileIdx * zMaxTiles;       

    return linearTileIdx;
  }

  DEVICE_FUNC void waitOnTile(int xTileIdx, int yTileIdx, int zTileIdx, int expectedInputStatusVal) {
    if (threadIdx.x == 0) {
      volatile int* waitBuffer = tileStatusMap;

      int linearTileIdx = getLinearTileIdx(xTileIdx, yTileIdx, zTileIdx);
      // printf("waitBuffer[linearTileIdx] %d iter %d expectedInputStatusVal\n", waitBuffer[linearTileIdx], iter, expectedInputStatusVal);
      while(waitBuffer[linearTileIdx] < iter * expectedInputStatusVal);
    }

    __syncthreads();
  }

  DEVICE_FUNC void setTileStatus(int xTileIdx, int yTileIdx, int zTileIdx, int tileStatus) {
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
      int linearTileIdx = getLinearTileIdx(xTileIdx, yTileIdx, zTileIdx);
      // printf("tileStatusMap[linearTileIdx] %d\n", tileStatusMap[linearTileIdx]);
      tileStatusMap[linearTileIdx] = iter * tileStatus;
    }

    __syncwarp();
  }

  DEVICE_FUNC HOST_FUNC bool isProducer() {return producerOrConsumer_;}
  DEVICE_FUNC HOST_FUNC bool isConsumer() {return !producerOrConsumer_;}
};

#endif