#include <assert.h>

#ifndef __OVERLAP_HANDLE__
#define __OVERLAP_HANDLE__

#define HOST_FUNC __host__
#define DEVICE_FUNC __device__

#define CUDA_CHECK(cmd) do {                        \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0);

template<typename T>
T divup(T x, T y) {
  return (x + y - 1)/y;
}

struct RowMajor {

};

template<typename Sched>
struct CuStage {
  dim3 grid_;
  dim3 tileSize_;
  __device__ __host__ CuStage() {}

  __device__ __host__ CuStage(dim3 grid, dim3 tileSize) : grid_(grid), tileSize_(tileSize) {}
  size_t numTiles() {return grid_.x * grid_.y * grid_.z;}
};

struct RowSync {
  __device__ int wait(dim3 tile) {
    return tile.x;
  } 
};

// template<typename Sched1, typename Sched2, typename Sync>
struct CuSync {
  CuStage<RowMajor> prod_;
  CuStage<RowMajor> cons_;
  using Sync = RowSync;

  volatile uint* tileStatus;
  int iter;

  __device__ __host__ CuSync() {}

  CuSync(CuStage<RowMajor> prod, CuStage<RowMajor> cons): prod_(prod), cons_(cons) {
    CUDA_CHECK(cudaMalloc(&tileStatus, prod.numTiles() * sizeof(int)));
    CUDA_CHECK(cudaMemset((uint*)tileStatus, 0, prod.numTiles() * sizeof(int)));
    iter = 1;
  }

  __device__ void wait(dim3 tile, uint expectedInputStatusVal) {
    if (threadIdx.x == 0) {
      uint linearTileIdx = Sync().wait(tile);
      // printf("waitBuffer[%d] %d iter %d expectedInputStatusVal %d blockIdx.x %d\n", linearTileIdx, waitBuffer[linearTileIdx], iter, expectedInputStatusVal, blockIdx.x);
      while(tileStatus[linearTileIdx + threadIdx.x] < iter * expectedInputStatusVal);
    }

    __syncthreads();
  }

  __device__ void post(dim3 tile, int value) {

  }
  
  DEVICE_FUNC HOST_FUNC bool isProducer() {return producerOrConsumer_;}
  DEVICE_FUNC HOST_FUNC bool isConsumer() {return !producerOrConsumer_;}

  uint waitValue;
  bool enable_;
  uint tileBatch; 

  int* isBlockRemaining;
  int* numProducerTBs;
  int* numConsumerTBs;
  int* blockIndexOrder;
  int* consumerBlockIndexOrder;
  //True for producer and false for consumer
  bool producerOrConsumer_;

  // DEVICE_FUNC HOST_FUNC OverlapHandle() : enable_(false), iter(0), xGridDim(0), yGridDim(0), zGridDim(0), tileBatch(1) {}

  // DEVICE_FUNC HOST_FUNC OverlapHandle(uint xSize_, uint ySize_, uint zSize_, 
  //                                     uint waitValue_) : 
  //   xSize(xSize_), ySize(ySize_), zSize(zSize_), 
  //   waitValue(waitValue_),
  //   enable_(true),
  //   producerOrConsumer_(false), iter(0), tileBatch(1)
  // {}

  // void setGridDims(uint xGridDim_, uint yGridDim_, uint zGridDim_) {
  //   xGridDim = xGridDim_;
  //   yGridDim = yGridDim_;
  //   zGridDim = zGridDim_;
  // }

  // HOST_FUNC bool validGridDims() {
  //   return xGridDim > 0 && yGridDim > 0 && zGridDim > 0;
  // }

  DEVICE_FUNC HOST_FUNC bool enable() {return enable_;}

  // HOST_FUNC cudaError_t clearTileStatusMap() {
  //   if (tileStatus == NULL) return cudaErrorInvalidValue;
  //   //TODO: a single function to get total size
  //   cudaError_t error = cudaMemset(tileStatus, 0, sizeof(int) * divup(xSize, xTile) * divup(ySize, yTile) * divup(zSize, zTile));
  //   return error;
  // }

  // HOST_FUNC cudaError_t allocTileStatusMap(uint xTile_, uint yTile_, uint zTile_) {
  //   xTile = xTile_;
  //   yTile = yTile_;
  //   zTile = zTile_;
    
  //   cudaError_t error;

  //   error = cudaMalloc(&tileStatusMap, sizeof(int) * divup(xSize, xTile) * divup(ySize, yTile) * divup(zSize, zTile));
  //   if (error != cudaSuccess) return error;

  //   return clearTileStatusMap();
  // }

  // DEVICE_FUNC int getLinearTileIdx(uint xTileIdx, uint yTileIdx, uint zTileIdx) {
  //   int xMaxTiles = xSize/xTile;
  //   int yMaxTiles = ySize/yTile;
  //   int zMaxTiles = zSize/zTile;
    
  //   assert(xTileIdx < xMaxTiles);
  //   assert(yTileIdx < yMaxTiles);
  //   assert(zTileIdx < zMaxTiles);

  //   int linearTileIdx = xTileIdx * yMaxTiles * zMaxTiles + yTileIdx * zMaxTiles;       

  //   return linearTileIdx;
  // }

  // #define batchTile 1
  // DEVICE_FUNC void waitOnTile(uint xTileIdx, uint yTileIdx, uint zTileIdx, uint expectedInputStatusVal, int threadId = 0) {
  //   volatile uint* waitBuffer = tileStatus;
  //   int yMaxTiles = ySize/yTile;
  //   uint linearTileIdx = xTileIdx*yMaxTiles + yTileIdx;//getLinearTileIdx(xTileIdx, yTileIdx, zTileIdx);
  //   if (linearTileIdx % batchTile != 0) return;
  //   if (threadIdx.x == threadId) {
  //     // if (linearTileIdx == 196)
  //     // printf("waitBuffer[%d] = %d ; %d xTileIdx %d yTileIdx %p iter %d expectedInputStatusVal %d\n", 
  //     //        linearTileIdx, waitBuffer[linearTileIdx], xTileIdx, yTileIdx, waitBuffer, iter, expectedInputStatusVal);
  //     while(waitBuffer[linearTileIdx/batchTile] < iter * expectedInputStatusVal * batchTile);
  //     // if (linearTileIdx == 196) 
  //     // printf("DONE: linearTileIdx %d\n", linearTileIdx);
  //   }
  //   // if (expectedInputStatusVal == 2) printf("114: threadIdx.x %d %d\n", threadIdx.x, waitBuffer[linearTileIdx]);
  //   __syncthreads();
  // }

  // DEVICE_FUNC void waitOnTilesWithSyncValue(uint xTileIdx, uint yTileIdx, uint zTileIdx, uint numTiles) {
  //   if (threadIdx.x < numTiles) {
  //     volatile uint* waitBuffer = tileStatus;

  //     uint linearTileIdx = getLinearTileIdx(xTileIdx, yTileIdx, zTileIdx);
  //     // printf("waitBuffer[%d] %d iter %d expectedInputStatusVal %d blockidx.x %d\n", 
  //     // linearTileIdx, waitBuffer[linearTileIdx], iter, waitValue, blockIdx.x);
  //     while(waitBuffer[linearTileIdx + threadIdx.x] < iter * waitValue);
  //   }

  //   __syncthreads();
  // }

  // DEVICE_FUNC void setRowStatus(uint xTileIdx, uint yTileIdx, uint zTileIdx, uint tileStatus, int blockIdx_x = 0, int blockIdx_y = 0) {
  //   __syncthreads();
  //   if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
  //     __threadfence_system();
  //     // uint linearTileIdx = xTileIdx*yMaxTiles + yTileIdx;
  //     uint linearTileIdx = getLinearTileIdx(xTileIdx, yTileIdx, zTileIdx);
  //     atomicAdd(&tileStatus[linearTileIdx], tileStatus);
      
  //     // printf("tileStatusmap[%d] %d xTileIdx %d yTileIdx %d blockIdx.x %d blockIdx.y %d\n", linearTileIdx, tileStatusMap[linearTileIdx], xTileIdx, yTileIdx, blockIdx.x, blockIdx.y);
  //     // tileStatusMap[linearTileIdx] = iter;
  //   }

  //   __syncwarp();
  // }

  // DEVICE_FUNC void setTiles(uint xTileIdx, uint yTileIdx, uint zTileIdx, uint tileStatus, int threadid = 0) {
  //   __syncthreads();
  //   if (threadIdx.x == threadid && threadIdx.y == 0 && threadIdx.z == 0) {
  //     __threadfence_system();
  //     int xMaxTiles = xSize/xTile;
  //     int yMaxTiles = ySize/yTile;
  //     int zMaxTiles = zSize/zTile;
  //     uint linearTileIdx = xTileIdx*yMaxTiles + yTileIdx;
  //     // uint linearTileIdx = getLinearTileIdx(xTileIdx, yTileIdx, zTileIdx);
  //     // printf("tileStatusMap[%d] %d xTileIdx %d yTileIdx %d tileStatus %d\n", linearTileIdx, tileStatusMap[linearTileIdx], xTileIdx, yTileIdx, tileStatus*iter);
  //     // atomicAdd(&tileStatusMap[linearTileIdx], 1);
  //     atomicAdd(&tileStatus[linearTileIdx], tileStatus);
  //   }

  //   __syncwarp();
  // }

  // DEVICE_FUNC void setTileStatus(uint xTileIdx, uint yTileIdx, uint zTileIdx, uint tileStatus, int threadid = 0) {
  //   __syncthreads();
  //   if (threadIdx.x == threadid && threadIdx.y == 0 && threadIdx.z == 0) {
  //     __threadfence_system();
  //     int xMaxTiles = xSize/xTile;
  //     int yMaxTiles = ySize/yTile;
  //     int zMaxTiles = zSize/zTile;
  //     uint linearTileIdx = xTileIdx*yMaxTiles + yTileIdx;
  //     // uint linearTileIdx = getLinearTileIdx(xTileIdx, yTileIdx, zTileIdx);
  //     // printf("tileStatusMap[%d] %d xTileIdx %d yTileIdx %d tileStatus %d\n", linearTileIdx, tileStatusMap[linearTileIdx], xTileIdx, yTileIdx, tileStatus*iter);
  //     atomicAdd(&tileStatus[linearTileIdx/batchTile], 1);
  //     // tileStatusMap[linearTileIdx] = tileStatus*iter;
  //   }

  //   __syncwarp();
  // }
};

#endif

// #include <assert.h>

// #ifndef __OVERLAP_HANDLE__
// #define __OVERLAP_HANDLE__

// #define HOST_FUNC __host__
// #define DEVICE_FUNC __device__

// // #define CUDA_CHECK(cmd) do {                        \
// //   cudaError_t e = cmd;                              \
// //   if( e != cudaSuccess ) {                          \
// //     printf("Failed: Cuda error %s:%d '%s'\n",       \
// //         __FILE__,__LINE__,cudaGetErrorString(e));   \
// //     exit(EXIT_FAILURE);                             \
// //   }                                                 \
// // } while(0);

// template<typename T>
// T divup(T x, T y) {
//   return (x + y - 1)/y;
// }

// struct OverlapHandle {
//   uint xSize;
//   uint ySize;
//   uint zSize;

//   uint xTile;
//   uint yTile;
//   uint zTile;

//   uint xGridDim;
//   uint yGridDim;
//   uint zGridDim;

//   uint* tileStatusMap;
//   uint waitValue;
//   uint iter;
//   bool enable_;
//   uint tileBatch; 

//   int* isBlockRemaining;
//   int* numProducerTBs;
//   int* numConsumerTBs;
//   int* blockIndexOrder;
//   int* consumerBlockIndexOrder;
//   //True for producer and false for consumer
//   bool producerOrConsumer_;

//   DEVICE_FUNC HOST_FUNC OverlapHandle() : enable_(false), iter(0), xGridDim(0), yGridDim(0), zGridDim(0), tileBatch(1) {}

//   DEVICE_FUNC HOST_FUNC OverlapHandle(uint xSize_, uint ySize_, uint zSize_, 
//                                       uint waitValue_) : 
//     xSize(xSize_), ySize(ySize_), zSize(zSize_), 
//     waitValue(waitValue_),
//     enable_(true),
//     producerOrConsumer_(false), iter(0), tileBatch(1)
//   {}

//   void setGridDims(uint xGridDim_, uint yGridDim_, uint zGridDim_) {
//     xGridDim = xGridDim_;
//     yGridDim = yGridDim_;
//     zGridDim = zGridDim_;
//   }

//   HOST_FUNC bool validGridDims() {
//     return xGridDim > 0 && yGridDim > 0 && zGridDim > 0;
//   }

//   DEVICE_FUNC HOST_FUNC bool enable() {return enable_;}

//   HOST_FUNC cudaError_t clearTileStatusMap() {
//     if (tileStatusMap == NULL) return cudaErrorInvalidValue;
//     //TODO: a single function to get total size
//     cudaError_t error = cudaMemset(tileStatusMap, 0, sizeof(int) * divup(xSize, xTile) * divup(ySize, yTile) * divup(zSize, zTile));
//     return error;
//   }

//   HOST_FUNC cudaError_t allocTileStatusMap(uint xTile_, uint yTile_, uint zTile_) {
//     xTile = xTile_;
//     yTile = yTile_;
//     zTile = zTile_;
    
//     cudaError_t error;

//     error = cudaMalloc(&tileStatusMap, sizeof(int) * divup(xSize, xTile) * divup(ySize, yTile) * divup(zSize, zTile));
//     if (error != cudaSuccess) return error;

//     return clearTileStatusMap();
//   }

//   DEVICE_FUNC int getLinearTileIdx(uint xTileIdx, uint yTileIdx, uint zTileIdx) {
//     int xMaxTiles = xSize/xTile;
//     int yMaxTiles = ySize/yTile;
//     int zMaxTiles = zSize/zTile;
    
//     assert(xTileIdx < xMaxTiles);
//     assert(yTileIdx < yMaxTiles);
//     assert(zTileIdx < zMaxTiles);

//     int linearTileIdx = xTileIdx * yMaxTiles * zMaxTiles + yTileIdx * zMaxTiles;       

//     return linearTileIdx;
//   }

//   #define batchTile 1
//   DEVICE_FUNC void waitOnTile(uint xTileIdx, uint yTileIdx, uint zTileIdx, uint expectedInputStatusVal, int threadId = 0) {
//     volatile uint* waitBuffer = tileStatusMap;
//     int yMaxTiles = ySize/yTile;
//     uint linearTileIdx = xTileIdx*yMaxTiles + yTileIdx;//getLinearTileIdx(xTileIdx, yTileIdx, zTileIdx);
//     if (linearTileIdx % batchTile != 0) return;
//     if (threadIdx.x == threadId) {
//       // if (linearTileIdx == 196)
//       // printf("waitBuffer[%d] = %d ; %d xTileIdx %d yTileIdx %p iter %d expectedInputStatusVal %d\n", 
//       //        linearTileIdx, waitBuffer[linearTileIdx], xTileIdx, yTileIdx, waitBuffer, iter, expectedInputStatusVal);
//       while(waitBuffer[linearTileIdx/batchTile] < iter * expectedInputStatusVal * batchTile);
//       // if (linearTileIdx == 196) 
//       // printf("DONE: linearTileIdx %d\n", linearTileIdx);
//     }
//     // if (expectedInputStatusVal == 2) printf("114: threadIdx.x %d %d\n", threadIdx.x, waitBuffer[linearTileIdx]);
//     __syncthreads();
//   }

//   DEVICE_FUNC void waitOnTilesWithSyncValue(uint xTileIdx, uint yTileIdx, uint zTileIdx, uint numTiles) {
//     if (threadIdx.x < numTiles) {
//       volatile uint* waitBuffer = tileStatusMap;

//       uint linearTileIdx = getLinearTileIdx(xTileIdx, yTileIdx, zTileIdx);
//       // printf("waitBuffer[%d] %d iter %d expectedInputStatusVal %d blockidx.x %d\n", 
//       // linearTileIdx, waitBuffer[linearTileIdx], iter, waitValue, blockIdx.x);
//       while(waitBuffer[linearTileIdx + threadIdx.x] < iter * waitValue);
//     }

//     __syncthreads();
//   }

//   DEVICE_FUNC void waitOnTiles(uint xTileIdx, uint yTileIdx, uint zTileIdx, uint numTiles, uint expectedInputStatusVal, int threadId = 0) {
//     if (threadIdx.x < numTiles) {
//       volatile uint* waitBuffer = tileStatusMap;

//       uint linearTileIdx = getLinearTileIdx(xTileIdx, yTileIdx, zTileIdx);
//       // printf("waitBuffer[%d] %d iter %d expectedInputStatusVal %d blockIdx.x %d\n", linearTileIdx, waitBuffer[linearTileIdx], iter, expectedInputStatusVal, blockIdx.x);
//       while(waitBuffer[linearTileIdx + threadIdx.x] < iter * expectedInputStatusVal);
//     }

//     __syncthreads();
//   }

//   DEVICE_FUNC void setRowStatus(uint xTileIdx, uint yTileIdx, uint zTileIdx, uint tileStatus, int blockIdx_x = 0, int blockIdx_y = 0) {
//     __syncthreads();
//     if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
//       __threadfence_system();
//       // uint linearTileIdx = xTileIdx*yMaxTiles + yTileIdx;
//       uint linearTileIdx = getLinearTileIdx(xTileIdx, yTileIdx, zTileIdx);
//       atomicAdd(&tileStatusMap[linearTileIdx], tileStatus);
      
//       // printf("tileStatusmap[%d] %d xTileIdx %d yTileIdx %d blockIdx.x %d blockIdx.y %d\n", linearTileIdx, tileStatusMap[linearTileIdx], xTileIdx, yTileIdx, blockIdx.x, blockIdx.y);
//       // tileStatusMap[linearTileIdx] = iter;
//     }

//     __syncwarp();
//   }

//   DEVICE_FUNC void setTiles(uint xTileIdx, uint yTileIdx, uint zTileIdx, uint tileStatus, int threadid = 0) {
//     __syncthreads();
//     if (threadIdx.x == threadid && threadIdx.y == 0 && threadIdx.z == 0) {
//       __threadfence_system();
//       int xMaxTiles = xSize/xTile;
//       int yMaxTiles = ySize/yTile;
//       int zMaxTiles = zSize/zTile;
//       uint linearTileIdx = xTileIdx*yMaxTiles + yTileIdx;
//       // uint linearTileIdx = getLinearTileIdx(xTileIdx, yTileIdx, zTileIdx);
//       // printf("tileStatusMap[%d] %d xTileIdx %d yTileIdx %d tileStatus %d\n", linearTileIdx, tileStatusMap[linearTileIdx], xTileIdx, yTileIdx, tileStatus*iter);
//       // atomicAdd(&tileStatusMap[linearTileIdx], 1);
//       atomicAdd(&tileStatusMap[linearTileIdx], tileStatus);
//     }

//     __syncwarp();
//   }

//   DEVICE_FUNC void setTileStatus(uint xTileIdx, uint yTileIdx, uint zTileIdx, uint tileStatus, int threadid = 0) {
//     __syncthreads();
//     if (threadIdx.x == threadid && threadIdx.y == 0 && threadIdx.z == 0) {
//       __threadfence_system();
//       int xMaxTiles = xSize/xTile;
//       int yMaxTiles = ySize/yTile;
//       int zMaxTiles = zSize/zTile;
//       uint linearTileIdx = xTileIdx*yMaxTiles + yTileIdx;
//       // uint linearTileIdx = getLinearTileIdx(xTileIdx, yTileIdx, zTileIdx);
//       // printf("tileStatusMap[%d] %d xTileIdx %d yTileIdx %d tileStatus %d\n", linearTileIdx, tileStatusMap[linearTileIdx], xTileIdx, yTileIdx, tileStatus*iter);
//       atomicAdd(&tileStatusMap[linearTileIdx/batchTile], 1);
//       // tileStatusMap[linearTileIdx] = tileStatus*iter;
//     }

//     __syncwarp();
//   }

//   DEVICE_FUNC HOST_FUNC bool isProducer() {return producerOrConsumer_;}
//   DEVICE_FUNC HOST_FUNC bool isConsumer() {return !producerOrConsumer_;}
// };

// #endif