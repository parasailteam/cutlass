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
  //overload call operator ()
  size_t order(dim3 grid, dim3 currTile) {
    return currTile.x * grid.y * grid.z + currTile.y * grid.z + currTile.z;
  }
};

struct RowSync {
  __device__ int wait(dim3 tile) {
    return tile.x;
  } 
};

template<typename Sched>
struct CuStage {
  dim3 grid_;
  dim3 tileSize_;
  uint* tileCounter;
  dim3* tileOrder;
  volatile uint* tileStatus_;
  int* kernelExecuted_;
  int iter;
  using Sync = RowSync;
  bool producerOrConsumer_;

  __device__ __host__ CuStage(): iter(0) {}

  __device__ __host__ CuStage(dim3 grid, dim3 tileSize) : grid_(grid), tileSize_(tileSize), iter(0) {}
  __host__ __device__ size_t numTiles() {return grid_.x * grid_.y * grid_.z;}

  void buildScheduleBuffer(volatile uint* tileStatus) {
    CUDA_CHECK(cudaMalloc(&tileCounter, sizeof(int)));
    CUDA_CHECK(cudaMemset(tileCounter, 0, sizeof(int)));
    printf("52: tileCounter %p\n", tileCounter);

    CUDA_CHECK(cudaMalloc(&tileOrder, sizeof(*tileOrder) * numTiles()));
    dim3* hTileOrder = new dim3[numTiles()];
  
    for (int x = 0; x < grid_.x; x++) {
    for (int y = 0; y < grid_.y; y++) {
    for (int z = 0; z < grid_.z; z++) {
      size_t id = RowMajor().order(grid_, {x, y, z});
      hTileOrder[id] = {x, y, z};
    }}}

    CUDA_CHECK(cudaMemcpy(tileOrder, hTileOrder, 
                          sizeof(*tileOrder) * numTiles(),
                          cudaMemcpyHostToDevice));
    delete[] hTileOrder;
    tileStatus_ = tileStatus;
  }

  __device__ void wait(dim3 tile, uint expectedInputStatusVal = 48) {
    if (threadIdx.x == 0) {
      uint linearTileIdx = Sync().wait(tile);
      // printf("%d iter %d expectedInputStatusVal %d blockIdx.x %d\n", linearTileIdx, iter, expectedInputStatusVal, tile.x);

      // printf("waitBuffer[%d] %d iter %d expectedInputStatusVal %d blockIdx.x %d\n", linearTileIdx, tileStatus[linearTileIdx], iter, expectedInputStatusVal, tile.x);
      while(tileStatus_[linearTileIdx] < iter * expectedInputStatusVal);
    }

    __syncthreads();
  }

  __device__ void post(dim3 tile, int value) {
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
      __threadfence_system();
      // uint linearTileIdx = xTileIdx*yMaxTiles + yTileIdx;
      uint linearTileIdx = Sync().wait(tile);
      atomicAdd((int*)&tileStatus_[linearTileIdx], value);
      
      // printf("tileStatus[%d] %d\n", linearTileIdx, tileStatus[linearTileIdx]);
      // tileStatusMap[linearTileIdx] = iter;
    }

    __syncwarp();
  }

  __device__ __host__ bool isProducer() {
    return producerOrConsumer_;
  }

  __device__ dim3 init() {
    
  }
  __device__ dim3 tile(dim3* shared_storage) {
    if (threadIdx.x == 0) {
      if (producerOrConsumer_) {
        if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
          *kernelExecuted_ = iter;
        }
      }
      // if (isProducerOrConsumer)
      // printf("stage.tileCounter %p stage.tileOrder %p stage.iter %d\n", stage.tileCounter, stage.tileOrder, stage.iter);   
      uint linear_id = atomicAdd(tileCounter, 1) - (iter-1)*numTiles();
      *shared_storage = tileOrder[linear_id];
    }

    __syncthreads();
    return *shared_storage;
  }
};

__device__ inline uint glLoad(volatile uint* addr) {
  uint val;
  asm ("ld.volatile.global.u32 {%0}, [%1];" : "=r"(val) : "l"(addr));
  return val;
}

__global__ void waitKernel(volatile uint* kernelExecuted, uint expectedValue) {
  if (threadIdx.x == 0) {
    uint v = glLoad(kernelExecuted);
    while(v < expectedValue) {
      v = glLoad(kernelExecuted);
    }
  }
}

// template<typename Sched1, typename Sched2, typename Sync>
struct CuSync {
  CuStage<RowMajor> prod_;
  __host__ __device__ CuStage<RowMajor>& prod() {return prod_;}
  CuStage<RowMajor> cons_;
  __host__ __device__ CuStage<RowMajor>& cons() {return cons_;}

  using Sync = RowSync;

  volatile uint* tileStatus;
  int* kernelExecuted;
  int iter;

  __device__ __host__ CuSync() {}

  void invokeWaitKernel(cudaStream_t stream) {
    waitKernel<<<1,1,0,stream>>>((uint*)kernelExecuted, prod().iter);
  }

  CuSync(CuStage<RowMajor> prod, CuStage<RowMajor> cons): prod_(prod), cons_(cons) {
    CUDA_CHECK(cudaMalloc(&tileStatus, prod.numTiles() * sizeof(int)));
    CUDA_CHECK(cudaMemset((uint*)tileStatus, 0, prod.numTiles() * sizeof(int)));
    iter = 0;
    prod_.buildScheduleBuffer(tileStatus);
    cons_.buildScheduleBuffer(tileStatus);
    prod_.producerOrConsumer_ = true;
    cons_.producerOrConsumer_ = false;
    CUDA_CHECK(cudaMalloc(&kernelExecuted, sizeof(int)));
    CUDA_CHECK(cudaMemset(kernelExecuted, 0, sizeof(int)));
    prod_.kernelExecuted_ = kernelExecuted;
  }

  DEVICE_FUNC HOST_FUNC bool isProducer() {return producerOrConsumer_;}
  DEVICE_FUNC HOST_FUNC bool isConsumer() {return !producerOrConsumer_;}

  uint waitValue;
  uint tileBatch; 

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