#include <assert.h>
#include <stdio.h>

#ifndef __CUSYNC__
#define __CUSYNC__

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

template<typename Sched, typename Sync> struct CuStage;
//todo: make args constant

struct RowSync {
  uint waitValue_;
  uint postValue_;
  __device__ __host__ uint waitValue() {return waitValue_;}
  __device__ __host__ uint postValue() {return postValue_;}
  __device__ __host__ RowSync()  : waitValue_(0), postValue_(0) {}
  __device__ __host__ RowSync(uint waitValue, uint postValue) : waitValue_(waitValue), postValue_(postValue) {}
  
  __device__ uint waitValue(dim3 tile, dim3 grid) {
    return waitValue_;
  }

  template<typename Sched, typename Sync>
  __device__ void wait(CuStage<Sched, Sync>& stage, dim3& tile, dim3& grid) {
    if (tile.y != 0) return;
    stage.waitUntil(tile.x, waitValue());
  }

  template<typename Sched, typename Sync>
  __device__ void post(CuStage<Sched, Sync>& stage, dim3& tile, dim3& grid) {
    stage.increment(tile.x, postValue());
  }

  __device__ uint postValue(dim3& tile, dim3& grid) {
    return 1;
  }
};

struct TileSync {
  __device__ __host__ TileSync() {}

  template<typename Sched, typename Sync>
  __device__ void wait(CuStage<Sched, Sync>& stage, dim3 tile, dim3 grid) {
    stage.waitUntil(tile.y * grid.x + tile.x, 1);
  }

  template<typename Sched, typename Sync>
  __device__ void post(CuStage<Sched, Sync>& stage, dim3 tile, dim3 grid) {
    stage.increment(tile.y * grid.x + tile.x, 1);
  }
};

template<typename Sched, typename Sync>
struct CuStage {
  dim3 grid_;
  dim3 prodGrid_;
  dim3 tileSize_;
  uint* tileCounter;
  dim3* tileOrder;
  volatile uint* tileStatus_;
  int* kernelExecuted_;
  int iter;
  bool producerOrConsumer_;
  Sync syncPolicy_;

  __device__ __host__ CuStage(): iter(0) {}

  __device__ __host__ CuStage(dim3 grid, dim3 tileSize, Sync syncPolicy) : 
    grid_(grid), tileSize_(tileSize), iter(0), prodGrid_(0), syncPolicy_(syncPolicy) {}
  __host__ __device__ size_t numTiles() {return grid_.x * grid_.y * grid_.z;}

  void buildScheduleBuffer(volatile uint* tileStatus) {
    CUDA_CHECK(cudaMalloc(&tileCounter, sizeof(int)));
    CUDA_CHECK(cudaMemset(tileCounter, 0, sizeof(int)));

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

  __device__ void waitUntil(uint tileIdx, uint value) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        // // printf("%d iter %d expectedInputStatusVal %d blockIdx.x %d\n", linearTileIdx, iter, expectedInputStatusVal, tile.x);
        // // printf("waitBuffer[%d] %d iter %d expectedInputStatusVal %d blockIdx.x %d\n", linearTileIdx, tileStatus[linearTileIdx], iter, expectedInputStatusVal, tile.x);
        // printf("119: tileIdx %d tileStatus_[tileIdx] %d \n", tileIdx, tileStatus_[tileIdx]);
        while(tileStatus_[tileIdx] < iter * value);
    }
    
    __syncthreads();
  }

  __device__ void increment(uint tileIdx, uint value) {
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
      __threadfence_system();
      atomicAdd((int*)&tileStatus_[tileIdx], value);
    }

    __syncwarp();
  }

  __device__ void wait(dim3& tile) {
    if (isProducer()) return;
    syncPolicy_.wait(*this, tile, prodGrid_);
  }

  __device__ void post(dim3& tile) {
    // if (!isProducer()) return;
    syncPolicy_.post(*this, tile, grid_);
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

template<typename Sched1, typename Sched2, typename Sync>
struct CuSync {
  CuStage<Sched1, Sync> prod_;
  __host__ __device__ CuStage<Sched1, Sync>& prod() {return prod_;}
  CuStage<Sched2, Sync> cons_;
  __host__ __device__ CuStage<Sched2, Sync>& cons() {return cons_;}

  volatile uint* tileStatus;
  int* kernelExecuted;
  int iter;

  __device__ __host__ CuSync() {}

  void invokeWaitKernel(cudaStream_t stream) {
    waitKernel<<<1,1,0,stream>>>((uint*)kernelExecuted, prod().iter);
  }

  CuSync(CuStage<Sched1, Sync> prod, CuStage<Sched2, Sync> cons): prod_(prod), cons_(cons) {
    CUDA_CHECK(cudaMalloc(&tileStatus, prod.numTiles() * sizeof(int)));
    CUDA_CHECK(cudaMemset((uint*)tileStatus, 0, prod.numTiles() * sizeof(int)));
    iter = 0;
    prod_.buildScheduleBuffer(tileStatus);
    cons_.buildScheduleBuffer(tileStatus);
    cons_.prodGrid_ = prod.grid_;
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
};

#endif