/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/**
This example shows how to run matrix multiplication kernels using functions and data structures
provided by CUTLASS using tensor cores; which we run on a NVIDIA Volta GPU.

Writing a single high performance matrix multiplication kernel is hard but do-able. Whereas writing
high performance kernels at scale which works for multiple problem sizes with good abstractions is
really hard. CUTLASS solves this problem by providing simplified abstractions to compose
multiple sections of gemm kernel. When used properly, the kernels can hit peak performance of GPU
easily.

CUTLASS divides a kernel into hierarchical composable sections. Which means, at each thread, warp
and thread-block level, they compute on their own tile-size with higher level of tile sizes being
composed from lower level ones. Multiple thread-tiles (tile size each thread computes) can be used
to form warp-tiles (tile size each warp computes) and multiple warp tiles can be used to compute
threadblock-tile (tile size computed by a threadblock).

In thie example, we split variable initialization into
1. Setting up data properties : describes how matrices are laid out in the memory and how the kernel
can view them (logical to physical mapping)
2. Setting up computation properties : describes how the above set matrices will be used to compute
output of matrix multiplication.

First, we setup the data types of matrices A, B, C and D along with alpha, beta as the equation for
GEMM is D = alpha * A * B + beta * C. In CUTLASS, the kernels first compute A * B and leaves the
rest of the computation to end of the kernel as alpha * X + beta * C is a simple element-wise
operation on X (A * B) and C. We call this as epilogue of kernel. Hence, we setup data types for
alpha and beta to be equal to ElementComputeEpilogue = float. As we want to MMA instructions on
Volta and they support only half-precision floating point (fp16 or half), we use data type for
elements in input matrix A and B as cutlass::half_t. Volta also supports accumulation of partial dot
product to fp32, which can store wider range of numbers, we use it as data type of output matrix
elements and accumulation. We convey this to CUTLASS kernel by initializing template variables
ElementAccumulator (float), ElementComputeEpilogue (float), ElementInputA (cutlass::half_t),
ElementInputB (cutlass::half_t), ElementOutput (float). Communicating just the data type is not
enough. As the data is laid out linearly in memory, we have to convey the layout of matrices. We do
that by initializing template variable LayoutInputA to column major cutlass variable, LayoutInputB
to row major and LayoutOutput to row major. Next, we setup rules to comptue alpha * X + beta * C
which is called epilogue of the kernel. We initialize template variable EpilogueOp, which takes the
data type of output ElementOutput (int32_t), the number of elements per vector memory access (16),
data type of accumulator (int32_t) and data type of computation of linear combination (alpha * X +
beta * C).

Now that we setup the properties of data, we have to setup properties of computation.

Second, we create template variables of tile sizes for thread-block, warp and mma-op to 128x128x32,
64x64x32, 8x8x4 (MxNxK) respectively. When passed to instantiate CUTLASS GEMM kernel, it internally
deduce the amount of threads needed per thread-block, amount of shared memory, storing data in
bank-conflict free manner, and ton of other variables required to compose, intialize and launch a
high performance GEMM kernel. This is the beauty of CUTLASS, it relieves developer from
understanding and coding complicated hardware optimizations which can easily go wrong.

CUTLASS also supports multiple MMA pipelines in a CTA. What are MMA pipelines? MMA pipelines
constitute the whole process of loading input data from global memory to shared memory, loading data
from shared memory to registers, doing matrix multiplication, store to global memory. The below flow
sequence shows a typical mma pipeline.

matrix in global memory -> registers -> tile in shared memory -> registers -> mma -> registers ->
output to global memory

The problem with single pipeline is, each stage is synchronous which means, each stage has to wait
until the previous finished executing. There are stages in the pipeline which do not have fixed
latency, for example, the loads from global memory and shared memory. Therefore, we can add one more
pipeline with a phase shift in mma kernel to hide latency from global and shared memory loads.
Finally, the pipeline in a kernel looks like

(1) matrix in global memory -> (2) registers -> (3) tile in shared memory -> (4) registers -> (5)
mma -> (6) registers -> (7) output to global memory (1) <null> -> (2) <null> -> (3) matrix in global
memory -> (4) registers -> (5) tile in shared memory -> (6) registers -> (7) mma -> (8) registers ->
(9) output to global memory

This way, you can hide the second global memoroy load latency by doing computation on already loaded
input data.

There are few more template variables initialized such as, which threadblock tile of output matrix
is done which threadblock launched on an SM, CUDA SM architecture of GPU you want to run on.

These are all put together to create a template variable which describes CUTLASS GEMM kernel using
cutlass::gemm::device::Gemm template.

The next step is to intialize physical data, instantiate and initialize CUTLASS kernel and run it.
We use CUTLASS utilities to initialize, fill, compare matrices as they are simple and doesn't come
in the way of learning CUTLASS.

Once all the matrices are initialized and filled with data, create arguments tuple to launch CUTLASS
kernel which takes problem size (M = 5120, N = 4096 and K = 4096), matrices, alpha, beta and the
important one, split k-dimension factor. Along with that, we query CUTLASS if any scratch-space
memory required by the kernel we instantiated. If yes, we create it and pass it along with other
arguments created to intialize CUTLASS kernel then, the kernel is launched.

In this example, we later on launch a reference gemm kernel (from CUTLASS utilities) to compare if
the output from CUTLASS kernel is same as reference GEMM kernel.
*/

#include<cuSync.h>

#ifdef ROWSYNC 
  using ProdCuStage = CuStage<CuStageType::Producer, RowMajor, RowSync>;
  using MiddleCuStage = CuStage<CuStageType::Producer|CuStageType::Consumer, RowMajor, RowSync>;
  using ConsCuStage = CuStage<CuStageType::Consumer, RowMajor, RowSync>;
  using Sync = RowSync;
#elif TILESYNC
  using ProdCuStage = CuStage<CuStageType::Producer, RowMajor, TileSync>;
  using MiddleCuStage = CuStage<CuStageType::Producer|CuStageType::Consumer, RowMajor, TileSync>;
  using ConsCuStage = CuStage<CuStageType::Consumer, RowMajor, TileSync>;
  using Sync = TileSync;
#else
  #error "Unknown Synchronization"
#endif 

using CuSyncImpl1 = CuSync<ProdCuStage, MiddleCuStage>;
using CuSyncImpl2 = CuSync<MiddleCuStage, ConsCuStage>;

#include "common.h"

#ifndef EVAL_TILE_SIZES
//Tile sizes of all GeMMs
using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256, 128, 32>;  
using ShapeMMAWarp = cutlass::gemm::GemmShape<128, 64, 32>;
#else
//<eval tiles>
using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256, 128, 32>;  
using ShapeMMAWarp = cutlass::gemm::GemmShape<128, 64, 32>;
//</eval tiles>
#endif

using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;  

const int SoftmaxRowTile = 4;

//Element types of A, B, and C
using ElementAccumulator = float;
using ElementInputA = cutlass::half_t;
using ElementInputB = cutlass::half_t;
using ElementOutput = cutlass::half_t;
using ElementComputeEpilogue = cutlass::half_t;

//All matrices are in RowMajor
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;

//Use FP-16 Tensor Cores
using MMAOp = cutlass::arch::OpClassTensorOp;

using SmArch = cutlass::arch::Sm70;

//Second GeMM in MLP performs no extra fused computations 
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                        
    128 / cutlass::sizeof_bits<ElementOutput>::value,     
    ElementAccumulator,
    ElementComputeEpilogue>;

template<bool splitK>
class BaseMLPGemm : public cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, 
                                                        ElementInputB, LayoutInputB,
                                                        ElementOutput, LayoutOutput,
                                                        ElementAccumulator, MMAOp,
                                                        SmArch, ShapeMMAThreadBlock,
                                                        ShapeMMAWarp, ShapeMMAOp,
                                                        EpilogueOp, 
                                                        cutlass::gemm::threadblock::GemmHorizontalThreadblockSwizzle, 
                                                        2, 8, 8, splitK> {};

// Baseline GeMMs
using Gemm1 = BaseMLPGemm<false>;
using Gemm2 = BaseMLPGemm<false>;

//Baseline GeMMs with SplitK enabled
using GemmSplitK1 = BaseMLPGemm<true>;
using GemmSplitK2 = BaseMLPGemm<true>;

//CuSync GeMMs
template<typename CuStage, bool splitK>
class CuSyncAttentionGemm : public cutlass::gemm::device::CuSyncGemm<CuStage, 
                                                        ElementInputA, LayoutInputA, 
                                                        ElementInputB, LayoutInputB,
                                                        ElementOutput, LayoutOutput,
                                                        ElementAccumulator, MMAOp,
                                                        SmArch, ShapeMMAThreadBlock,
                                                        ShapeMMAWarp, ShapeMMAOp,
                                                        EpilogueOp, 
                                                        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 
                                                        2, 8, 8, splitK> {};

using CuSyncGemm1 = CuSyncAttentionGemm<ProdCuStage, false>;
using CuSyncGemm2 = CuSyncAttentionGemm<ConsCuStage, false>;

using CuSyncGemmSplitK1 = CuSyncAttentionGemm<ProdCuStage, true>;
using CuSyncGemmSplitK2 = CuSyncAttentionGemm<ConsCuStage, true>;

using HostTensor = cutlass::HostTensor<ElementInputA, LayoutInputA>;

struct AttentionParams {
  HostTensor x;
  HostTensor qkv;
  HostTensor xqkv;
  HostTensor xdot;
  HostTensor w2;
  HostTensor xw12;

  HostTensor ref_xqkv;
  HostTensor ref_xdot;
  HostTensor ref_xw12;

  cutlass::gemm::GemmCoord gemm_size1, gemm_size2;
  curandState* randStates;
  bool refCheck;
  ElementComputeEpilogue alpha;
  ElementComputeEpilogue beta;

  AttentionParams(int problem[4], bool check) {
    gemm_size1 = cutlass::gemm::GemmCoord(problem[0], problem[1] * 3, problem[2]);
    gemm_size2 = cutlass::gemm::GemmCoord(problem[0], problem[3], problem[1]);
    alpha = ElementComputeEpilogue(1);
    beta = ElementComputeEpilogue(0);
  
    x    = HostTensor(gemm_size1.mk());
    qkv  = HostTensor(gemm_size1.kn());
    xqkv = HostTensor(gemm_size1.mn());
    xdot = HostTensor({gemm_size1.m(), gemm_size1.n()/3});
    w2   = HostTensor(gemm_size2.kn());
    xw12 = HostTensor(gemm_size2.mn());

    ref_xdot = HostTensor({gemm_size1.m(), gemm_size1.n()/3});
    ref_xqkv = HostTensor(gemm_size1.mn());
    ref_xw12 = HostTensor(gemm_size2.mn());

    size_t numRandStates = gemm_size1.m() * 1024;
    CUDA_CHECK(cudaMalloc(&randStates, sizeof(curandState)*(numRandStates)));
    init_curand_states<<<numRandStates/128, 128>>>(randStates, numRandStates);
    CUDA_CHECK(cudaDeviceSynchronize());
    refCheck = check;
  }

  void initIns() {
    if (refCheck) {
      memset_random2(x.host_data(), ElementOutput(0.02), 
                     ElementOutput(0.03), x.size());
      memset_random2(qkv.host_data(), ElementOutput(0.01), 
                     ElementOutput(0.035), qkv.size());
      memset_random2(w2.host_data(), ElementOutput(0.01),
                     ElementOutput(0.05), w2.size());
    } else {
      cutlass::reference::host::TensorFill(x.host_view(),
                                           ElementOutput(0.05));
      cutlass::reference::host::TensorFill(qkv.host_view(),
                                           ElementOutput(0.5));
      cutlass::reference::host::TensorFill(w2.host_view(),
                                           ElementOutput(0.01));
    }

    // Copy data from host to GPU
    x.sync_device();
    qkv.sync_device();
    w2.sync_device();
  }

  void initOuts() {
    //Zeros all output tensors
    cutlass::reference::host::TensorFill(xqkv.host_view());
    cutlass::reference::host::TensorFill(xw12.host_view());
    cutlass::reference::host::TensorFill(xdot.host_view());
  }

  void initRefs() {
    cutlass::reference::host::TensorFill(ref_xqkv.host_view());
    cutlass::reference::host::TensorFill(ref_xdot.host_view());
    cutlass::reference::host::TensorFill(ref_xw12.host_view());
  }
};

template<uint NTHREADS, typename T, typename AT, int TileM, int TileN, uint RowTile, bool enableOverlap>
__global__ void selfAttnDotProdSoftmaxDropout(uint32_t M, uint32_t N,
                                              T* XQKV, T* out, float p,
                                              curandState* randStates,
                                              MiddleCuStage cons1, MiddleCuStage prod2) {
  extern __shared__ half xqkRows[];

  __shared__ AT sum;
  if (enableOverlap)
    prod2.tile((dim3*)xqkRows);
  int linearThreadId = blockIdx.x * blockDim.x + threadIdx.x;
  curandState* localRandState = &randStates[linearThreadId];
  // __shared__ shRandStates[sizeof(curandState) * NTHREADS];
  uint ROW = blockIdx.x * RowTile;
  const uint tileRow = blockIdx.x;
  const uint tileM = ROW/TileM;
  if (enableOverlap) {
    // && tileM == 0) printf("TileM %d TileN %d ROW %d\n", TileM, TileN, ROW);
    // handle1.waitOnTilesWithSyncValue(tileM, 0, 0, 1);
    // if (tileM < M/TileM) {
    //   {tileM + 1, 0, 0};
    //   handle1.waitOnTile();
    // }
  }

  for (uint ti = 0; ti < RowTile && ROW < M; ti++) {
    if (threadIdx.x == 0) {
      sum = 0;
    }

    AT threadSum = (AT)0.0f;

    for (int COL = threadIdx.x; COL < N; COL += blockDim.x) {
      if (enableOverlap) {
        if (ti == 0) {
          dim3 tile = {tileM, COL/TileN, 0};
          cons1.wait(tile, (COL/TileN)%NTHREADS);
        }
      }
      T xq = XQKV[ROW * 3 * N + COL];
      if (enableOverlap  && ti == 0) {
        dim3 tile = {tileM, N/TileN + COL/TileN, 0};
        cons1.wait(tile, (COL/TileN)%NTHREADS);
      }
      T xk = XQKV[ROW * 3 * N + (COL + N)];
      // if (enableOverlap) {
      //   handle.waitOnTiles();
      // }
      T xqk = xq * xk;
      threadSum += (AT)exp((AT)xqk);
      xqkRows[COL] = xqk;
    }
    __syncthreads();
    atomicAdd(&sum, (AT)threadSum);
    __syncthreads();
    // if (threadIdx.x == 0) printf("sum %f\n", sum);
    // for (int COL = threadIdx.x; COL < N; COL += blockDim.x) {
    //   xqkRows[COL] = xqkRows[COL]/(__half)sum;
    // }
    // __syncthreads();
    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //   printf("185: %p\n", out);
    // }
    for (int COL = threadIdx.x; COL < N; COL += blockDim.x) {
      float r = curand_uniform(localRandState);
      // if (enableOverlap && ti == 0) {
      //   if (rowSyncOrTileSync) {

      //   } else {
      if (enableOverlap && ti == 0) {
        dim3 tile = {tileM, N/TileN*2 + COL/TileN, 0};
        cons1.wait(tile, (COL/TileN)%NTHREADS);
      }
      __half v = (r <= p) ? (__half)(((float)(exp((AT)xqkRows[COL]) * (float)XQKV[ROW* 3 * N + (COL + 2 * N)]))/sum) : (__half)0.0f;
      // if (COL == 0 && blockIdx.x < 8) {
      //   printf("199: %f %f %f %f %f\n", r, p, (float)v, (float)xqkRows[COL], (float)XQKV[ROW*N + COL]);
      // }
      out[ROW * N + COL] = v;
      if (enableOverlap && ti == SoftmaxRowTile - 1) {
        // printf("206: COL %d TileN %d threadIdx.x %d\n", COL, TileN, threadIdx.x);
        dim3 tile = {tileM, COL/TileN, 0};
        prod2.post(tile, ((COL/TileN)*TileN)%NTHREADS);
      }
    }
    __syncthreads();

    ROW++;
  }

  // if (enableOverlap) {
  //   if (rowSyncOrTileSync) {
  //     // tileM = ROW/TileM;
  //     handle2.setRowStatus(tileM, 0, 0, RowTile);
  //   } else {
      
  //   }
  // }
}

void attnRefMatmul(cutlass::gemm::GemmCoord size, ElementOutput* a, ElementOutput* b, ElementOutput* c) {
  ref_matmul<ElementOutput, ElementAccumulator>(size.m(), size.n(), 
                                                size.k(), a, b, c);
}

cudaError_t host_attention(AttentionParams& attnParams) {
  attnRefMatmul(attnParams.gemm_size1, attnParams.x.device_data(), 
                attnParams.qkv.device_data(), attnParams.ref_xqkv.host_data());
  
  size_t xq_size = attnParams.ref_xdot.size();
  assert(attnParams.ref_xdot.size() == attnParams.gemm_size1.m() * attnParams.gemm_size1.n()/3);
  size_t N = attnParams.gemm_size1.n()/3;
  ElementOutput* host_xqkv = attnParams.ref_xqkv.host_data();
  ElementOutput* host_xdot = attnParams.ref_xdot.host_data();

  for (size_t row = 0; row < attnParams.gemm_size1.m(); row++) {
    for (size_t col = 0; col < attnParams.gemm_size1.n()/3; col++) {
      ElementOutput xqk = host_xqkv[row * 3 * N + col] * host_xqkv[row * 3 * N + (col + N)];
      host_xdot[row * N + col] = xqk;
    }
  }

  for (size_t ROW = 0; ROW < attnParams.gemm_size1.m(); ROW++) {
    float sum = 0.0f;
    for (size_t COL = 0; COL < attnParams.gemm_size1.n()/3; COL++) {
      sum += exp((float)host_xdot[ROW*N + COL]);
    }
    
    for (size_t COL = 0; COL < attnParams.gemm_size1.n()/3; COL++) {
      //Assume dropout probability is 1.0
      host_xdot[ROW*N + COL] = (exp(host_xdot[ROW*N + COL]) * host_xqkv[ROW*3*N + COL+2*N])/sum;
    }
  }
  
  attnParams.ref_xdot.sync_device();

  attnRefMatmul(attnParams.gemm_size2, attnParams.ref_xdot.device_data(), 
                attnParams.w2.device_data(), attnParams.ref_xw12.host_data());
  
  return cudaSuccess;
}

cudaError_t check_results(AttentionParams& attnParams) {
  ElementOutput* hostXQKV = new ElementOutput[attnParams.xqkv.size()];
  CUDA_CHECK(cudaMemcpy(hostXQKV, attnParams.xqkv.device_data(), 
                        attnParams.xqkv.size() * sizeof(ElementOutput), 
                        cudaMemcpyDeviceToHost));
  printf("Checking First GeMM output\n");
  bool eq = equals(attnParams.ref_xqkv.size(), 
                   attnParams.ref_xqkv.host_data(), 
                   hostXQKV, 1e-1f);
  if (eq == false) {
    printf("First GeMM not correct\n");
    return cudaErrorUnknown;
  }

  ElementOutput* hostxdot = new ElementOutput[attnParams.xdot.size()];
  CUDA_CHECK(cudaMemcpy(hostxdot, attnParams.xdot.device_data(), 
                        attnParams.xdot.size() * sizeof(ElementOutput),
                        cudaMemcpyDeviceToHost));
  printf("Checking Dot Dropout kernel\n");
  eq = equals(attnParams.ref_xdot.size(), 
              attnParams.ref_xdot.host_data(), hostxdot, 1e-1f);
  if (eq == false) {
    printf("Dot not correct\n");
    return cudaErrorUnknown;
  }

  ElementOutput* hostxw12 = new ElementOutput[attnParams.xw12.size()];
  CUDA_CHECK(cudaMemcpy(hostxw12, attnParams.xw12.device_data(), 
                        attnParams.xw12.size() * sizeof(ElementOutput),
                        cudaMemcpyDeviceToHost));
  printf("Checking second GeMM\n");
  eq = equals(attnParams.ref_xw12.size(), attnParams.ref_xw12.host_data(), 
              hostxw12, 1e-1);
  if (eq == false) {
    printf("Second GeMM not correct\n");
    return cudaErrorUnknown;
  }
  printf("Self-Attention Passed\n");

  return cudaSuccess;
}

__global__ void print_kernel(ElementOutput* data) {
  if (threadIdx.x < 10) {
    printf("%p %f\n", data, (float)data[threadIdx.x]);
  }
}

//Run our baseline of Self-Attention
template<typename GemmTy1, typename GemmTy2>
cudaError_t runAttentionBaseline(int split_k1, int split_k2,
                                 AttentionParams& attnParams,
                                 cudaStream_t streams[],
                                 double& execTime,
                                 double& matmul1Time,
                                 double& softmaxTime,
                                 double& matmul2Time,
                                 int iters = 100) {  
  // ElementOutput* device_xqkv = tensor_xqkv.device_data();
  cutlass::Status status;

  //Setup First GeMM
  typename GemmTy1::Arguments args1{attnParams.gemm_size1,
                                    attnParams.x.device_ref(),
                                    attnParams.qkv.device_ref(),
                                    attnParams.xqkv.device_ref(),
                                    attnParams.xqkv.device_ref(),
                                    {attnParams.alpha, attnParams.beta},
                                    split_k1};
  size_t workspace_size = GemmTy1::get_workspace_size(args1);
  cutlass::device_memory::allocation<uint8_t> workspace1(workspace_size);
  GemmTy1 gemm_op1;
  status = gemm_op1.can_implement(args1);
  CUTLASS_CHECK(status);
  status = gemm_op1.initialize(args1, workspace1.get());
  CUTLASS_CHECK(status);

  //Setup Second GeMM
  typename GemmTy2::Arguments args2{attnParams.gemm_size2,
                                    attnParams.xdot.device_ref(),
                                    attnParams.w2.device_ref(),
                                    attnParams.xw12.device_ref(),
                                    attnParams.xw12.device_ref(),
                                    {attnParams.alpha, attnParams.beta},
                                    split_k2};
  workspace_size = GemmTy2::get_workspace_size(args2);
  cutlass::device_memory::allocation<uint8_t> workspace2(workspace_size);
  GemmTy2 gemm_op2;
  status = gemm_op2.can_implement(args2);
  CUTLASS_CHECK(status);
  status = gemm_op2.initialize(args2, workspace2.get());
  CUTLASS_CHECK(status);

  const int SoftmaxThreads = ShapeMMAThreadBlock::kN;
  execTime = 0;
  
  //Launch kernels
  for (int r = 0; r < iters; r++) {
    double start = timeInMicroSeconds();
    status = gemm_op1(streams[0]);
    CUTLASS_CHECK(status);
    CUDA_CHECK(cudaStreamSynchronize(streams[0]));
    double middle1 = timeInMicroSeconds();
    double iterMatMul1 = middle1-start;
    matmul1Time += iterMatMul1;
    
    selfAttnDotProdSoftmaxDropout<SoftmaxThreads, half, float, 
                                  ShapeMMAThreadBlock::kM, ShapeMMAThreadBlock::kN,
                                  SoftmaxRowTile, false>
                                  <<<DIVUP(attnParams.gemm_size1.m(), SoftmaxRowTile), 
                                    SoftmaxThreads, 
                                    attnParams.gemm_size1.n()/3 * sizeof(half), 
                                    streams[0]>>>
                                    (attnParams.gemm_size1.m(), 
                                    attnParams.gemm_size1.n()/3, 
                                    (half*)attnParams.xqkv.device_data(),
                                    (half*)attnParams.xdot.device_data(), 
                                    1.0f, attnParams.randStates,
                                    MiddleCuStage(), MiddleCuStage());
    CUDA_CHECK(cudaStreamSynchronize(streams[0]));
    double middle2 = timeInMicroSeconds();
    double iterSoftmax = middle2-middle1;
    softmaxTime += iterSoftmax;
    status = gemm_op2(streams[0]);
    CUTLASS_CHECK(status);
    CUDA_CHECK(cudaDeviceSynchronize());
    double middle3 = timeInMicroSeconds();
    double iterMatmul2 = middle3-middle2;
    matmul2Time += iterMatmul2;
    double end = timeInMicroSeconds();
    if (iters > 10)
      printf("{\"Total\": %lf, \"matmul1Time\": %lf, \"softmaxTime\": %lf, \"matmul2Time\": %lf}\n",end-start,iterMatMul1,iterSoftmax, iterMatmul2);
    execTime += end-start;
  }

  return cudaSuccess;
}

template<typename GemmTy1, typename GemmTy2, typename GemmSplitKTy1, typename GemmSplitKTy2>
cudaError_t runAttentionBaseline(int split_k1, int split_k2,
                                 AttentionParams& attnParams, 
                                 cudaStream_t streams[],
                                 double& execTime,
                                 double& matmul1Time,
                                 double& softmaxTime,
                                 double& matmul2Time,
                                 int iters = 100) {
  cudaError_t result;
  if (split_k1 == 1) {
    result = runAttentionBaseline<GemmTy1, GemmTy2>(split_k1, split_k2, attnParams, streams, execTime, matmul1Time, softmaxTime, matmul2Time, iters);
  } else {
    result = runAttentionBaseline<GemmSplitKTy1, GemmSplitKTy2>(split_k1, split_k2, attnParams, streams, execTime, matmul1Time, softmaxTime, matmul2Time, iters);
  }

  return result;
}

template<typename GemmTy1, typename GemmTy2>
cudaError_t runAttentionCuSync(int split_k1, int split_k2, cutlass::gemm::GemmCoord problem_size1,
                     cutlass::gemm::GemmCoord problem_size2,
                     ElementComputeEpilogue alpha,
                     ElementComputeEpilogue beta,
                     cutlass::HostTensor<ElementInputA, LayoutInputA>& tensor_x,
                     cutlass::HostTensor<ElementInputB, LayoutInputB>& tensor_qkv,
                     cutlass::HostTensor<ElementOutput, LayoutOutput>& tensor_xqkv,
                     cutlass::HostTensor<ElementOutput, LayoutOutput>& tensor_dropout,
                     cutlass::HostTensor<ElementOutput, LayoutOutput>& tensor_w2,
                     cutlass::HostTensor<ElementOutput, LayoutOutput>& tensor_out,
                     CuSyncImpl1& handle1,
                     CuSyncImpl2& handle2,
                     cudaStream_t streams[],
                     cudaEvent_t event,
                     curandState* randStates,
                     double& execTime,
                     double& matmul1Time,
                     double& softmaxTime,
                     double& matmul2Time,
                     int iters = 100) {  
  ElementOutput* device_xqkv = tensor_xqkv.device_data();
  size_t xq_size = tensor_dropout.size();
  // ElementOutput* device_xq = device_xqkv;
  // ElementOutput* device_xk = device_xqkv + xq_size;
  // ElementOutput* device_xv = device_xqkv + xq_size * 2;
  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename GemmTy1::Arguments args1{handle1.prod(),
                                     problem_size1,  // <- problem size of matrix multiplication
                                     tensor_x.device_ref(),  // <- reference to matrix A on device
                                     tensor_qkv.device_ref(),  // <- reference to matrix B on device
                                     tensor_xqkv.device_ref(),  // <- reference to matrix C on device
                                     tensor_xqkv.device_ref(),  // <- reference to matrix C on device
                                     {alpha, beta},          // <- tuple of alpha and beta
                                     split_k1};        // <- k-dimension split factor
  
  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = GemmTy1::get_workspace_size(args1);
  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace1(workspace_size);

  typename GemmTy2::Arguments args2{handle2.cons(),
                                     problem_size2,  // <- problem size of matrix multiplication
                                     tensor_dropout.device_ref(),  // <- reference to matrix A on device
                                     tensor_w2.device_ref(),  // <- reference to matrix B on device
                                     tensor_out.device_ref(),  // <- reference to matrix C on device
                                     tensor_out.device_ref(),  // <- reference to matrix C on device
                                     {alpha, beta},          // <- tuple of alpha and beta
                                     split_k2};        // <- k-dimension split factor
  
  // Using the arguments, query for extra workspace required for matrix multiplication computation
  workspace_size = GemmTy2::get_workspace_size(args2);
  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace2(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  GemmTy1 gemm_op1;
  GemmTy2 gemm_op2;
  cutlass::Status status;
  {
    // Check the problem size is supported or not 
    status = gemm_op1.can_implement(args1);
    CUTLASS_CHECK(status);

    // Initialize CUTLASS kernel with arguments and workspace pointer
    status = gemm_op1.initialize(args1, workspace1.get());
    CUTLASS_CHECK(status);
  }
  {
    // Check the problem size is supported or not 
    cutlass::Status status = gemm_op2.can_implement(args2);
    CUTLASS_CHECK(status);

    // Initialize CUTLASS kernel with arguments and workspace pointer
    status = gemm_op2.initialize(args2, workspace2.get());
    CUTLASS_CHECK(status);
  }
  const int SoftmaxThreads = ShapeMMAThreadBlock::kN;
  execTime = 0;
  
  // Launch initialized CUTLASS kernel
  for (int r = 0; r < iters; r++) {
    handle1.prod().iter += 1;
    handle1.cons().iter += 1;
    handle2.prod().iter += 1;
    handle2.cons().iter += 1;

    typename GemmTy1::Arguments args1{handle1.prod(),
                                    problem_size1,  // <- problem size of matrix multiplication
                                    tensor_x.device_ref(),  // <- reference to matrix A on device
                                    tensor_qkv.device_ref(),  // <- reference to matrix B on device
                                    tensor_xqkv.device_ref(),  // <- reference to matrix C on device
                                    tensor_xqkv.device_ref(),  // <- reference to matrix C on device
                                    {alpha, beta},          // <- tuple of alpha and beta
                                    split_k1};        // <- k-dimension split factor
    double start = timeInMicroSeconds();
    // dim3 grid = {problem_size1.m()/128, 1, 1};
    // int lastBlockIdxX = (grid.x/80)*80;
    status = gemm_op1(args1, true, NULL, workspace1.get(), streams[0]);
    CUTLASS_CHECK(status);


    if (status != cutlass::Status::kSuccess) {
      return cudaErrorUnknown;
    }
    // printf("427: *kernelExecuted %d handle.iter %d\n", *kernelExecuted, handle.iter);
    // {
    //   double start = timeInMicroSeconds();
    //   while(*kernelExecuted < handle.iter);
    //   double end = timeInMicroSeconds();
    //   printf("456: %lf microseconds\n", end-start);
    // }
    // printf("429: *kernelExecuted %d handle.iter %d\n", *kernelExecuted, handle.iter);
    // cudaEventRecord(event, producer_stream);
    // cudaStreamWaitEvent(consumer_stream, event, 0);
    // CUDA_CHECK(cudaStreamSynchronize(producer_stream));

    // status = gemm_op1(args1, true, lastBlockIdxX, grid.x, NULL, producer_stream);
    CUDA_CHECK(cudaDeviceSynchronize());

    handle1.invokeWaitKernel(streams[1]);
    CUDA_CHECK(cudaDeviceSynchronize());

    // waitKernel<<<1,1,0,streams[1]>>>((uint*)&kernelExecuted[0], handle1.iter);
    // printf("498:\n");
    selfAttnDotProdSoftmaxDropout<SoftmaxThreads, half, float, ShapeMMAThreadBlock::kM, ShapeMMAThreadBlock::kN, SoftmaxRowTile, true><<<DIVUP(problem_size1.m(), SoftmaxRowTile), SoftmaxThreads, problem_size1.n()/3 * sizeof(half), streams[1]>>>(problem_size1.m(), problem_size1.n()/3, 
                                                                (half*)device_xqkv,
                                                                (half*)tensor_dropout.device_data(),
                                                                1.0f,
                                                                randStates, 
                                                                handle1.cons(), handle2.prod());
    // print_kernel<<<1, 32, 0, streams[1]>>>(tensor_dropout.device_data());
    typename GemmTy2::Arguments args2{handle2.cons(),
      problem_size2,  // <- problem size of matrix multiplication
      tensor_dropout.device_ref(),  // <- reference to matrix A on device
      tensor_w2.device_ref(),  // <- reference toatrix m on device
      tensor_out.device_ref(),  // <- reference to matrix C on device
      tensor_out.device_ref(),  // <- reference to matrix C on device
      {alpha, beta},          // <- tuple of alpha and beta
      split_k2};        // <- k-dimension split factor
    // waitKernel<<<1,1,0,streams[2]>>>((uint*)&kernelExecuted[1], handle2.iter);
    CUDA_CHECK(cudaDeviceSynchronize());
    handle2.invokeWaitKernel(streams[2]);
    CUDA_CHECK(cudaDeviceSynchronize());

    status = gemm_op2(args2, true, NULL, workspace2.get(), streams[2]);
    CUTLASS_CHECK(status);

    if (status != cutlass::Status::kSuccess) {
      return cudaErrorUnknown;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    // CUDA_CHECK(cudaStreamSynchronize(consumer_stream));
    // CUDA_CHECK(cudaStreamSynchronize(producer_stream));
    double end = timeInMicroSeconds();
    if (iters > 10)
      printf("{\"Total\": %lf, \"matmul1Time\": -1, \"softmaxTime\": -1, \"matmul2Time\": -1}\n",end-start);
    execTime += end-start;
    matmul1Time = -1;
    softmaxTime = -1;
    matmul2Time = -1;
  }

  return cudaSuccess;
}

template<typename GemmTy1, typename GemmTy2, typename GemmSplitKTy1, typename GemmSplitKTy2>
cudaError_t runAttentionCuSync(int split_k1, int split_k2, cutlass::gemm::GemmCoord problem_size1,
                        cutlass::gemm::GemmCoord problem_size2,
                        ElementComputeEpilogue alpha,
                        ElementComputeEpilogue beta,
                        cutlass::HostTensor<ElementInputA, LayoutInputA>& tensor_x,
                        cutlass::HostTensor<ElementInputB, LayoutInputB>& tensor_qkv,
                        cutlass::HostTensor<ElementOutput, LayoutOutput>& tensor_xqkv,
                        cutlass::HostTensor<ElementOutput, LayoutOutput>& tensor_dropout,
                        cutlass::HostTensor<ElementOutput, LayoutOutput>& tensor_w2,
                        cutlass::HostTensor<ElementOutput, LayoutOutput>& tensor_out,
                        CuSyncImpl1& handle1,
                        CuSyncImpl2& handle2,
                        cudaStream_t streams[],
                        cudaEvent_t event,
                        curandState* randStates,
                        double& execTime,
                        double& matmul1Time,
                        double& softmaxTime,
                        double& matmul2Time,
                        int iters = 100) {
  #define ENABLE_NORMAL_GEMM
  cudaError_t result;
  if (split_k1 == 1) {
    result = runAttentionCuSync<GemmTy1, GemmTy2>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, 
      tensor_x, tensor_qkv, tensor_xqkv, tensor_dropout, tensor_w2, tensor_out, handle1, handle2, streams, event, randStates, execTime, matmul1Time, softmaxTime, matmul2Time, iters);
  } else {
    result = runAttentionCuSync<GemmSplitKTy1, GemmSplitKTy2>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, 
      tensor_x, tensor_qkv, tensor_xqkv, tensor_dropout, tensor_w2, tensor_out, handle1, handle2, streams, event, randStates, execTime, matmul1Time, softmaxTime, matmul2Time, iters);
  }

  return result;
}

int run(int argc, char* arg[]) {
  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (props.major != 7) {
    std::cerr << "Volta Tensor Ops must be run on a machine with compute capability of 70, 72, or 75."
              << std::endl;

    // Return 0 so tests are considered passing if run on unsupported architectures or CUDA Toolkits.
    return 0;
  }
  bool doChecking = false;
  // GEMM problem dimensions.
  int problem[4] = {128, 128, 128, 128 };

  for (int i = 1; i < argc && i < 5; ++i) {
    std::stringstream ss(arg[i]);
    ss >> problem[i - 1];
  }

  // Scalars used for linear scaling the result of the matrix product.
  float scalars[2] = { 1, 0 };

  // for (int i = 5; i < argc && i < 7; ++i) {
  //   std::stringstream ss(arg[i]);
  //   ss >> scalars[i - 4];
  // }

  if (strcmp(arg[5], "check=false") == 0) {
    doChecking = false;
  } else if (strcmp(arg[5], "check=true") == 0) {
    doChecking = true;
  } else {
    printf("invalid arg[5] %s\n", arg[7]);
    abort();
  }

  int split_k1 = 1;
  int split_k2 = 1;
  if (strstr(arg[6], "split_k1_slices=") != NULL) {
    split_k1 = atoi(arg[6] + strlen("split_k1_slices="));
  } else {
    printf("invalid arg[6] %s\n", arg[6]);
    abort();
  }

  if (strstr(arg[7], "split_k2_slices=") != NULL) {
    split_k2 = atoi(arg[7] + strlen("split_k2_slices="));
  } else {
    printf("invalid arg[7] %s\n", arg[7]);
    abort();
  }

  bool rowSyncOrTileSync;
  if (strstr(arg[8], "rowSyncOrTileSync=") != NULL) {
    int val = atoi(arg[8] + strlen("rowSyncOrTileSync="));
    if (val == 0) rowSyncOrTileSync = false; else rowSyncOrTileSync = true;
  } else {
    printf("invalid arg[8] %s\n", arg[8]);
    abort();
  }
  printf("rowSyncOrTileSync %d\n", rowSyncOrTileSync);
  //
  // Run the CUTLASS GEMM test.
  //

  int highestPriority;
  int lowestPriority;
  
  CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lowestPriority, &highestPriority));
  if (highestPriority >= lowestPriority) {
    printf("Wrong priorites: Lowest %d highest %d\n", lowestPriority, highestPriority);
  }
  cudaStream_t streams[(lowestPriority - highestPriority + 1)];
  for (int i = highestPriority; i <= lowestPriority; i++) {
    CUDA_CHECK(cudaStreamCreateWithPriority(&streams[i - highestPriority], 0, i));
  }
  printf("problem[0] %d problem[1] %d problem[2] %d problem[3] %d\n", problem[0], problem[1], problem[2], problem[3]);
  printf("doChecking=%d split_k1_slices=%d split_k2_slices=%d\n", doChecking, split_k1, split_k2);

  // Create and initialize attention tensors
  AttentionParams attnParams(problem, doChecking);
  attnParams.initIns();
  attnParams.initOuts();
  attnParams.initRefs();
  
  cudaError_t result;
  int epochs = 30;
  int warmup = 20;

  if (doChecking) {
    result = host_attention(attnParams);
    CUDA_CHECK(result);
  }
  
  double baselineTime = 0;
  double matmul1Time = 0;
  double softmaxTime = 0;
  double matmul2Time = 0;
  #define ENABLE_NORMAL_GEMM

  if (true) {
    result = runAttentionBaseline<Gemm1, Gemm2, GemmSplitK1, GemmSplitK2>(split_k1, split_k2, attnParams, streams, baselineTime, matmul1Time, softmaxTime, matmul2Time, 1);

    CUDA_CHECK(cudaDeviceSynchronize());
    if (doChecking) {
      result = check_results(attnParams);
      CUDA_CHECK(result);
    }

    result = runAttentionBaseline<Gemm1, Gemm2, GemmSplitK1, GemmSplitK2>(split_k1, split_k2, attnParams, streams, baselineTime, matmul1Time, softmaxTime, matmul2Time, warmup);

    CUDA_CHECK(cudaDeviceSynchronize());
    matmul1Time = 0;
    softmaxTime = 0;
    matmul2Time = 0;
    printf("START-BASELINE:\n");
    result = runAttentionBaseline<Gemm1, Gemm2, GemmSplitK1, GemmSplitK2>(split_k1, split_k2, attnParams, streams, baselineTime, matmul1Time, softmaxTime, matmul2Time, epochs);

    CUDA_CHECK(result);
  
    printf("END-BASELINE: {\"Total\": %lf, \"matmul1Time\": %lf, \"softmaxTime\": %lf, \"matmul2Time\": %lf} microseconds\n", baselineTime/(float)epochs, matmul1Time/(float)epochs, softmaxTime/(float)epochs, matmul2Time/(float)epochs);
  }
  #if 0

  cutlass::reference::host::TensorFill(
    tensor_xqkv.host_view());  // <- Fill matrix C on host with zeros
  cutlass::reference::host::TensorFill(
    tensor_dropout.host_view());  // <- fill matrix E on host with zeros
  cutlass::reference::host::TensorFill(
    tensor_out.host_view());  // <- fill matrix E on host with zeros
    
  tensor_xqkv.sync_device();
  tensor_dropout.sync_device();
  tensor_out.sync_device();

  // print_kernel<<<1,32>>>(tensor_xqkv.device_data());
  dim3 gridDim1 = {DIVUP(problem_size1.m(), ShapeMMAThreadBlock::kM), DIVUP(problem_size1.n(), ShapeMMAThreadBlock::kN), split_k1};
  dim3 gridDim2 = {DIVUP(problem_size1.m(), SoftmaxRowTile), 1, 1};
  dim3 gridDim3 = {DIVUP(problem_size2.m(), ShapeMMAThreadBlock::kM), DIVUP(problem_size2.n(), ShapeMMAThreadBlock::kN), split_k2};
  dim3 tileSize = {ShapeMMAThreadBlock::kM, ShapeMMAThreadBlock::kN, 1};
  
#if ROWSYNC
  using Sync1 = RowSync;
  RowSync sync1(gridDim1.y);
  using Sync2 = RowSync;
  Sync2 sync2(ShapeMMAThreadBlock::kM, SoftmaxRowTile); 
#elif TILESYNC
  using Sync1 = TileSync;
  using Sync2 = Sync1;
  TileSync sync1;
  TileSync sync2(ShapeMMAThreadBlock::kM/SoftmaxRowTile, 1);
#else
  #error "Unknown Policy"
#endif

  ProdCuStage prod1(gridDim1, tileSize, sync1);
  MiddleCuStage cons1(gridDim2, {SoftmaxRowTile, 1, 1}, sync1);
  ConsCuStage cons2(gridDim3, tileSize, sync2);

  CuSyncImpl1 handle1(prod1, cons1);
  CuSyncImpl2 handle2(cons1, cons2);
  handle2.iter = 0;
  handle1.iter = 0;
  handle1.prod().iter = handle1.cons().iter = 0;
  handle2.prod().iter = handle2.cons().iter = 0;
  
  double overlapTime = 0;
  matmul1Time = 0;
  softmaxTime = 0;
  matmul2Time = 0;
  if (true) {
    result = runAttentionCuSync<CuSyncGemm1, CuSyncGemm2, CuSyncGemmSplitK1, CuSyncGemmSplitK2>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, 
      tensor_x, tensor_qkv, tensor_xqkv, tensor_dropout, tensor_w2, tensor_out, handle1, handle2, streams, event, randStates, overlapTime, matmul1Time, softmaxTime, matmul2Time, 1);

    CUDA_CHECK(cudaDeviceSynchronize());
    if (doChecking) {
      result = check_results(problem_size1, problem_size2, alpha, beta, 
        tensor_x, tensor_qkv, tensor_xqkv, tensor_dropout, tensor_w2, tensor_out, tensor_ref_xqkv, tensor_ref_dropout, tensor_ref_out);
      if (result != cudaSuccess) {
        return 1;
      }
    }
    //warmup
    result = runAttentionCuSync<CuSyncGemm1, CuSyncGemm2, CuSyncGemmSplitK1, CuSyncGemmSplitK2>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, 
      tensor_x, tensor_qkv, tensor_xqkv, tensor_dropout, tensor_w2, tensor_out, handle1, handle2, streams, event, randStates, overlapTime, matmul1Time, softmaxTime, matmul2Time, warmup);

    CUDA_CHECK(cudaDeviceSynchronize());
    printf("START-OVERLAPPED\n");
    result = runAttentionCuSync<CuSyncGemm1, CuSyncGemm2, CuSyncGemmSplitK1, CuSyncGemmSplitK2>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, 
      tensor_x, tensor_qkv, tensor_xqkv, tensor_dropout, tensor_w2, tensor_out, handle1, handle2, streams, event, randStates, overlapTime, matmul1Time, softmaxTime, matmul2Time, epochs);
    
    printf("END-OVERLAPPED: {\"Total\": %lf, \"matmul1Time\": %lf, \"softmaxTime\": %lf, \"matmul2Time\": %lf} microseconds\n", overlapTime/(float)epochs, matmul1Time/(float)epochs, softmaxTime/(float)epochs, matmul2Time/(float)epochs);
  }
  #endif

  return 0;
}
