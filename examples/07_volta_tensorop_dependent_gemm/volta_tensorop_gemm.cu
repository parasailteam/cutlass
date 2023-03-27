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

#include <iostream>
#include <sstream>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "helper.h"
#include "cutlass/overlap_handle.h"

#include<time.h>
#include<sys/time.h>
#include <cublas_v2.h>

#define DIVUP(x, y) (((x) + (y) - 1)/(y))

static double convertTimeValToDouble(struct timeval _time) {
  return ((double)_time.tv_sec)*1e6 + ((double)_time.tv_usec);
}

static struct timeval getTimeOfDay () {
  struct timeval _time;

  if (gettimeofday (&_time, NULL) == -1) {
    fprintf (stderr, "gettimeofday returned -1\n");
    perror ("");
    abort ();
  }

  return _time;
}

static double timeInMicroSeconds() {
  return convertTimeValToDouble(getTimeOfDay());
}

static double getCurrentTime() {
  return timeInMicroSeconds();
}

#define CUBLASCHECK(cmd) do {                       \
  cublasStatus_t e = cmd;                           \
  if (e != CUBLAS_STATUS_SUCCESS) {                 \
    printf("Failed: CUBLAS error %s: %d '%d'\n",    \
           __FILE__, __LINE__, cmd);                \
    assert(false);                                  \
  }                                                 \
} while(0)                                        

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = float;                   // <- data type of accumulator
using ElementInputA = cutlass::half_t;              // <- data type of elements in input matrix A
using ElementInputB = cutlass::half_t;              // <- data type of elements in input matrix B
using ElementOutput = cutlass::half_t;
using ElementComputeEpilogue = cutlass::half_t;  // <- data type of epilogue operations

                        // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices. Column Major for
// Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm70;

/*
For inference:
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 64, 32>;
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 32>;

For training:
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<256, 128, 32>;
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<128, 64, 32>;
*/


// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 64, 32>;  // <- threadblock tile M = 128, N = 128, K = 32
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 32>;  // <- warp tile M = 64, N = 64, K = 32 
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;  // <- MMA Op tile M = 8, N = 8, K = 4

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmHorizontalThreadblockSwizzle; //;  // <- ??

// This code section describes ?
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- this is the number of elements per
                                                       // vectorized memory access. For half
                                                       // precision, it's 8 elements. This becomes
                                                       // the vector width of math instructions in
                                                       // epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

// Number of pipelines you want to use
constexpr int NumStages = 1;

using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp,
                                         EpilogueOp,
                                         SwizzleThreadBlock, 2, 8, 8>;

using GemmSplitK = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp,
                                         EpilogueOp,
                                         SwizzleThreadBlock,
                                         2, 8, 8, true>;

using OverlapGemm1 = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp,
                                         EpilogueOp,
                                         cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

using OverlapGemm2 = OverlapGemm1;

using OverlapGemmSplitK = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp,
                                         EpilogueOp,
                                         cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
                                         2, 8, 8, true>;

template<typename T, typename AT>
__global__ void matrixMultiplicationKernel(uint32_t M, uint32_t N, uint32_t K,
                                           T* A, T* B, T* C) {
    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    AT tmpSum = 0;

    if (ROW < M && COL < N) {
        // each thread computes one element of the block sub-matrix
        for (uint32_t i = 0; i < K; i++) {
            tmpSum += ((AT)A[ROW * K + i]) * ((AT)B[i * N + COL]);
        }

        C[ROW * N + COL] = (T)tmpSum;
    }
}

template<typename T, typename AT>
void gpumatmul(uint32_t M, uint32_t N, uint32_t K, T* mat1, T* mat2, T* host_res) {
  ElementOutput* dev_refC = NULL;
  CUDA_CHECK(cudaMalloc(&dev_refC, sizeof(ElementOutput)*M*N));
  dim3 block = {32, 32, 1};
  dim3 grid = {N/block.y + 1, M/block.x + 1, 1};
  matrixMultiplicationKernel<T,AT><<<grid, block>>>(M, N, K, mat1, mat2, dev_refC);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(host_res, dev_refC, sizeof(ElementOutput)*M*N, cudaMemcpyDeviceToHost));
}

template<typename T, typename AT>
void matmul(uint32_t M, uint32_t N, uint32_t K, T* mat1, T* mat2, T* res)
{
  uint32_t i, j, k;
    for (i = 0; i < M; i++) {
      #pragma omp parallel for
      for (j = 0; j < N; j++) {
          AT accum = 0;
          for (k = 0; k < K; k++)
              accum += ((float)mat1[i*K + k]) * ((float)mat2[k*N + j]);
          res[i*N + j] = T(accum);
      }
    }
}

template<typename T>
bool equals(size_t size, T* mat1, T* mat2) {
  for (size_t i = 0; i < size; i++) {
    float e1 = (float)mat1[i];
    float e2 = (float)mat2[i];
    
    float v = 1e-1;
    bool ret = true;
    if (abs(e1) < v && abs(e2) < v) {
      printf("%f , %f at %lu\n", e1, e2, i);
      ret = true;
    } else if (abs(e1) < v) {
      ret = false;
    } else if (abs(e2) < v) {
      ret = false;
    } else {
      float err = abs((e1 - e2)/e2);
      if (err <= v) {
        ret = true;
      } else {
        ret = false;
      }
    }

    if (ret == false) {
      printf("%f != %f at %lu\n", e1, e2, i);
      return false;
    }
  }

  return true;
}

template<typename T>
__global__ void printKernel(size_t sz, T* data) {
  if (threadIdx.x == 0) {
    for (size_t i = 65536; i < sz; i++) {
      printf("%f at %lu \n", (float)data[i], i);
    }
  }
}

cudaError_t host_matmul(cutlass::gemm::GemmCoord problem_size1,
  cutlass::gemm::GemmCoord problem_size2,
  ElementComputeEpilogue alpha,
  ElementComputeEpilogue beta,
  cutlass::HostTensor<ElementInputA, LayoutInputA>& tensor_a,
  cutlass::HostTensor<ElementInputB, LayoutInputB>& tensor_b,
  cutlass::HostTensor<ElementOutput, LayoutOutput>& tensor_ref_c,
  cutlass::HostTensor<ElementOutput, LayoutOutput>& tensor_d,
  cutlass::HostTensor<ElementOutput, LayoutOutput>& tensor_ref_e,
  cutlass::HostTensor<ElementOutput, LayoutOutput>& tensor_c,
  cutlass::HostTensor<ElementOutput, LayoutOutput>& tensor_e) {
  printf("Host C = A*B\n");
  // printKernel<<<1, 32>>>(tensor_c.size(), tensor_c.device_data());
  // CUDA_CHECK(cudaDeviceSynchronize());
  // matmul<ElementOutput, ElementAccumulator>(problem_size1.m(), problem_size1.n(), problem_size1.k(), tensor_a.host_data(), tensor_b.host_data(), tensor_ref_c.host_data());
  gpumatmul<ElementOutput, ElementAccumulator>(problem_size1.m(), problem_size1.n(), problem_size1.k(), tensor_a.device_data(), tensor_b.device_data(), tensor_ref_c.host_data());
  // CUDA_CHECK(cudaDeviceSynchronize());
  // CUDA_CHECK(cudaMemcpyDeviceToHost(ten
  CUDA_CHECK(cudaMemcpy(tensor_ref_c.device_data(), tensor_ref_c.host_data(), sizeof(ElementOutput) * tensor_ref_c.size(), cudaMemcpyHostToDevice));
  gpumatmul<ElementOutput, ElementAccumulator>(problem_size2.m(), problem_size2.n(), problem_size2.k(), tensor_ref_c.device_data(), tensor_d.device_data(), tensor_ref_e.host_data());
  // matrixMultiplicationKernel
  // CUDA_CHECK(cudaDeviceSynchronize());
  return cudaSuccess;
}

cudaError_t check_results(cutlass::gemm::GemmCoord problem_size1,
                    cutlass::gemm::GemmCoord problem_size2,
                    ElementComputeEpilogue alpha,
                    ElementComputeEpilogue beta,
                    cutlass::HostTensor<ElementInputA, LayoutInputA>& tensor_a,
                    cutlass::HostTensor<ElementInputB, LayoutInputB>& tensor_b,
                    cutlass::HostTensor<ElementOutput, LayoutOutput>& tensor_ref_c,
                    cutlass::HostTensor<ElementOutput, LayoutOutput>& tensor_d,
                    cutlass::HostTensor<ElementOutput, LayoutOutput>& tensor_ref_e,
                    cutlass::HostTensor<ElementOutput, LayoutOutput>& tensor_c,
                    cutlass::HostTensor<ElementOutput, LayoutOutput>& tensor_e) {
  // printKernel<<<1, 32>>>(tensor_c.size(), tensor_c.device_data());
  // CUDA_CHECK(cudaDeviceSynchronize());
  ElementOutput* hostC = new ElementOutput[tensor_ref_c.size()];
  CUDA_CHECK(cudaMemcpy(hostC, tensor_c.device_data(), tensor_c.size() * sizeof(ElementOutput), cudaMemcpyDeviceToHost));
  printf("checking C tensor_c.size() %lu %lu\n", tensor_c.size(), tensor_ref_c.size());
  bool eqC = equals(tensor_ref_c.size(), tensor_ref_c.host_data(), hostC);
  if (eqC == false) {
    printf("C not correct\n");
    return cudaErrorUnknown;
  }

  ElementOutput* hostE = new ElementOutput[tensor_ref_e.size()];
  CUDA_CHECK(cudaMemcpy(hostE, tensor_e.device_data(), tensor_e.size() * sizeof(ElementOutput), cudaMemcpyDeviceToHost));
  printf("checking E tensor_e.size() %lu %lu\n", tensor_e.size(), tensor_ref_e.size());
  bool eqE = equals(tensor_ref_e.size(), tensor_ref_e.host_data(), hostE);
  if (eqE == false) {
    printf("E not correct\n");
    return cudaErrorUnknown;
  }

  printf("passed\n");

  return cudaSuccess;
}

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

template<typename GemmTy1, typename GemmTy2>
cudaError_t runhgemm(int split_k1, int split_k2, cutlass::gemm::GemmCoord problem_size1,
                     cutlass::gemm::GemmCoord problem_size2,
                     ElementComputeEpilogue alpha,
                     ElementComputeEpilogue beta,
                     cutlass::HostTensor<ElementInputA, LayoutInputA>& tensor_a,
                     cutlass::HostTensor<ElementInputB, LayoutInputB>& tensor_b,
                     cutlass::HostTensor<ElementOutput, LayoutOutput>& tensor_c,
                     cutlass::HostTensor<ElementOutput, LayoutOutput>& tensor_d,
                     cutlass::HostTensor<ElementOutput, LayoutOutput>& tensor_e,
                     OverlapHandle& handle,
                     cudaStream_t producer_stream, cudaStream_t consumer_stream,
                     cudaEvent_t event,
                     volatile int* kernelExecuted,
                     bool rowSyncOrTileSync,
                     double& execTime,
                     int iters = 100) {  
  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename GemmTy1::Arguments args1{handle,
                                     problem_size1,  // <- problem size of matrix multiplication
                                     tensor_a.device_ref(),  // <- reference to matrix A on device
                                     tensor_b.device_ref(),  // <- reference to matrix B on device
                                     tensor_c.device_ref(),  // <- reference to matrix C on device
                                     tensor_c.device_ref(),  // <- reference to matrix C on device
                                     {alpha, beta},          // <- tuple of alpha and beta
                                     split_k1};        // <- k-dimension split factor
  
  typename GemmTy2::Arguments args2{handle,
                                     problem_size2,  // <- problem size of matrix multiplication
                                     tensor_c.device_ref(),  // <- reference to matrix A on device
                                     tensor_d.device_ref(),  // <- reference to matrix B on device
                                     tensor_e.device_ref(),  // <- reference to matrix C on device
                                     tensor_e.device_ref(),  // <- reference to matrix C on device
                                     {alpha, beta},          // <- tuple of alpha and beta
                                     split_k2};        // <- k-dimension split factor
  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = GemmTy1::get_workspace_size(args1);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace1(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  GemmTy1 gemm_op1;

  // Check the problem size is supported or not 
  cutlass::Status status = gemm_op1.can_implement(args1);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op1.initialize(args1, workspace1.get());
  CUTLASS_CHECK(status);

  GemmTy2 gemm_op2;
  workspace_size = GemmTy2::get_workspace_size(args2);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace2(workspace_size);

  // Check the problem size is supported or not 
  status = gemm_op2.can_implement(args2);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op2.initialize(args2, workspace2.get());
  CUTLASS_CHECK(status);
  execTime = 0;
  if (!handle.enable()) {
    printf("440\n");
    // Launch initialized CUTLASS kernel
    for (int r = 0; r < iters; r++) {
      handle.iter += 1;
      
      typename GemmTy1::Arguments args1{handle,
        problem_size1,  // <- problem size of matrix multiplication
        tensor_a.device_ref(),  // <- reference to matrix A on device
        tensor_b.device_ref(),  // <- reference to matrix B on device
        tensor_c.device_ref(),  // <- reference to matrix C on device
        tensor_c.device_ref(),  // <- reference to matrix C on device
        {alpha, beta},          // <- tuple of alpha and beta
        split_k1};        // <- k-dimension split factor

      typename GemmTy2::Arguments args2{handle,
        problem_size2,  // <- problem size of matrix multiplication
        tensor_c.device_ref(),  // <- reference to matrix A on device
        tensor_d.device_ref(),  // <- reference to matrix B on device
        tensor_e.device_ref(),  // <- reference to matrix C on device
        tensor_e.device_ref(),  // <- reference to matrix C on device
        {alpha, beta},          // <- tuple of alpha and beta
        split_k2};        // <- k-dimension split factor
      
      handle.producerOrConsumer_ = true;
      double start = timeInMicroSeconds();
      status = gemm_op1(args1, false, workspace1.get(), producer_stream);
      CUTLASS_CHECK(status);
      
      if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
      }

      handle.producerOrConsumer_ = false;
      // CUDA_CHECK(cudaStreamSynchronize(consumer_stream));

      status = gemm_op2(args2, false, workspace2.get(), consumer_stream);
      CUTLASS_CHECK(status);

      if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
      }

      CUDA_CHECK(cudaDeviceSynchronize());
      double end = timeInMicroSeconds();
      if (iters > 10 && producer_stream == consumer_stream)
        printf("%lf\n",end-start);
      execTime += end-start;
    }
  } else {
    // Launch initialized CUTLASS kernel
    for (int r = 0; r < iters; r++) {
      handle.iter += 1;
      handle.producerOrConsumer_ = true;
      typename GemmTy1::Arguments args1{handle,
        problem_size1,  // <- problem size of matrix multiplication
        tensor_a.device_ref(),  // <- reference to matrix A on device
        tensor_b.device_ref(),  // <- reference to matrix B on device
        tensor_c.device_ref(),  // <- reference to matrix C on device
        tensor_c.device_ref(),  // <- reference to matrix C on device
        {alpha, beta},          // <- tuple of alpha and beta
        split_k1};        // <- k-dimension split factor
      
      handle.producerOrConsumer_ = false;
      typename GemmTy2::Arguments args2{handle,
        problem_size2,  // <- problem size of matrix multiplication
        tensor_c.device_ref(),  // <- reference to matrix A on device
        tensor_d.device_ref(),  // <- reference to matrix B on device
        tensor_e.device_ref(),  // <- reference to matrix C on device
        tensor_e.device_ref(),  // <- reference to matrix C on device
        {alpha, beta},          // <- tuple of alpha and beta
        split_k2};        // <- k-dimension split factor
      
      double start = timeInMicroSeconds();
      // dim3 grid = {problem_size1.m()/128, 1, 1};
      // int lastBlockIdxX = (grid.x/80)*80;
      status = gemm_op1(args1, true, rowSyncOrTileSync, (int*)kernelExecuted, workspace1.get(), producer_stream);
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
      // CUDA_CHECK(cudaDeviceSynchronize());
      waitKernel<<<1,1,0,consumer_stream>>>((uint*)kernelExecuted, handle.iter);
      status = gemm_op2(args2, true, rowSyncOrTileSync, (int*)kernelExecuted, workspace2.get(), consumer_stream);
      CUTLASS_CHECK(status);

      if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
      }
      CUDA_CHECK(cudaDeviceSynchronize());
      // CUDA_CHECK(cudaStreamSynchronize(consumer_stream));
      // CUDA_CHECK(cudaStreamSynchronize(producer_stream));
      double end = timeInMicroSeconds();
      if (iters > 10)
        printf("%lf\n",end-start);
      execTime += end-start;
    }
  }

  return cudaSuccess;
}

void cublasRowMajor(cublasHandle_t handle, const half *alpha, const half *beta, const half* a, const half* b, half* c, const half* d, half* e, 
                    int M, int N, int K, int L, double& cublasTime, int epochs) {
  cublasTime = 0;
  
  for (int i = 0; i < epochs; i++) {
    //bT = NxK
    //aT = KxM
    //cT = NxM
    //dT = LxN
    //eT = LxM
    double t1 = getCurrentTime();
    CUBLASCHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
      N, M, K, 
      (const void*)alpha,
      (const void*)b, CUDA_R_16F, N,
      (const void*)a, CUDA_R_16F, K,
      (const void*)beta, 
      (void*)c, CUDA_R_16F, N,
      CUDA_R_16F, CUBLAS_GEMM_DFALT_TENSOR_OP));
    // CUDA_CHECK(cudaDeviceSynchronize());
    // CUBLASCHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
    //   L, M, N, 
    //   (const void*)alpha,
    //   (const void*)d, CUDA_R_16F, L,
    //   (const void*)c, CUDA_R_16F, N,
    //   (const void*)beta, 
    //   (void*)e, CUDA_R_16F, L,
    //   CUDA_R_16F, CUBLAS_GEMM_DFALT_TENSOR_OP));
    
    CUDA_CHECK(cudaDeviceSynchronize());
    double t2 = getCurrentTime();
    cublasTime += t2 - t1;
  }
}

template<class T>
void memset_value(T*f, T v, size_t nelems) 
{
  T* h_buff = (T*)malloc(sizeof(T)*nelems);
  assert(h_buff != nullptr);
  for (uint64_t i = 0; i < nelems; i++) {
    h_buff[i] = v;
  }

  CUDA_CHECK(cudaMemcpy(f, h_buff, sizeof(T)*nelems, cudaMemcpyHostToDevice));
  free(h_buff);
}

template<class T>
void memset_random2(T*f, T v1, T v2, size_t nelems)
{
  // T* h_buff = (T*)malloc(sizeof(T)*nelems);
  assert(f != nullptr);
  for (uint64_t i = 0; i < nelems; i++) {
    if (rand()%2 == 0)
      f[i] = v1;
    else
      f[i] = v2;
    // printf("%f\n", (float)f[i]);
  }

  // CUDA_CHECK(cudaMemcpy(f, h_buff, sizeof(T)*nelems, cudaMemcpyHostToDevice));
  // free(h_buff);
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

  cudaStream_t producer_stream;
  cudaStream_t consumer_stream;
  CUDA_CHECK(cudaStreamCreate(&producer_stream));
  CUDA_CHECK(cudaStreamCreate(&consumer_stream));
  
  printf("problem[0] %d problem[1] %d problem[2] %d problem[3] %d\n", problem[0], problem[1], problem[2], problem[3]);
  printf("doChecking=%d split_k1_slices=%d split_k2_slices=%d\n", doChecking, split_k1, split_k2);
  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size1(problem[0], problem[1], problem[2]);
  cutlass::gemm::GemmCoord problem_size2(problem[0], problem[3], problem[1]);

  // Initialize tensors using CUTLASS helper functions
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
      problem_size1.mk());  // <- Create matrix A with dimensions M x K
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
      problem_size1.kn());  // <- Create matrix B with dimensions K x N
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
      problem_size1.mn());  // <- Create matrix C with dimensions M x N
  
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(
      problem_size2.kn());  // <- Create matrix D with dimensions M x N used to store output from
                           // CUTLASS kernel
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_e(
      problem_size2.mn());  // <- Create matrix D with dimensions M x N used to store output from
                        // CUTLASS kernel
  
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_c(
    problem_size1.mn());  // <- Create matrix D with dimensions M x N used to store output from
                           // reference kernel
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_e(
    problem_size2.mn());  // <- Create matrix D with dimensions M x N used to store output from
                          // reference kernel

  // Fill input and output matrices on host using CUTLASS helper functions
  // cutlass::reference::host::TensorFillRandomUniform(
  //     tensor_a.host_view(),
  //     1,
  //     ElementInputA(2),
  //     ElementInputA(-2),
  //     0);  // <- Fill matrix A on host with uniform-distribution random data
  // cutlass::reference::host::TensorFillRandomUniform(
  //     tensor_b.host_view(),
  //     1,
  //     ElementInputB(2),
  //     ElementInputB(-2),
  //     0);  // <- Fill matrix B on host with uniform-distribution random data
  if (doChecking) {
    memset_random2(tensor_a.host_data(), ElementOutput(0.05), ElementOutput(0.2), tensor_a.size());
    memset_random2(tensor_b.host_data(), ElementOutput(0.01), ElementOutput(0.2), tensor_b.size());
    memset_random2(tensor_d.host_data(), ElementOutput(0.01), ElementOutput(0.05), tensor_d.size());
  } else {
    cutlass::reference::host::TensorFill(
      tensor_a.host_view(),
      ElementOutput(0.05));  // <- Fill matrix B on host with uniform-distribution random data
    cutlass::reference::host::TensorFill(
      tensor_b.host_view(),
      ElementOutput(0.5));  // <- Fill matrix B on host with uniform-distribution random data
    cutlass::reference::host::TensorFill(
      tensor_d.host_view(),
      ElementOutput(0.01));  // <- Fill matrix B on host with uniform-distribution random data
  }
  // cutlass::reference::host::TensorFill(
  //   tensor_a.host_view());
  // cutlass::reference::host::TensorFill(
  //   tensor_b.host_view());
  // cutlass::reference::host::TensorFill(
  //   tensor_d.host_view());
  cutlass::reference::host::TensorFill(
      tensor_c.host_view());  // <- Fill matrix C on host with zeros
  cutlass::reference::host::TensorFill(
      tensor_ref_c.host_view());  // <- Fill matrix C on host with zeros
  cutlass::reference::host::TensorFill(
      tensor_e.host_view());  // <- fill matrix E on host with zeros
  cutlass::reference::host::TensorFill(
    tensor_ref_e.host_view());  // <- fill matrix E on host with zeros

  // Copy data from host to GPU
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_d.sync_device();

  tensor_c.sync_device();
  tensor_ref_c.sync_device();

  tensor_e.sync_device();
  tensor_ref_e.sync_device();

  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);
  
  OverlapHandle baselineHandle;
  cudaError_t result;
  int epochs = 40;
  int warmup = 10;

  double cublasTime = 0;
  cublasHandle_t cublasHandle;
  CUBLASCHECK(cublasCreate(&cublasHandle));
  CUBLASCHECK(cublasSetStream(cublasHandle, producer_stream));
  CUBLASCHECK(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));
  
  if (false) {
    half* dAlpha, *dBeta;
    half halpha = __float2half(1.0);
    CUDA_CHECK(cudaMalloc(&dAlpha, sizeof(half)));
    CUDA_CHECK(cudaMemcpy(dAlpha, &halpha, sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dBeta, sizeof(half)));
    half hbeta = __float2half(0);
    CUDA_CHECK(cudaMemcpy(dBeta, &hbeta, sizeof(half), cudaMemcpyHostToDevice));
    CUBLASCHECK(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_DEVICE));

    int M = problem[0];
    int N = problem[1];
    int K = problem[2];
    int L = problem[3];
    printf("M %d N %d K %d L %d\n", M, N, K, L);
    half* a;
    CUDA_CHECK(cudaMalloc(&a, M*K * sizeof(half)));
      // cudaMemRandInt(m1, M*K);
    memset_value(a, __float2half(1.0f), M*K);
    half* b;
    CUDA_CHECK(cudaMalloc(&b, K*N * sizeof(half)));
    // cudaMemRandInt(m2, K*N);
    memset_value(b, __float2half(1.0f), K*N);
    half* c;
    CUDA_CHECK(cudaMalloc(&c,  M*N* sizeof(half)));
      
    half* d;
    CUDA_CHECK(cudaMalloc(&d,  L*N* sizeof(half)));
    memset_value(d, __float2half(1.0f), L*N);

    half* e;
    CUDA_CHECK(cudaMalloc(&e,  M*L* sizeof(half)));

    cublasRowMajor(cublasHandle, dAlpha, dBeta, a, b, c, d, e, M, N, K, L, cublasTime, 1);

    CUDA_CHECK(cudaDeviceSynchronize());

    cublasRowMajor(cublasHandle, dAlpha, dBeta, a, b, c, d, e, M, N, K, L, cublasTime, warmup);

    CUDA_CHECK(cudaDeviceSynchronize());

    cublasTime = 0;
    cublasRowMajor(cublasHandle, dAlpha, dBeta, a, b, c, d, e, M, N, K, L, cublasTime, epochs);

    printf("cublas-baseline %lf microseconds\n", cublasTime/(float)epochs);
  }

  if (doChecking) {
    result = host_matmul(problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_ref_c, tensor_d, tensor_ref_e, tensor_c, tensor_e);
    if (result != cudaSuccess) {
      return 1;
    }
  }

  cudaEvent_t start;
  cudaEvent_t end;
  cudaEvent_t event;
  CUDA_CHECK(cudaEventCreate(&event));
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));
  double baselineTime = 0;
  #define ENABLE_NORMAL_GEMM

  if (true) {
    if (split_k1 == 1 && split_k2 == 1) {
      #ifdef ENABLE_NORMAL_GEMM
      result = runhgemm<Gemm, Gemm>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, baselineHandle, producer_stream, producer_stream, event, NULL, false, baselineTime, 1);
      #endif
    } else if (split_k1 > 1 && split_k2 == 1) {
      #ifdef ENABLE_NORMAL_GEMM
      result = runhgemm<GemmSplitK, Gemm>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, baselineHandle, producer_stream, producer_stream, event, NULL, false, baselineTime, 1);
      #endif
    } else if (split_k1 == 1 && split_k2 > 1) {
      #ifdef ENABLE_NORMAL_GEMM
      result = runhgemm<Gemm, GemmSplitK>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, baselineHandle, producer_stream, producer_stream, event, NULL, false, baselineTime, 1);
      #endif
    } else {
      result = runhgemm<GemmSplitK, GemmSplitK>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, baselineHandle, producer_stream, producer_stream, event, NULL, false, baselineTime, 1);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    if (doChecking) {
      result = check_results(problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_ref_c, tensor_d, tensor_ref_e, tensor_c, tensor_e);
      if (result != cudaSuccess) {
        return 1;
      }
    }

    //warmup
    if (split_k1 == 1 && split_k2 == 1) {
      #ifdef ENABLE_NORMAL_GEMM
      result = runhgemm<Gemm, Gemm>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, baselineHandle, producer_stream, producer_stream, event, NULL, false, baselineTime, warmup);
      #endif
    } else if (split_k1 > 1 && split_k2 == 1) {
      #ifdef ENABLE_NORMAL_GEMM
      result = runhgemm<GemmSplitK, Gemm>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, baselineHandle, producer_stream, producer_stream, event, NULL, false, baselineTime, warmup);
      #endif
    } else if (split_k1 == 1 && split_k2 > 1) {
      #ifdef ENABLE_NORMAL_GEMM
      result = runhgemm<Gemm, GemmSplitK>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, baselineHandle, producer_stream, producer_stream, event, NULL, false, baselineTime, warmup);
      #endif
    } else {
      result = runhgemm<GemmSplitK, GemmSplitK>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, baselineHandle, producer_stream, producer_stream, event, NULL, false, baselineTime, warmup);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // double startTime = convertTimeValToDouble(getTimeOfDay());    
    if (split_k1 == 1 && split_k2 == 1) {
      #ifdef ENABLE_NORMAL_GEMM
      result = runhgemm<Gemm, Gemm>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, baselineHandle, producer_stream, producer_stream, event, NULL, false, baselineTime, epochs);
      #endif
    } else if (split_k1 > 1 && split_k2 == 1) {
      #ifdef ENABLE_NORMAL_GEMM
      result = runhgemm<GemmSplitK, Gemm>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, baselineHandle, producer_stream, producer_stream, event, NULL, false, baselineTime, epochs);
      #endif
    } else if (split_k1 == 1 && split_k2 > 1) {
      #ifdef ENABLE_NORMAL_GEMM
      result = runhgemm<Gemm, GemmSplitK>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, baselineHandle, producer_stream, producer_stream, event, NULL, false, baselineTime, epochs);
      #endif
    } else {
      result = runhgemm<GemmSplitK, GemmSplitK>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, baselineHandle, producer_stream, producer_stream, event, NULL, false, baselineTime, epochs);
    }

    if (result != cudaSuccess) {
      std::cerr << "CUTLASS GEMM kernel failed: "
        << cudaGetErrorString(result) << std::endl;
      return result;
    }
    // CUDA_CHECK(cudaStreamSynchronize(producer_stream));
    // double endTime = convertTimeValToDouble(getTimeOfDay());
    // baselineTime = endTime - startTime;
    printf("cutlass-baseline elapsedtime %lf microseconds\n", baselineTime/(float)epochs);
  }

  double minimumTime = (1<<20);
  if (true) {
    minimumTime = 0;
    cudaStream_t consumer_stream;
    CUDA_CHECK(cudaStreamCreate(&consumer_stream));
    if (split_k1 == 1 && split_k2 == 1) {
      #ifdef ENABLE_NORMAL_GEMM
      result = runhgemm<Gemm, Gemm>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, baselineHandle, producer_stream, consumer_stream, event, NULL, false, minimumTime, epochs);
      #endif
    } else if (split_k1 > 1 && split_k2 == 1) {
      #ifdef ENABLE_NORMAL_GEMM
      result = runhgemm<GemmSplitK, Gemm>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, baselineHandle, producer_stream, producer_stream, event, NULL, false, minimumTime, epochs);
      #endif
    } else if (split_k1 == 1 && split_k2 > 1) {
      #ifdef ENABLE_NORMAL_GEMM
      result = runhgemm<Gemm, GemmSplitK>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, baselineHandle, producer_stream, producer_stream, event, NULL, false, minimumTime, epochs);
      #endif
    } else {
      result = runhgemm<GemmSplitK, GemmSplitK>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, baselineHandle, producer_stream, consumer_stream, event, NULL, false, minimumTime, epochs);
    }

    if (result != cudaSuccess) {
      std::cerr << "CUTLASS GEMM kernel failed: "
        << cudaGetErrorString(result) << std::endl;
      return result;
    }
    // CUDA_CHECK(cudaStreamSynchronize(producer_stream));
    // double endTime = convertTimeValToDouble(getTimeOfDay());
    // baselineTime = endTime - startTime;
  }
  printf("minimum elapsedtime %lf microseconds\n", minimumTime/(float)epochs);

  cutlass::reference::host::TensorFill(
    tensor_c.host_view());  // <- Fill matrix C on host with zeros
  cutlass::reference::host::TensorFill(
    tensor_e.host_view());  // <- fill matrix E on host with zeros
  
  tensor_c.sync_device();
  tensor_e.sync_device();
  
  OverlapHandle overlapHandle(problem_size1.m(), problem_size1.n(), 1, 1);
  int highestPriority;
  int lowestPriority;
  CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lowestPriority, &highestPriority));
  CUDA_CHECK(cudaStreamCreateWithPriority(&consumer_stream, 0, lowestPriority));
  int* kernelExecuted;
  CUDA_CHECK(cudaMalloc(&kernelExecuted, sizeof(int)));
  CUDA_CHECK(cudaMemset(kernelExecuted, 0, sizeof(int)));
  
  int* numProducerTBs;
  CUDA_CHECK(cudaMalloc(&numProducerTBs, sizeof(int)));
  CUDA_CHECK(cudaMemset(numProducerTBs, 0, sizeof(int)));
  overlapHandle.numProducerTBs = numProducerTBs;
  int* numConsumerTBs;
  CUDA_CHECK(cudaMalloc(&numConsumerTBs, sizeof(int) * 80));
  CUDA_CHECK(cudaMemset(numConsumerTBs, 0, sizeof(int) * 80));
  overlapHandle.numConsumerTBs = numConsumerTBs;
  
  overlapHandle.allocTileStatusMap(ShapeMMAThreadBlock::kM, ShapeMMAThreadBlock::kN, 1);
  double overlapTime = 0;
  dim3 gridDim = {DIVUP(problem_size1.m(), ShapeMMAThreadBlock::kM), DIVUP(problem_size1.n(), ShapeMMAThreadBlock::kN), split_k1};
  
  int* dBlockIndexOrder;
  CUDA_CHECK(cudaMalloc(&dBlockIndexOrder, sizeof(int) * gridDim.x * gridDim.y * gridDim.z * 3));
  CUDA_CHECK(cudaMemset(dBlockIndexOrder, 0, sizeof(int) * gridDim.x * gridDim.y * gridDim.z * 3));
  printf("gridDim.x %d gridDim.y %d\n", gridDim.x, gridDim.y);
  int* hBlockIndexOrder = new int[gridDim.x * gridDim.y * gridDim.z * 3];
  int linearid = 0;
  int Ny = 1;

  if (split_k1 > 1 && split_k2 > 1) {
    for (int x = 0; x < gridDim.x; x++) {
    for (int z = 0; z < gridDim.z; z++) {
    for (int y = 0; y < gridDim.y; y++) {
      hBlockIndexOrder[linearid] = x;
      hBlockIndexOrder[linearid + 1] = y;
      hBlockIndexOrder[linearid + 2] = z;
      // printf("linearid %d x %d y %d\n", linearid, x, y);
      linearid += 3;
    }
    }
    }
  } else {
    // for (int x = 0; x < gridDim.x; x += 1) {
    //   // for (int xx = x; xx < min(N_X, gridDim.x - x); xx++) {
    //   //   for (int y = 0; y < N_Y; y++) {
    //   //     hBlockIndexOrder[linearid] = xx;
    //   //     hBlockIndexOrder[linearid + 1] = y;
    //   //     // printf("linearid %d x %d y %d\n", linearid, xx, 0);
    //   //     linearid += 2;
    //   //   }
    //   // }
    //   for (int y = 0; y < gridDim.y; y += Ny) {
    //     for (int z = 0; z < gridDim.z; z++) {
    //       for (int yy = 0; yy < Ny && yy + y < gridDim.y; yy++) {
    //         hBlockIndexOrder[linearid] = x;
    //         hBlockIndexOrder[linearid + 1] = y + yy;
    //         hBlockIndexOrder[linearid + 2] = z;
    //         // printf("linearid %d x %d y %d\n", linearid, xx, y);
    //         linearid += 3;
    //       }
    //     }
    //   }
    // }

    for (int x = 0; x < gridDim.x; x++) {
    for (int z = 0; z < gridDim.z; z++) {
    for (int y = 0; y < gridDim.y; y++) {
        hBlockIndexOrder[linearid] = x;
        hBlockIndexOrder[linearid + 1] = y;
        hBlockIndexOrder[linearid + 2] = z;
        // printf("linearid %d x %d y %d\n", linearid, x, y);
        linearid += 3;
    }
    }
    }
  }
    

  printf("dBlockIndexOrder %p\n", dBlockIndexOrder);
  CUDA_CHECK(cudaMemcpy(dBlockIndexOrder, hBlockIndexOrder, sizeof(int) * gridDim.x * gridDim.y * gridDim.z * 3, cudaMemcpyHostToDevice));

  dim3 grid2Dim = {problem_size2.m()/ShapeMMAThreadBlock::kM, problem_size2.n()/ShapeMMAThreadBlock::kN, split_k2};
  int* dConsumerBlockIndexOrder;
  CUDA_CHECK(cudaMalloc(&dConsumerBlockIndexOrder, sizeof(int) * grid2Dim.x * grid2Dim.y * grid2Dim.z * 3));
  CUDA_CHECK(cudaMemset(dConsumerBlockIndexOrder, 0, sizeof(int) * grid2Dim.x * grid2Dim.y * grid2Dim.z * 3));

  hBlockIndexOrder = new int[grid2Dim.x * grid2Dim.y * grid2Dim.z * 3];
  linearid = 0;

  if (split_k1 > 1 && split_k2 > 1) {
    for (int x = 0; x < grid2Dim.x; x++) {
    for (int z = 0; z < grid2Dim.z; z++) {
    for (int y = 0; y < grid2Dim.y; y++) {
      hBlockIndexOrder[linearid] = x;
      hBlockIndexOrder[linearid + 1] = y;
      hBlockIndexOrder[linearid + 2] = z;
      // printf("linearid %d x %d y %d\n", linearid, x, y);
      linearid += 3;
    }
    }
    }
  } else {
    // for (int x = 0; x < grid2Dim.x; x += 1) {
    //   // for (int xx = x; xx < min(N_X, gridDim.x - x); xx++) {
    //   //   for (int y = 0; y < N_Y; y++) {
    //   //     hBlockIndexOrder[linearid] = xx;
    //   //     hBlockIndexOrder[linearid + 1] = y;
    //   //     // printf("linearid %d x %d y %d\n", linearid, xx, 0);
    //   //     linearid += 2;
    //   //   }
    //   // }
    //   for (int y = 0; y < grid2Dim.y; y += Ny) {
    //     for (int z = 0; z < grid2Dim.z; z++) {
    //       for (int yy = 0; yy < Ny && yy + y < grid2Dim.y; yy++) {
    //         hBlockIndexOrder[linearid] = x;
    //         hBlockIndexOrder[linearid + 1] = y + yy;
    //         hBlockIndexOrder[linearid + 2] = z;
    //         // printf("linearid %d x %d y %d z %d\n", linearid, x, y + yy, z);
    //         linearid += 3;
    //       }
    //     }
    //   }
    // }

    for (int x = 0; x < grid2Dim.x; x++) {
    for (int z = 0; z < grid2Dim.z; z++) {
    for (int y = 0; y < grid2Dim.y; y++) {
      hBlockIndexOrder[linearid] = x;
      hBlockIndexOrder[linearid + 1] = y;
      hBlockIndexOrder[linearid + 2] = z;
      // printf("linearid %d x %d y %d\n", linearid, x, y);
      linearid += 3;
    }
    }
    }
  }

  // printf("803:\n");
  CUDA_CHECK(cudaMemcpy(dConsumerBlockIndexOrder, hBlockIndexOrder, sizeof(int) * grid2Dim.x * grid2Dim.y * grid2Dim.z * 3, cudaMemcpyHostToDevice));

  int* dIsRemainingBlock;
  int* hIsRemainingBlock = new int[gridDim.x*gridDim.y];
  CUDA_CHECK(cudaMalloc(&dIsRemainingBlock, sizeof(int)*gridDim.x*gridDim.y));
  int totalBlocks = 0;
  // const int startRemainingBlockId = ((gridDim.x*gridDim.y)/(3*80))*(3*80) + 1;
  printf("Number of TBs: %d\n", gridDim.x*gridDim.y*gridDim.z);
  if ((gridDim.x*gridDim.y*gridDim.z)%80 == 0) {
    printf("Invalid\n");
    return 0;
  }
  // printf("startRemainingBlockId %d to %d\n", startRemainingBlockId, gridDim.x*gridDim.y);
  // for (int x = 0; x < gridDim.x; x++) {
  //   for (int y = 0; y < gridDim.y; y++) {
  //     if (totalBlocks >= startRemainingBlockId) {
  //       hIsRemainingBlock[totalBlocks] = 1;
  //     } else {
  //       hIsRemainingBlock[totalBlocks] = 0;
  //     }

  //     totalBlocks++;
  //   }
  // }

  CUDA_CHECK(cudaMemcpy(dIsRemainingBlock, hIsRemainingBlock, sizeof(int) * gridDim.x * gridDim.y, cudaMemcpyHostToDevice));

  overlapHandle.isBlockRemaining = dIsRemainingBlock;
  overlapHandle.blockIndexOrder = dBlockIndexOrder;
  overlapHandle.consumerBlockIndexOrder = dConsumerBlockIndexOrder;
  if (true) {
    if (split_k1 == 1 && split_k2 == 1) {
      #ifdef ENABLE_NORMAL_GEMM
      result = runhgemm<OverlapGemm1, OverlapGemm2>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, overlapHandle, producer_stream, consumer_stream,  event, kernelExecuted, rowSyncOrTileSync, overlapTime, 1);
      #endif
    } else if (split_k1 > 1 && split_k2 == 1) {
      #ifdef ENABLE_NORMAL_GEMM
      result = runhgemm<OverlapGemmSplitK, OverlapGemm2>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, overlapHandle, producer_stream, consumer_stream,  event, kernelExecuted, rowSyncOrTileSync, overlapTime, 1);
      #endif
    } else if (split_k1 == 1 && split_k2 > 1) {
      #ifdef ENABLE_NORMAL_GEMM
      result = runhgemm<OverlapGemm1, OverlapGemmSplitK>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, overlapHandle, producer_stream, consumer_stream,  event, kernelExecuted, rowSyncOrTileSync, overlapTime, 1);
      #endif
    } else {
      result = runhgemm<OverlapGemmSplitK, OverlapGemmSplitK>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, overlapHandle, producer_stream, consumer_stream,  event, kernelExecuted, rowSyncOrTileSync, overlapTime, 1);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    if (doChecking) {
      result = check_results(problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_ref_c, tensor_d, tensor_ref_e, tensor_c, tensor_e);
      if (result != cudaSuccess) {
        return 1;
      }
    }
    //warmup
    if (split_k1 == 1 && split_k2 == 1) {
      #ifdef ENABLE_NORMAL_GEMM
      result = runhgemm<OverlapGemm1, OverlapGemm2>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, overlapHandle, producer_stream, consumer_stream, event, kernelExecuted, rowSyncOrTileSync, overlapTime, warmup);
      #endif
    } else if (split_k1 > 1 && split_k2 == 1) {
      #ifdef ENABLE_NORMAL_GEMM
      result = runhgemm<OverlapGemmSplitK, OverlapGemm2>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, overlapHandle, producer_stream, consumer_stream, event, kernelExecuted, rowSyncOrTileSync, overlapTime, warmup);
      #endif
    } else if (split_k1 == 1 && split_k2 > 1) {
      #ifdef ENABLE_NORMAL_GEMM
      result = runhgemm<OverlapGemm1, OverlapGemmSplitK>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, overlapHandle, producer_stream, consumer_stream, event, kernelExecuted, rowSyncOrTileSync, overlapTime, warmup);
      #endif
    } else {
      result = runhgemm<OverlapGemmSplitK, OverlapGemmSplitK>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, overlapHandle, producer_stream, consumer_stream, event, kernelExecuted, rowSyncOrTileSync, overlapTime, warmup);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    printf("728:\n");
    // double startTime = convertTimeValToDouble(getTimeOfDay());
    if (split_k1 == 1 && split_k2 == 1) {
      #ifdef ENABLE_NORMAL_GEMM
      result = runhgemm<OverlapGemm1, OverlapGemm2>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, overlapHandle, producer_stream, consumer_stream, event, kernelExecuted, rowSyncOrTileSync, overlapTime, epochs);
      #endif
    } else if (split_k1 > 1 && split_k2 == 1) {
      #ifdef ENABLE_NORMAL_GEMM
      result = runhgemm<OverlapGemmSplitK, OverlapGemm2>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, overlapHandle, producer_stream, consumer_stream, event, kernelExecuted, rowSyncOrTileSync, overlapTime, epochs);
      #endif
    } else if (split_k1 == 1 && split_k2 > 1) {
      #ifdef ENABLE_NORMAL_GEMM
      result = runhgemm<OverlapGemm1, OverlapGemmSplitK>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, overlapHandle, producer_stream, consumer_stream, event, kernelExecuted, rowSyncOrTileSync, overlapTime, epochs);
      #endif
    } else {
      result = runhgemm<OverlapGemmSplitK, OverlapGemmSplitK>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, overlapHandle, producer_stream, consumer_stream, event, kernelExecuted, rowSyncOrTileSync, overlapTime, epochs);
    }
    CUDA_CHECK(cudaStreamSynchronize(consumer_stream));
    CUDA_CHECK(cudaStreamSynchronize(producer_stream));
    // double endTime = convertTimeValToDouble(getTimeOfDay());
    // overlapTime = endTime - startTime;

    printf("overlapped elapsedtime %lf microseconds\n", overlapTime/(float)epochs);
  }

  return 0;
}

int main(int argc, char* argv[]) {

  // Volta Tensor Core operations exposed with mma.sync are first available in CUDA 10.1.
  //
  // CUTLASS must be compiled with CUDA 10.1 Toolkit to run these examples.
  if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 1))) {
    std::cerr << "Volta Tensor Core operations must be compiled with CUDA 10.1 Toolkit or later." << std::endl;

    // Returning zero when built on older Toolkits so tests pass. The actions of this SDK example are no-op.
    return 0;
  }
  else {
    return run(argc, argv);
  }
}

