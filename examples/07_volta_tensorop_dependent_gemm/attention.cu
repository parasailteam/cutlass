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

#include "common.h"


template<typename T, typename AT>
__global__ void selfAttnDotProdSoftmaxDropout(uint32_t M, uint32_t N,
                                              T* XQ, T* XK, T* XV, T* out, float p) {
  __shared__ half xqkRows[1536];
  int ROW = blockIdx.x;
  if (ROW >= M) return;
  __shared__ AT sum;
  int linearThreadId = blockIdx.x * blockDim.x + threadIdx.x;
  // curandState state;
  // curand_init(1234, linearThreadId, 0, &state);

  if (threadIdx.x == 0) {
    sum = 0;
  }
  AT threadSum = (AT)0.0f;
  for (int COL = threadIdx.x; COL < N; COL += blockDim.x) {
      T xqk = XQ[ROW * N + COL] * XK[ROW * N + COL];
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
  for (int COL = threadIdx.x; COL < N; COL += blockDim.x) {
    float r = 0 ;//curand_uniform(&state);
    __half v = (r <= p) ? (__half)(((float)(exp((AT)xqkRows[COL]) * (float)XV[ROW*N + COL]))/sum) : (__half)0.0f;
    out[ROW * N + COL] = v;
  }
}

cudaError_t host_attention(cutlass::gemm::GemmCoord problem_size1,
  cutlass::gemm::GemmCoord problem_size2,
  ElementComputeEpilogue alpha,
  ElementComputeEpilogue beta,
  cutlass::HostTensor<ElementInputA, LayoutInputA>& tensor_x,
  cutlass::HostTensor<ElementInputB, LayoutInputB>& tensor_qkv,
  cutlass::HostTensor<ElementOutput, LayoutOutput>& tensor_ref_xqkv,
  cutlass::HostTensor<ElementOutput, LayoutOutput>& tensor_ref_dropout) {
  printf("Host C = A*B tensor_ref_xqkv.size() %d\n", tensor_ref_xqkv.size());
  gpumatmul<ElementOutput, ElementAccumulator>(problem_size1.m(), problem_size1.n(), problem_size1.k(), tensor_x.device_data(), tensor_qkv.device_data(), tensor_ref_xqkv.host_data());
  // CUDA_CHECK(cudaMemcpy(tensor_ref_xqkv.device_data(), tensor_ref_xqkv.host_data(), sizeof(ElementOutput) * tensor_ref_xqkv.size(), cudaMemcpyHostToDevice));
  printf("Host Dropout(Softmax(XQ . XK))\n");
  size_t xq_size = tensor_ref_dropout.size();
  assert(tensor_ref_dropout.size() == problem_size1.m() * problem_size1.n()/3);
  ElementOutput* host_xq = tensor_ref_xqkv.host_data();
  ElementOutput* host_xk = tensor_ref_xqkv.host_data() + xq_size;
  ElementOutput* host_xv = tensor_ref_xqkv.host_data() + xq_size * 2;
  ElementOutput* host_dropout = tensor_ref_dropout.host_data();

  for (size_t i = 0; i < xq_size; i++) {
    ElementOutput xqk = host_xq[i] * host_xk[i];
    host_dropout[i] = xqk;
  }

  for (size_t ROW = 0; ROW < problem_size1.m(); ROW++) {
    float sum = 0.0f;
    for (size_t COL = 0; COL < problem_size1.n()/3; COL++) {
      // printf("(float)host_dropout[ROW*(problem_size1.n()/3) + COL] %f\n", (float)host_dropout[ROW*(problem_size1.n()/3) + COL]);
      sum += exp((float)host_dropout[ROW*(problem_size1.n()/3) + COL]);
    }
    // printf("sum %f\n", sum);
    // for (size_t COL = 0; COL < problem_size1.n()/3; COL++) {
    //   host_dropout[ROW*(problem_size1.n()/3) + COL] = (ElementOutput)exp((float)host_dropout[ROW*(problem_size1.n()/3) + COL])/sum;
    // }

    for (size_t COL = 0; COL < problem_size1.n()/3; COL++) {
      //Assume dropout probability is 1.0
      host_dropout[ROW*(problem_size1.n()/3) + COL] = (exp(host_dropout[ROW*(problem_size1.n()/3) + COL]) * host_xv[ROW*(problem_size1.n()/3) + COL])/sum;
    }
  }

  return cudaSuccess;
}

cudaError_t check_results(cutlass::gemm::GemmCoord problem_size1,
                    cutlass::gemm::GemmCoord problem_size2,
                    ElementComputeEpilogue alpha,
                    ElementComputeEpilogue beta,
                    cutlass::HostTensor<ElementInputA, LayoutInputA>& tensor_x,
                    cutlass::HostTensor<ElementInputB, LayoutInputB>& tensor_qkv,
                    cutlass::HostTensor<ElementOutput, LayoutOutput>& tensor_xqkv,
                    cutlass::HostTensor<ElementOutput, LayoutOutput>& tensor_dropout,
                    cutlass::HostTensor<ElementOutput, LayoutOutput>& tensor_ref_xqkv,
                    cutlass::HostTensor<ElementOutput, LayoutOutput>& tensor_ref_dropout) {
  ElementOutput* hostXQKV = new ElementOutput[tensor_xqkv.size()];
  CUDA_CHECK(cudaMemcpy(hostXQKV, tensor_xqkv.device_data(), tensor_xqkv.size() * sizeof(ElementOutput), cudaMemcpyDeviceToHost));
  printf("checking C tensor_xqkv.size() %lu %lu\n", tensor_xqkv.size(), tensor_ref_xqkv.size());
  bool eqC = equals(tensor_ref_xqkv.size(), tensor_ref_xqkv.host_data(), hostXQKV, 1e-1);
  if (eqC == false) {
    printf("XQKV not correct\n");
    return cudaErrorUnknown;
  }

  ElementOutput* hostDropout = new ElementOutput[tensor_dropout.size()];
  CUDA_CHECK(cudaMemcpy(hostDropout, tensor_dropout.device_data(), tensor_dropout.size() * sizeof(ElementOutput), cudaMemcpyDeviceToHost));
  printf("checking E tensor_e.size() %lu %lu\n", tensor_dropout.size(), tensor_ref_dropout.size());
  bool eqE = equals(tensor_ref_dropout.size(), tensor_ref_dropout.host_data(), hostDropout, 1e-1);
  if (eqE == false) {
    printf("E not correct\n");
    return cudaErrorUnknown;
  }

  printf("passed\n");

  return cudaSuccess;
}

template<typename GemmTy1>
cudaError_t runAttention(int split_k1, int split_k2, cutlass::gemm::GemmCoord problem_size1,
                     cutlass::gemm::GemmCoord problem_size2,
                     ElementComputeEpilogue alpha,
                     ElementComputeEpilogue beta,
                     cutlass::HostTensor<ElementInputA, LayoutInputA>& tensor_x,
                     cutlass::HostTensor<ElementInputB, LayoutInputB>& tensor_qkv,
                     cutlass::HostTensor<ElementOutput, LayoutOutput>& tensor_xqkv,
                     cutlass::HostTensor<ElementOutput, LayoutOutput>& tensor_dropout,
                     OverlapHandle& handle,
                     cudaStream_t producer_stream, cudaStream_t consumer_stream,
                     cudaEvent_t event,
                     volatile int* kernelExecuted,
                     bool rowSyncOrTileSync,
                     double& execTime,
                     int iters = 100) {  
  ElementOutput* device_xqkv = tensor_xqkv.device_data();
  size_t xq_size = tensor_dropout.size();
  printf("xq_size %ld\n", xq_size);
  ElementOutput* device_xq = device_xqkv;
  ElementOutput* device_xk = device_xqkv + xq_size;
  ElementOutput* device_xv = device_xqkv + xq_size * 2;
  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename GemmTy1::Arguments args1{handle,
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

  // Instantiate CUTLASS kernel depending on templates
  GemmTy1 gemm_op1;

  // Check the problem size is supported or not 
  cutlass::Status status = gemm_op1.can_implement(args1);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op1.initialize(args1, workspace1.get());
  CUTLASS_CHECK(status);

  execTime = 0;
  if (!handle.enable()) {
    printf("440\n");
    // Launch initialized CUTLASS kernel
    for (int r = 0; r < iters; r++) {
      handle.iter += 1;

      // typename GemmTy2::Arguments args2{handle,
      //   problem_size2,  // <- problem size of matrix multiplication
      //   tensor_c.device_ref(),  // <- reference to matrix A on device
      //   tensor_d.device_ref(),  // <- reference to matrix B on device
      //   tensor_e.device_ref(),  // <- reference to matrix C on device
      //   tensor_e.device_ref(),  // <- reference to matrix C on device
      //   {alpha, beta},          // <- tuple of alpha and beta
      //   split_k2};        // <- k-dimension split factor
      
      handle.producerOrConsumer_ = true;
      double start = timeInMicroSeconds();
      status = gemm_op1(args1, false, workspace1.get(), producer_stream);
      CUTLASS_CHECK(status);
      
      if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
      }

      handle.producerOrConsumer_ = false;
      // CUDA_CHECK(cudaStreamSynchronize(consumer_stream));

      // status = gemm_op2(args2, false, workspace2.get(), consumer_stream);
      // CUTLASS_CHECK(status);

      // if (status != cutlass::Status::kSuccess) {
      //   return cudaErrorUnknown;
      // }
      // CUDA_CHECK(cudaDeviceSynchronize());
      selfAttnDotProdSoftmaxDropout<half, float><<<problem_size1.m(), 256>>>(problem_size1.m(), problem_size1.n()/3, 
                                                                 (half*)device_xq, (half*)device_xk, (half*)device_xv, 
                                                                 (half*)tensor_dropout.device_data(), 
                                                                 1.0f);
      CUDA_CHECK(cudaDeviceSynchronize());
      double end = timeInMicroSeconds();
      if (iters > 10 && producer_stream == consumer_stream)
        printf("%lf\n",end-start);
      execTime += end-start;
    }
  } else {
    // // Launch initialized CUTLASS kernel
    // for (int r = 0; r < iters; r++) {
    //   handle.iter += 1;
    //   handle.producerOrConsumer_ = true;
    //   typename GemmTy1::Arguments args1{handle,
    //     problem_size1,  // <- problem size of matrix multiplication
    //     tensor_a.device_ref(),  // <- reference to matrix A on device
    //     tensor_b.device_ref(),  // <- reference to matrix B on device
    //     tensor_c.device_ref(),  // <- reference to matrix C on device
    //     tensor_c.device_ref(),  // <- reference to matrix C on device
    //     {alpha, beta},          // <- tuple of alpha and beta
    //     split_k1};        // <- k-dimension split factor
      
    //   handle.producerOrConsumer_ = false;
    //   typename GemmTy2::Arguments args2{handle,
    //     problem_size2,  // <- problem size of matrix multiplication
    //     tensor_c.device_ref(),  // <- reference to matrix A on device
    //     tensor_d.device_ref(),  // <- reference to matrix B on device
    //     tensor_e.device_ref(),  // <- reference to matrix C on device
    //     tensor_e.device_ref(),  // <- reference to matrix C on device
    //     {alpha, beta},          // <- tuple of alpha and beta
    //     split_k2};        // <- k-dimension split factor
      
    //   double start = timeInMicroSeconds();
    //   // dim3 grid = {problem_size1.m()/128, 1, 1};
    //   // int lastBlockIdxX = (grid.x/80)*80;
    //   status = gemm_op1(args1, true, rowSyncOrTileSync, (int*)kernelExecuted, workspace1.get(), producer_stream);
    //   CUTLASS_CHECK(status);

      
    //   if (status != cutlass::Status::kSuccess) {
    //     return cudaErrorUnknown;
    //   }
    //   // printf("427: *kernelExecuted %d handle.iter %d\n", *kernelExecuted, handle.iter);
    //   // {
    //   //   double start = timeInMicroSeconds();
    //   //   while(*kernelExecuted < handle.iter);
    //   //   double end = timeInMicroSeconds();
    //   //   printf("456: %lf microseconds\n", end-start);
    //   // }
    //   // printf("429: *kernelExecuted %d handle.iter %d\n", *kernelExecuted, handle.iter);
    //   // cudaEventRecord(event, producer_stream);
    //   // cudaStreamWaitEvent(consumer_stream, event, 0);
    //   // CUDA_CHECK(cudaStreamSynchronize(producer_stream));

    //   // status = gemm_op1(args1, true, lastBlockIdxX, grid.x, NULL, producer_stream);
    //   // CUDA_CHECK(cudaDeviceSynchronize());
    //   waitKernel<<<1,1,0,consumer_stream>>>((uint*)kernelExecuted, handle.iter);
    //   status = gemm_op2(args2, true, rowSyncOrTileSync, (int*)kernelExecuted, workspace2.get(), consumer_stream);
    //   CUTLASS_CHECK(status);

    //   if (status != cutlass::Status::kSuccess) {
    //     return cudaErrorUnknown;
    //   }
    //   CUDA_CHECK(cudaDeviceSynchronize());
    //   // CUDA_CHECK(cudaStreamSynchronize(consumer_stream));
    //   // CUDA_CHECK(cudaStreamSynchronize(producer_stream));
    //   double end = timeInMicroSeconds();
    //   if (iters > 10)
    //     printf("%lf\n",end-start);
    //   execTime += end-start;
    // }
  }

  return cudaSuccess;
}

template<typename GemmTy1, typename GemmTy2, typename GemmSplitKTy1, typename GemmSplitKTy2>
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
  #define ENABLE_NORMAL_GEMM
  cudaError_t result;
  if (split_k1 == 1 && split_k2 == 1) {
    result = runhgemm<GemmTy1, GemmTy2>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, handle, producer_stream, consumer_stream, event, kernelExecuted, rowSyncOrTileSync, execTime, iters);
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

  cudaStream_t producer_stream;
  cudaStream_t consumer_stream;
  CUDA_CHECK(cudaStreamCreate(&producer_stream));
  CUDA_CHECK(cudaStreamCreate(&consumer_stream));
  
  printf("problem[0] %d problem[1] %d problem[2] %d problem[3] %d\n", problem[0], problem[1], problem[2], problem[3]);
  printf("doChecking=%d split_k1_slices=%d split_k2_slices=%d\n", doChecking, split_k1, split_k2);
  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size1(problem[0], problem[1] * 3, problem[2]);
  cutlass::gemm::GemmCoord problem_size2(problem[0], problem[3], problem[1]);

  // Initialize tensors using CUTLASS helper functions
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_x(
      problem_size1.mk());  // <- Create matrix A with dimensions M x K
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_qkv(
      problem_size1.kn());  // <- Create matrix B with dimensions K x N
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_xqkv(
      problem_size1.mn());  // <- Create matrix C with dimensions M x N
  
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_dropout(
    {problem_size1.m(), problem_size1.n()/3});  // <- Create matrix D with dimensions M x N used to store output from
                           // CUTLASS kernel
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_e(
      problem_size2.mn());  // <- Create matrix D with dimensions M x N used to store output from
                        // CUTLASS kernel
  
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_xqkv(
    problem_size1.mn());  // <- Create matrix D with dimensions M x N used to store output from
                           // reference kernel
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_dropout(
    {problem_size1.m(), problem_size1.n()/3});  // <- Create matrix D with dimensions M x N used to store output from
                          // reference kernel
  printf("%ld\n", tensor_ref_dropout.size());
  
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
    memset_random2(tensor_x.host_data(), ElementOutput(0.02), ElementOutput(0.03), tensor_x.size());
    memset_random2(tensor_qkv.host_data(), ElementOutput(0.01), ElementOutput(0.035), tensor_qkv.size());
    // memset_random2(tensor_d.host_data(), ElementOutput(0.01), ElementOutput(0.05), tensor_d.size());
  } else {
    cutlass::reference::host::TensorFill(
      tensor_x.host_view(),
      ElementOutput(0.05));  // <- Fill matrix B on host with uniform-distribution random data
    cutlass::reference::host::TensorFill(
      tensor_qkv.host_view(),
      ElementOutput(0.5));  // <- Fill matrix B on host with uniform-distribution random data
    // cutlass::reference::host::TensorFill(
    //   tensor_d.host_view(),
    //   ElementOutput(0.01));  // <- Fill matrix B on host with uniform-distribution random data
  }
  // cutlass::reference::host::TensorFill(
  //   tensor_a.host_view());
  // cutlass::reference::host::TensorFill(
  //   tensor_b.host_view());
  // cutlass::reference::host::TensorFill(
  //   tensor_d.host_view());
  cutlass::reference::host::TensorFill(
    tensor_xqkv.host_view());  // <- Fill matrix C on host with zeros
  cutlass::reference::host::TensorFill(
    tensor_ref_xqkv.host_view());  // <- Fill matrix C on host with zeros
  cutlass::reference::host::TensorFill(
    tensor_ref_dropout.host_view());  // <- fill matrix E on host with zeros
  // cutlass::reference::host::TensorFill(
  //   tensor_ref_e.host_view());  // <- fill matrix E on host with zeros

  // Copy data from host to GPU
  tensor_x.sync_device();
  tensor_qkv.sync_device();
  // tensor_d.sync_device();

  tensor_xqkv.sync_device();
  tensor_ref_xqkv.sync_device();

  // tensor_e.sync_device();
  tensor_ref_dropout.sync_device();

  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);
  
  OverlapHandle baselineHandle;
  cudaError_t result;
  int epochs = 40;
  int warmup = 10;

  if (doChecking) {
    result = host_attention(problem_size1, problem_size2, alpha, beta, tensor_x, tensor_qkv, tensor_ref_xqkv, tensor_ref_dropout);
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
    result = runAttention<Gemm>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, tensor_x, tensor_qkv, tensor_xqkv, tensor_dropout, baselineHandle, producer_stream, producer_stream, event, NULL, false, baselineTime, 1);

    CUDA_CHECK(cudaDeviceSynchronize());
    if (doChecking) {
      result = check_results(problem_size1, problem_size2, alpha, beta, tensor_x, tensor_qkv, tensor_xqkv, tensor_dropout, tensor_ref_xqkv, tensor_ref_dropout);
      if (result != cudaSuccess) {
        return 1;
      }
    }

    result = runAttention<Gemm>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, tensor_x, tensor_qkv, tensor_xqkv, tensor_dropout, baselineHandle, producer_stream, producer_stream, event, NULL, false, baselineTime, warmup);

    CUDA_CHECK(cudaDeviceSynchronize());

    // double startTime = convertTimeValToDouble(getTimeOfDay());    
    result = runAttention<Gemm>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, tensor_x, tensor_qkv, tensor_xqkv, tensor_dropout, baselineHandle, producer_stream, producer_stream, event, NULL, false, baselineTime, epochs);

    if (result != cudaSuccess) {
      std::cerr << "CUTLASS GEMM kernel failed: "
        << cudaGetErrorString(result) << std::endl;
      return result;
    }
    CUDA_CHECK(cudaStreamSynchronize(producer_stream));
    // double endTime = convertTimeValToDouble(getTimeOfDay());
    // baselineTime = endTime - startTime;
    printf("cutlass-baseline elapsedtime %lf microseconds\n", baselineTime/(float)epochs);
  }

  return 0;
  #if 0
  double minimumTime = (1<<20);
  if (true) {
    minimumTime = 0;
    cudaStream_t consumer_stream;
    CUDA_CHECK(cudaStreamCreate(&consumer_stream));
    result = runhgemm<Gemm, Gemm, GemmSplitK, GemmSplitK>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, baselineHandle, producer_stream, producer_stream, event, NULL, false, minimumTime, epochs);

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

  dim3 grid2Dim = {DIVUP(problem_size2.m(), ShapeMMAThreadBlock::kM), DIVUP(problem_size2.n(), ShapeMMAThreadBlock::kN), split_k2};
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
    result = runhgemm<OverlapGemm1, OverlapGemm2, OverlapGemmSplitK, OverlapGemmSplitK>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, overlapHandle, producer_stream, consumer_stream,  event, kernelExecuted, rowSyncOrTileSync, overlapTime, 1);

    CUDA_CHECK(cudaDeviceSynchronize());
    if (doChecking) {
      result = check_results(problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_ref_c, tensor_d, tensor_ref_e, tensor_c, tensor_e);
      if (result != cudaSuccess) {
        return 1;
      }
    }
    //warmup
    result = runhgemm<OverlapGemm1, OverlapGemm2, OverlapGemmSplitK, OverlapGemmSplitK>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, overlapHandle, producer_stream, consumer_stream,  event, kernelExecuted, rowSyncOrTileSync, overlapTime, warmup);

    CUDA_CHECK(cudaDeviceSynchronize());
    printf("728:\n");
    // double startTime = convertTimeValToDouble(getTimeOfDay());
    result = runhgemm<OverlapGemm1, OverlapGemm2, OverlapGemmSplitK, OverlapGemmSplitK>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, overlapHandle, producer_stream, consumer_stream,  event, kernelExecuted, rowSyncOrTileSync, overlapTime, epochs);
         // double startTime = convertTimeValToDouble(getTimeOfDay());
  //  if (split_k1 == 1 && split_k2 == 1) {
  //     #ifdef ENABLE_NORMAL_GEMM
  //     result = runhgemm<OverlapGemm1, OverlapGemm2>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, overlapHandle, producer_stream, consumer_stream, event, kernelExecuted, rowSyncOrTileSync, overlapTime, epochs);
  //     #endif
  //   } else if (split_k1 > 1 && split_k2 == 1) {
  //     #ifdef ENABLE_NORMAL_GEMM
  //     result = runhgemm<OverlapGemmSplitK, OverlapGemm2>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, overlapHandle, producer_stream, consumer_stream, event, kernelExecuted, rowSyncOrTileSync, overlapTime, epochs);
  //     #endif
  //   } else if (split_k1 == 1 && split_k2 > 1) {
  //     #ifdef ENABLE_NORMAL_GEMM
  //     result = runhgemm<OverlapGemm1, OverlapGemmSplitK>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, overlapHandle, producer_stream, consumer_stream, event, kernelExecuted, rowSyncOrTileSync, overlapTime, epochs);
  //     #endif
  //   } else {
  //     result = runhgemm<OverlapGemmSplitK, OverlapGemmSplitK>(split_k1, split_k2, problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, overlapHandle, producer_stream, consumer_stream, event, kernelExecuted, rowSyncOrTileSync, overlapTime, epochs);
  //   }
  
    
    CUDA_CHECK(cudaStreamSynchronize(consumer_stream));
    CUDA_CHECK(cudaStreamSynchronize(producer_stream));
    // double endTime = convertTimeValToDouble(getTimeOfDay());
    // overlapTime = endTime - startTime;

    printf("overlapped elapsedtime %lf microseconds\n", overlapTime/(float)epochs);
  }
  #endif

  return 0;
}