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


// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = cutlass::half_t;                   // <- data type of accumulator
using ElementComputeEpilogue = cutlass::half_t;  // <- data type of epilogue operations
using ElementInputA = cutlass::half_t;              // <- data type of elements in input matrix A
using ElementInputB = cutlass::half_t;              // <- data type of elements in input matrix B
using ElementOutput = cutlass::half_t;                        // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices. Column Major for
// Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm70;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 128, 32>;  // <- threadblock tile M = 128, N = 128, K = 32
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;  // <- warp tile M = 64, N = 64, K = 32 
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;  // <- MMA Op tile M = 8, N = 8, K = 4

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; //;  // <- ??

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
                                         SwizzleThreadBlock>;

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
  // Create instantiation for device reference gemm kernel
  cutlass::reference::device::Gemm<ElementInputA,
  LayoutInputA,
  ElementInputB,
  LayoutInputB,
  ElementOutput,
  LayoutOutput,
  ElementComputeEpilogue,
  ElementComputeEpilogue>
  gemm_device1;
  cutlass::reference::device::Gemm<ElementInputA,
  LayoutInputA,
  ElementInputB,
  LayoutInputB,
  ElementOutput,
  LayoutOutput,
  ElementComputeEpilogue,
  ElementComputeEpilogue>
  gemm_device2;
  // Launch device reference gemm kernel
  gemm_device1(problem_size1,
              alpha,
              tensor_a.device_ref(),
              tensor_b.device_ref(),
              beta,
              tensor_ref_c.device_ref(),
              tensor_ref_c.device_ref());

  // Wait for kernels to finish
  cudaDeviceSynchronize();

  // Copy output data from CUTLASS and reference kernel to host for comparison
  tensor_c.sync_host();
  tensor_ref_c.sync_host();
  bool passed;
  // Check if output from CUTLASS kernel and reference kernel are equal or not
  // passed = cutlass::reference::host::TensorEquals(
  //   tensor_c.host_view(),
  //   tensor_ref_c.host_view());

  // if (!passed) {
  //   std::cout << "Wrong results for C = A * B" << std::endl;
  //   return cudaErrorUnknown;
  // }
  tensor_ref_c.sync_device();
  cudaDeviceSynchronize();

  // Launch device reference gemm kernel
  std::cout << "283: " << problem_size2.m() << " " << problem_size2.n() << " " << problem_size2.k() << std::endl;
  gemm_device2(problem_size2,
    alpha,
    tensor_ref_c.device_ref(),
    tensor_d.device_ref(),
    beta,
    tensor_ref_e.device_ref(),
    tensor_ref_e.device_ref());

  // Wait for kernels to finish
  cudaDeviceSynchronize();

  // Copy output data from CUTLASS and reference kernel to host for comparison
  tensor_e.sync_host();
  tensor_ref_e.sync_host();

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  passed = cutlass::reference::host::TensorEquals(
  tensor_e.host_view(),
  tensor_ref_e.host_view());

  if (!passed) {
  std::cout << "Wrong results for E = C * D" << std::endl;
  return cudaErrorUnknown;
  }
  std::cout << "passed" << std::endl;
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
                     
cudaError_t runhgemm(cutlass::gemm::GemmCoord problem_size1,
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
                     double& execTime,
                     int iters = 100) {
  
    int split_k_slices = 1;
  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename Gemm::Arguments args1{handle,
                                     problem_size1,  // <- problem size of matrix multiplication
                                     tensor_a.device_ref(),  // <- reference to matrix A on device
                                     tensor_b.device_ref(),  // <- reference to matrix B on device
                                     tensor_c.device_ref(),  // <- reference to matrix C on device
                                     tensor_c.device_ref(),  // <- reference to matrix C on device
                                     {alpha, beta},          // <- tuple of alpha and beta
                                     split_k_slices};        // <- k-dimension split factor
  
  typename Gemm::Arguments args2{handle,
                                     problem_size2,  // <- problem size of matrix multiplication
                                     tensor_c.device_ref(),  // <- reference to matrix A on device
                                     tensor_d.device_ref(),  // <- reference to matrix B on device
                                     tensor_e.device_ref(),  // <- reference to matrix C on device
                                     tensor_e.device_ref(),  // <- reference to matrix C on device
                                     {alpha, beta},          // <- tuple of alpha and beta
                                     split_k_slices};        // <- k-dimension split factor
  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(args1);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op1;

  // Check the problem size is supported or not 
  cutlass::Status status = gemm_op1.can_implement(args1);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op1.initialize(args1, workspace.get());
  CUTLASS_CHECK(status);

  Gemm gemm_op2;
  workspace_size = Gemm::get_workspace_size(args2);

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
    printf("385\n");
    // Launch initialized CUTLASS kernel
    for (int r = 0; r < iters; r++) {
      handle.iter += 1;
      
      typename Gemm::Arguments args1{handle,
        problem_size1,  // <- problem size of matrix multiplication
        tensor_a.device_ref(),  // <- reference to matrix A on device
        tensor_b.device_ref(),  // <- reference to matrix B on device
        tensor_c.device_ref(),  // <- reference to matrix C on device
        tensor_c.device_ref(),  // <- reference to matrix C on device
        {alpha, beta},          // <- tuple of alpha and beta
        split_k_slices};        // <- k-dimension split factor

      typename Gemm::Arguments args2{handle,
        problem_size2,  // <- problem size of matrix multiplication
        tensor_c.device_ref(),  // <- reference to matrix A on device
        tensor_d.device_ref(),  // <- reference to matrix B on device
        tensor_e.device_ref(),  // <- reference to matrix C on device
        tensor_e.device_ref(),  // <- reference to matrix C on device
        {alpha, beta},          // <- tuple of alpha and beta
        split_k_slices};        // <- k-dimension split factor
      
      handle.producerOrConsumer_ = true;
      double start = timeInMicroSeconds();
      status = gemm_op1(args1, false, NULL, producer_stream);
      CUTLASS_CHECK(status);
      
      if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
      }

      handle.producerOrConsumer_ = false;
      // CUDA_CHECK(cudaStreamSynchronize(consumer_stream));

      status = gemm_op2(args2, false, NULL, consumer_stream);
      CUTLASS_CHECK(status);

      if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
      }

      CUDA_CHECK(cudaDeviceSynchronize());
      double end = timeInMicroSeconds();
      execTime += end-start;
    }
  } else {
    // Launch initialized CUTLASS kernel
    for (int r = 0; r < iters; r++) {
      handle.iter += 1;
      handle.producerOrConsumer_ = true;
      typename Gemm::Arguments args1{handle,
        problem_size1,  // <- problem size of matrix multiplication
        tensor_a.device_ref(),  // <- reference to matrix A on device
        tensor_b.device_ref(),  // <- reference to matrix B on device
        tensor_c.device_ref(),  // <- reference to matrix C on device
        tensor_c.device_ref(),  // <- reference to matrix C on device
        {alpha, beta},          // <- tuple of alpha and beta
        split_k_slices};        // <- k-dimension split factor
      
      handle.producerOrConsumer_ = false;
      typename Gemm::Arguments args2{handle,
        problem_size2,  // <- problem size of matrix multiplication
        tensor_c.device_ref(),  // <- reference to matrix A on device
        tensor_d.device_ref(),  // <- reference to matrix B on device
        tensor_e.device_ref(),  // <- reference to matrix C on device
        tensor_e.device_ref(),  // <- reference to matrix C on device
        {alpha, beta},          // <- tuple of alpha and beta
        split_k_slices};        // <- k-dimension split factor
      
      double start = timeInMicroSeconds();
      dim3 grid = {problem_size1.m()/128, 1, 1};
      int lastBlockIdxX = (grid.x/80)*80;
      status = gemm_op1(args1, true, 0, grid.x, (int*)kernelExecuted, NULL, producer_stream);
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
      status = gemm_op2(args2, true, 0, grid.x,  (int*)kernelExecuted, NULL, consumer_stream);
      CUTLASS_CHECK(status);

      if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
      }
      CUDA_CHECK(cudaDeviceSynchronize());
      // CUDA_CHECK(cudaStreamSynchronize(consumer_stream));
      // CUDA_CHECK(cudaStreamSynchronize(producer_stream));
      double end = timeInMicroSeconds();
      execTime += end-start;
    }
  }

  return cudaSuccess;
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

  // GEMM problem dimensions.
  int problem[4] = {128, 128, 128, 128 };

  for (int i = 1; i < argc && i < 5; ++i) {
    std::stringstream ss(arg[i]);
    ss >> problem[i - 1];
  }

  // Scalars used for linear scaling the result of the matrix product.
  float scalars[2] = { 1, 0 };

  for (int i = 5; i < argc && i < 7; ++i) {
    std::stringstream ss(arg[i]);
    ss >> scalars[i - 4];
  }

  //
  // Run the CUTLASS GEMM test.
  //

  cudaStream_t producer_stream;
  cudaStream_t consumer_stream;
  CUDA_CHECK(cudaStreamCreate(&producer_stream));
  CUDA_CHECK(cudaStreamCreate(&consumer_stream));
  
  printf("problem[0] %d problem[1] %d problem[2] %d problem[3] %d\n", problem[0], problem[1], problem[2], problem[3]);

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
  cutlass::reference::host::TensorFill(
    tensor_a.host_view(),
    ElementOutput(0.05));  // <- Fill matrix B on host with uniform-distribution random data
  cutlass::reference::host::TensorFill(
    tensor_b.host_view(),
    ElementOutput(0.05));  // <- Fill matrix B on host with uniform-distribution random data
  cutlass::reference::host::TensorFill(
    tensor_d.host_view(),
    ElementOutput(0.05));  // <- Fill matrix B on host with uniform-distribution random data

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
  int epochs = 100;
  int warmup = 10;
  double baselineTime = 0;
  cudaEvent_t start;
  cudaEvent_t end;
  cudaEvent_t event;
  CUDA_CHECK(cudaEventCreate(&event));
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));
  if (true) {
    result = runhgemm(problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, baselineHandle, producer_stream, producer_stream, event, NULL, baselineTime, 1);

    CUDA_CHECK(cudaDeviceSynchronize());

    result = check_results(problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_ref_c, tensor_d, tensor_ref_e, tensor_c, tensor_e);
    if (result != cudaSuccess) {
      return 1;
    }

    //warmup
    result = runhgemm(problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, baselineHandle, producer_stream, producer_stream, event, NULL, baselineTime, warmup);

    CUDA_CHECK(cudaDeviceSynchronize());

    // double startTime = convertTimeValToDouble(getTimeOfDay());    
    result = runhgemm(problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, baselineHandle, producer_stream, producer_stream, event, NULL, baselineTime, epochs);

    if (result != cudaSuccess) {
      std::cerr << "CUTLASS GEMM kernel failed: "
        << cudaGetErrorString(result) << std::endl;
      return result;
    }
    // CUDA_CHECK(cudaStreamSynchronize(producer_stream));
    // double endTime = convertTimeValToDouble(getTimeOfDay());
    // baselineTime = endTime - startTime;
    printf("baseline elapsedtime %lf microseconds\n", baselineTime/(float)epochs);
  }

  double minimumTime = (1<<20);
  if (false) {
    minimumTime = 0;
    cudaStream_t consumer_stream;
    CUDA_CHECK(cudaStreamCreate(&consumer_stream));
    result = runhgemm(problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, baselineHandle, producer_stream, consumer_stream, event, NULL, minimumTime, epochs);

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
    tensor_ref_c.host_view());  // <- Fill matrix C on host with zeros
  cutlass::reference::host::TensorFill(
    tensor_e.host_view());  // <- fill matrix E on host with zeros
  cutlass::reference::host::TensorFill(
    tensor_ref_e.host_view());  // <- fill matrix E on host with zeros

    
  tensor_c.sync_device();
  tensor_ref_c.sync_device();

  tensor_e.sync_device();
  tensor_ref_e.sync_device();

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
  
  
  
  overlapHandle.allocTileStatusMap(128, 128, 1);
  double overlapTime = 0;
  dim3 gridDim = {problem_size1.m()/128, problem_size1.n()/128, 1};
  
  int* dBlockIndexOrder;
  CUDA_CHECK(cudaMalloc(&dBlockIndexOrder, sizeof(int) * gridDim.x * gridDim.y * 2));
  CUDA_CHECK(cudaMemset(dBlockIndexOrder, 0, sizeof(int) * gridDim.x * gridDim.y * 2));

  int* hBlockIndexOrder = new int[gridDim.x * gridDim.y * 2];
  int linearid = 0;
  for (int x = 0; x < gridDim.x; x++) {
  for (int y = 0; y < gridDim.y; y++) {
    hBlockIndexOrder[linearid] = x;
    hBlockIndexOrder[linearid + 1] = y;
    // printf("linearid %d x %d y %d\n", linearid, x, y);
    linearid += 2;
  }
  }

  printf("dBlockIndexOrder %p\n", dBlockIndexOrder);
  CUDA_CHECK(cudaMemcpy(dBlockIndexOrder, hBlockIndexOrder, sizeof(int) * gridDim.x * gridDim.y * 2, cudaMemcpyHostToDevice));

  int* dIsRemainingBlock;
  int* hIsRemainingBlock = new int[gridDim.x*gridDim.y];
  CUDA_CHECK(cudaMalloc(&dIsRemainingBlock, sizeof(int)*gridDim.x*gridDim.y));
  int totalBlocks = 0;
  const int startRemainingBlockId = ((gridDim.x*gridDim.y)/(3*80))*(3*80) + 1;
  printf("startRemainingBlockId %d to %d\n", startRemainingBlockId, gridDim.x*gridDim.y);
  for (int x = 0; x < gridDim.x; x++) {
    for (int y = 0; y < gridDim.y; y++) {
      if (totalBlocks >= startRemainingBlockId) {
        hIsRemainingBlock[totalBlocks] = 1;
      } else {
        hIsRemainingBlock[totalBlocks] = 0;
      }

      totalBlocks++;
    }
  }

  CUDA_CHECK(cudaMemcpy(dIsRemainingBlock, hIsRemainingBlock, sizeof(int) * gridDim.x * gridDim.y, cudaMemcpyHostToDevice));

  overlapHandle.isBlockRemaining = dIsRemainingBlock;
  overlapHandle.blockIndexOrder = dBlockIndexOrder;

  if (true) {
    result = runhgemm(problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, overlapHandle, producer_stream, consumer_stream,  event, kernelExecuted, overlapTime, 1);

    CUDA_CHECK(cudaDeviceSynchronize());

    result = check_results(problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_ref_c, tensor_d, tensor_ref_e, tensor_c, tensor_e);
    if (result != cudaSuccess) {
      return 1;
    }

    //warmup
    result = runhgemm(problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, overlapHandle, producer_stream, consumer_stream, event, kernelExecuted, overlapTime, warmup);

    CUDA_CHECK(cudaDeviceSynchronize());
    printf("728:\n");
    // double startTime = convertTimeValToDouble(getTimeOfDay());
    result = runhgemm(problem_size1, problem_size2, alpha, beta, tensor_a, tensor_b, tensor_c, tensor_d, tensor_e, overlapHandle, producer_stream, consumer_stream, event, kernelExecuted, overlapTime, epochs);
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

