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


This example shows how to run convolution kernels using functions and data structures
provided by CUTLASS using tensor cores; which we run on a NVIDIA Turing GPU.

Writing a single high performance convolution kernel is hard but do-able. Whereas writing
high performance kernels at scale which works for multiple problem sizes with good abstractions is
really hard. CUTLASS solves this problem by providing simplified abstractions to compose
multiple sections of implicit gemm kernel. When used properly, the kernels can hit peak performance
of GPU easily.

CUTLASS divides a kernel into hierarchical composable sections. Which means, at each thread, warp
and thread-block level, they compute on their own tile-size with higher level of tile sizes being
composed from lower level ones. Multiple thread-tiles (tile size each thread computes) can be used
to form warp-tiles (tile size each warp computes) and multiple warp tiles can be used to compute
threadblock-tile (tile size computed by a threadblock).

In thie example, we split variable initialization into
1. Setting up data properties : describes how tensors are laid out in the memory and how the kernel
can view them (logical to physical mapping)
2. Setting up computation properties : describes how the above set tensors will be used to compute
output of convolution.

First, we setup the data types of the input tensor A, weights' tensor B and output tensor C along
with alpha, beta as the equation for convolution is C = alpha * Conv(A, B) + beta * C. In CUTLASS,
the kernels first compute Conv(A, B) and leave the rest of the computation to end of the kernel as
alpha * X + beta * C is a simple element-wise operation on X (Conv(A, B)) and C. We call this as 
epilogue of kernel. Hence, we setup data types for alpha and beta to be equal to 
ElementComputeEpilogue = float. We want to use MMA instructions on Turing and they support 4-bit
signed integer. But int4b_t is not fully supported by Nvidia software stack, so CUTLASS introduces
cutlass::int4b_t. We use the data type for elements in input tensor A and B as cutlass::int4b_t. We
convey this to CUTLASS kernel by initializing template variables ElementAccumulator (int32_t),
ElementComputeEpilogue (float), ElementInputA (cutlass::int4b_t), ElementInputB (cutlass::int4b_t),
ElementOutput (int32_t). Communicating just the data type is not enough. As the data is laid out 
linearly in memory, we have to convey the layout of tensors. We do that by initializing template
variables LayoutInputA, LayoutInputB and LayoutOutput to TensorNHWC cutlass variable. Next, we setup
rules to comptue alpha * X + beta * C which is called epilogue of the kernel. We initialize template
variable EpilogueOp, which takes the data type of output ElementOutput (int32_t), the number of
elements per vector memory access (32), data type of accumulator (int32_t) and data type of
computation of linear combination (alpha * X + beta * C).

Now that we setup the properties of data, we have to setup properties of computation.

Second, we create template variables of tile sizes for thread-block, warp and mma-op to 128x128x128,
64x64x128, 8x8x32 (MxNxK) respectively. When passed to instantiate CUTLASS Implicit GEMM kernel, it
internally deduces the amount of threads needed per thread-block, amount of shared memory, storing
data in bank-conflict free manner, and ton of other variables required to compose, intialize and
launch a high performance Implicit GEMM kernel. This is the beauty of CUTLASS, it relieves developer
from understanding and coding complicated hardware optimizations which can easily go wrong.

CUTLASS also supports multiple MMA pipelines in a threadblock. What are MMA pipelines? MMA pipelines
constitute the whole process of loading input data from global memory to shared memory, loading data
from shared memory to registers, doing matrix multiplication, store to global memory. The below flow
sequence shows a typical mma pipeline.

tensor in global memory -> registers -> tile in shared memory -> registers -> mma -> registers ->
output to global memory

The problem with single pipeline is, each stage is synchronous which means, each stage has to wait
until the previous finished executing. There are stages in the pipeline which do not have fixed
latency, for example, the loads from global memory and shared memory. Therefore, we can add one more
pipeline with a phase shift in mma kernel to hide latency from global and shared memory loads.
Finally, the pipeline in a kernel looks like

(1) tensor in global memory -> (2) registers -> (3) tile in shared memory -> (4) registers -> (5)
mma -> (6) registers -> (7) output to global memory (1) <null> -> (2) <null> -> (3) tensor in global
memory -> (4) registers -> (5) tile in shared memory -> (6) registers -> (7) mma -> (8) registers ->
(9) output to global memory

This way, you can hide the second global memory load latency by doing computation on already loaded
input data.

There are few more template variables initialized such as, which threadblock tile of output matrix
is done which threadblock launched on an SM, CUDA SM architecture of GPU you want to run on.

These are all put together to create a template variable which describes CUTLASS Implicit GEMM
kernel using cutlass::conv::device::ImplicitGemm template.

The next step is to intialize physical data, instantiate and initialize CUTLASS kernel and run it.
We use CUTLASS utilities to initialize, fill, compare tensors as they are simple and doesn't come
in the way of learning CUTLASS.

Once all the tensors are initialized and filled with data, create arguments tuple to launch CUTLASS
kernel which takes problem size (N = 1, H = 64, W = 64, C = 128), filter size (K = 64,
R = 3, S = 3, C = 128 ), padding, strides, dilation, tensors, alpha, beta and the
important one, split k-dimension factor. Along with that, we query CUTLASS if any scratch-space
memory required by the kernel we instantiated. If yes, we create it and pass it along with other
arguments created to intialize CUTLASS kernel then, the kernel is launched.

In this example, we later on launch a reference convolution kernel (from CUTLASS utilities) to
compare if the output from CUTLASS kernel is same as the reference implicit GEMM kernel.
*/

#include <iostream>
#include <fstream>
#include <sstream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/convolution.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/convolution.h"
#include "cutlass/util/tensor_view_io.h"

#include "helper.h"
#include<time.h>
#include<sys/time.h>
#include<overlap_handle.h>

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

// The code section below describes datatype for input, output tensors and computation between
// elements
using ElementAccumulator = float;                 // Data type of accumulator
using ElementComputeEpilogue = cutlass::half_t;               // Data type of epilogue computation (alpha, beta)
using ElementInputA = cutlass::half_t;             // Data type of elements in input tensor
using ElementInputB = ElementInputA;             // Data type of elements in input tensor
using ElementOutput = ElementInputB;             // Data type of elements in output tensor

using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm70;

// This code section describes the tile size a thread block will compute
using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;  // Threadblock tile shape

// This code section describes tile size a warp will compute
using WarpShape = cutlass::gemm::GemmShape<32, 32, 32>;         // Warp tile shape

// This code section describes the size of MMA op
using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;    // TensorCore instruction shape

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// Number of pipelines you want to use
constexpr int NumStages = 2;

// This code section describes the epilogue part of the kernel, we use default value
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // Data type of output matrix.
    8,                                                 // The number of elements per vectorized.
                                                       // memory access. This becomes the vector width of
                                                       // math instructions in the epilogue too.
    ElementAccumulator,                                // Data type of accumulator
    ElementComputeEpilogue>;                           // Data type for alpha/beta in linear combination


using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
  ElementInputA, LayoutInputA,
  ElementInputB, LayoutInputB,
  ElementOutput, LayoutOutput,
  ElementAccumulator,
  MMAOp,
  SmArch,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOp,
  SwizzleThreadBlock,
  NumStages,
  cutlass::arch::OpMultiplyAdd,
  cutlass::conv::IteratorAlgorithm::kAnalytic
>::Kernel;

using ImplicitGemm1 = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;
using ImplicitGemm2 = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;


/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;
  cutlass::Tensor4DCoord input_size;
  cutlass::Tensor4DCoord filter_size;
  cutlass::Tensor4DCoord padding;
  cutlass::MatrixCoord conv_stride;
  cutlass::MatrixCoord dilation;
  bool reference_check;
  bool measure_performance;
  int iterations;
  bool save_workspace;
  ElementComputeEpilogue alpha;
  ElementComputeEpilogue beta;
  bool benchmark;
  std::string tag;
  int split_k_slices;
  bool rowSyncOrTileSync;

  Options():
    help(false),
    input_size(1, 32, 32, 32),
    filter_size(32, 3, 3, 32),
    padding(1, 1, 1, 1),
    conv_stride(1, 1),
    dilation(1, 1),
    reference_check(false),
    measure_performance(true),
    iterations(20),
    save_workspace(false),
    alpha(1),
    beta(0),
    split_k_slices(0),
    rowSyncOrTileSync(false),
    benchmark(false) { }

  // Verify the problem size is compatible with the CUTLASS Convolution implementation.
  bool valid() {

    //
    // CUTLASS attempts to load 128b vectors of int4b_t elements. Consequently,
    // all pointers, strides, and tensor extents must be divisible by 32 elements.
    //
    int const kAlignment = 32;

    if ((input_size.c() % kAlignment) ||
      (filter_size.n() % kAlignment)) {

      // misaligned tensors
      return false;
    }

    // Invalid padding
    if ((padding.h() != filter_size.h() / 2) ||
      (padding.w() != filter_size.w() / 2)) {

      return false;
    }

    return true;
  }

  /// Updates input and filter sizes
  void update(
    cutlass::Tensor4DCoord input_size,
    cutlass::Tensor4DCoord filter_size) {

    this->input_size = input_size;
    this->filter_size = filter_size;

    padding.n() = filter_size.h() / 2;
    padding.h() = filter_size.h() / 2;
    padding.w() = filter_size.w() / 2;
    padding.c() = filter_size.w() / 2;
  }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    if (cmd.check_cmd_line_flag("ref-check")) {
      reference_check = true;
    }

    if (cmd.check_cmd_line_flag("perf-check")) {
      measure_performance = true;
    }

    if (cmd.check_cmd_line_flag("save-workspace")) {
      save_workspace = true;
    }

    if (cmd.check_cmd_line_flag("benchmark")) {
      benchmark = true;
    }

    cmd.get_cmd_line_argument("split_k_slices", split_k_slices);
  
    cmd.get_cmd_line_argument("n", input_size.n());
    cmd.get_cmd_line_argument("h", input_size.h());
    cmd.get_cmd_line_argument("w", input_size.w());
    cmd.get_cmd_line_argument("c", input_size.c());

    cmd.get_cmd_line_argument("k", filter_size.n());
    cmd.get_cmd_line_argument("r", filter_size.h());
    cmd.get_cmd_line_argument("s", filter_size.w());
    filter_size.c() = input_size.c(); 

    int syncType;
    cmd.get_cmd_line_argument("syncType", syncType);
    if (syncType == 0) rowSyncOrTileSync = false;
    else rowSyncOrTileSync = true;

    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);
    
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("tag", tag);

    if (filter_size.h() == 3 && filter_size.w() == 3) {
      padding = {1, 1, 1, 1};
    }
    else {
      filter_size.h() = 1;
      filter_size.w() = 1;
      padding = {0, 0, 0, 0};
    }
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "09_turing_tensorop_conv2dfprop example\n\n"
      << "  This example uses Turing's Tensor Core operators on int4 data types to compute\n"
      << "  forward convolution on tensors of layout NHWC.\n\n"
      << "Options:\n\n"
      << "  --help               If specified, displays this usage statement.\n\n"
      << "  --n=<int>            Input tensor extent N\n"
      << "  --h=<int>            Input tensor extent H\n"
      << "  --w=<int>            Input tensor extent W\n"
      << "  --c=<int>            Input tensor extent C\n"
      << "  --k=<int>            Filter extent K\n"
      << "  --r=<int>            Filter extent R\n"
      << "  --s=<int>            Filter extent S\n\n"
      << "  --alpha=<float>      Epilogue scalar alpha\n"
      << "  --beta=<float>       Epilogue scalar beta\n\n"
      << "  --ref-check          If set (true), reference check on the host is computed\n"
      << "  --perf-check         If set (true), performance is measured.\n"
      << "  --benchmark          If set (true), performance benchmarking on several layers and batch-size.\n"
      << "  --iterations=<int>   Number of profiling iterations to perform.\n"
      << "  --save-workspace     If set, workspace is written to a text file.\n"
      << "  --tag=<string>       String to replicate across the first column in the results table\n"
      << "  --split_k_slices=<int> ";

    out << "\n\nExamples:\n\n"
      << "$ ./examples/09_turing_tensorop_conv2dfprop/09_turing_tensorop_conv2dfprop  --n=32 --h=224 --w=224 --c=128 --k=256 --r=1 --s=1\n\n"
      << "$ ./examples/09_turing_tensorop_conv2dfprop/09_turing_tensorop_conv2dfprop  --n=1 --h=224 --w=224 --c=32 --k=32 --r=3 --s=3 --ref-check\n\n";

    return out;
  }
  
  /// Computes the output tensor size (NPQK)
  cutlass::Tensor4DCoord output_size() const {
    return cutlass::Tensor4DCoord(
      input_size.n(),
      (input_size.h() + padding.n() + padding.h() - filter_size.h()) / conv_stride.row() + 1,
      (input_size.w() + padding.w() + padding.c() - filter_size.w()) / conv_stride.column() + 1,
      filter_size.n());
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {

    // Number of multiply-adds = NPQK * CRS
    int64_t fmas = output_size().product() * int64_t(filter_size.h() * filter_size.w() * filter_size.c());
    
    // Two flops per multiply-add
    return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

struct Result {
  double runtime_ms;
  double gflops;
  cutlass::Status status;
  cutlass::Status reference_check;
  cudaError_t error;

  Result(): 
    runtime_ms(0), 
    gflops(0),
    status(cutlass::Status::kSuccess),
    reference_check(cutlass::Status::kInvalid),
    error(cudaSuccess) { }

  static std::ostream & print_header(std::ostream &out, Options const &options) {

    if (!options.tag.empty()) {
      out << "Name,";
    }

    out << "Layer,N,H,W,C,K,R,S,Runtime,GFLOPs";

    return out;
  }

  std::ostream & print(std::ostream &out, int idx, Options const &options) {

    if (!options.tag.empty()) {
      out << options.tag << ",";
    }

    out 
      << "conv_" << idx << ","
      << options.input_size.n() << ","
      << options.input_size.h() << ","
      << options.input_size.w() << ","
      << options.input_size.c() << ","
      << options.filter_size.n() << ","
      << options.filter_size.h() << ","
      << options.filter_size.w() << ","
      << runtime_ms << ","
      << gflops;

    return out;
  }
};

__device__ inline uint glLoad(volatile uint* addr) {
  uint val;
  // asm ("ld.volatile.global.u32 {%0}, [%1];" : "=r"(val) : "l"(addr));
  return *addr;
}

__global__ void waitKernel(volatile uint* kernelExecuted, uint expectedValue) {
  if (threadIdx.x == 0) {
    // printf("expectedValue %d\n", expectedValue);
    uint v = glLoad(kernelExecuted);
    while(v < expectedValue) {
      v = glLoad(kernelExecuted);
    }
  }
}
/////////////////////////////////////////////////////////////////////////////////////////////////

template<bool baselineOrOverlap, typename ImplicitGemm1, typename ImplicitGemm2, typename TensorA, typename TensorB, typename TensorC>
void runConvolution(cutlass::conv::Conv2dProblemSize problem_size, const Options& options, cudaStream_t* streams, OverlapHandle& overlapHandle,
                    TensorA& tensor_x, TensorB& tensor_w1, TensorB& tensor_w2, TensorC& tensor_y1, TensorC& tensor_y2, 
                    volatile uint* kernelExecuted,
                    double& elapsedTime, double& conv1Time, double& conv2Time, int runs) {
  // Construct ImplicitGemm::Argument structure with conv2d 
  // problem size, data pointers, and epilogue values
  typename ImplicitGemm1::Arguments args1{
    overlapHandle,
    problem_size,
    tensor_x.device_ref(),
    tensor_w1.device_ref(),
    tensor_y1.device_ref(),
    tensor_y1.device_ref(),
    {options.alpha, options.beta},
  };

  typename ImplicitGemm2::Arguments args2{
    overlapHandle,
    problem_size,
    tensor_y1.device_ref(),
    tensor_w2.device_ref(),
    tensor_y2.device_ref(),
    tensor_y2.device_ref(),
    {options.alpha, options.beta},
  };

  //
  // Initialize CUTLASS Convolution
  //

  ImplicitGemm1 implicit_gemm_op1;
  ImplicitGemm2 implicit_gemm_op2;
  
    size_t workspace_size1 = implicit_gemm_op1.get_workspace_size(args1);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace1(workspace_size1);

    auto status = implicit_gemm_op1.can_implement(args1);
    CUTLASS_CHECK(status);

    status = implicit_gemm_op1.initialize(args1, workspace1.get());
    CUTLASS_CHECK(status);
  
    size_t workspace_size2 = implicit_gemm_op2.get_workspace_size(args2);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace2(workspace_size2);

    status = implicit_gemm_op2.can_implement(args2);
    CUTLASS_CHECK(status);

    status = implicit_gemm_op2.initialize(args2, workspace2.get());
    CUTLASS_CHECK(status);

  if (baselineOrOverlap == true) {
    for (int i = 0; i < runs; i++) {
      double start = getCurrentTime();
      auto status = implicit_gemm_op1(args1, workspace1.get(), streams[0]);

      CUTLASS_CHECK(status);
      cudaDeviceSynchronize();
      double middle1 = getCurrentTime();
      conv1Time += middle1 - start;
      status = implicit_gemm_op2(args2, workspace2.get(), streams[0]);

      CUTLASS_CHECK(status);
      cudaDeviceSynchronize();
      double end = getCurrentTime();
      conv2Time += end - middle1;
      elapsedTime += end - start;
      printf("{\"Total\": %lf, \"conv1\": %lf, \"conv2\": %lf}\n",end-start,middle1-start,end-middle1);
    }
  } else {
    for (int i = 0; i < runs; i++) {
      args1.overlap_handle.iter += 1;
      args2.overlap_handle.iter += 1;
      double start = getCurrentTime();
      args1.overlap_handle.producerOrConsumer_ = true;
      auto status = implicit_gemm_op1(args1, true, options.rowSyncOrTileSync, kernelExecuted, workspace1.get(), streams[0]);
      // waitKernel<<<1,1,0,streams[1]>>>((uint*)&kernelExecuted[0], args1.overlap_handle.iter);

      CUTLASS_CHECK(status);
      // cudaDeviceSynchronize();
      args2.overlap_handle.producerOrConsumer_ = false;
      // double middle1 = getCurrentTime();
      // conv1Time += middle1 - start;
      status = implicit_gemm_op2(args2, true, options.rowSyncOrTileSync, kernelExecuted, workspace2.get(), streams[1]);

      CUTLASS_CHECK(status);
      cudaDeviceSynchronize();
      double end = getCurrentTime();
      // conv2Time += end - middle1;
      elapsedTime += end - start;
      printf("{\"Total\": %lf, \"conv1\": %lf, \"conv2\": %lf}\n",end-start,conv1Time,conv2Time);
    }
  }
  overlapHandle.iter = args1.overlap_handle.iter;
}

/// Runs one benchmark
Result profile_convolution(Options const &options) {
  // Check the problem size is supported or not 

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

  Result result;

  //
  // Allocate host-device tensors using the CUTLASS Utilities.
  //

  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_x(options.input_size);
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_w1(options.filter_size);
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_w2(options.filter_size);
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_y1(options.output_size());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_y2(options.output_size());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_y1(options.output_size());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_y2(options.output_size());

  //
  // Initialize tensors
  //

  cutlass::reference::host::TensorFillRandomUniform(
      tensor_x.host_view(), ElementInputA(1.0f));
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_w1.host_view(), ElementInputB(1.0f));
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_y1.host_view(), ElementOutput(1.0f));
  // Fill tensor A on host with uniform-distribution random data
  // cutlass::reference::host::TensorFillRandomUniform(
  //     tensor_x.host_view(),
  //     1,
  //     ElementInputA(1),
  //     ElementInputA(-1),
  //     0);

  // // Fill tensor B on host with uniform-distribution random data
  // cutlass::reference::host::TensorFillRandomUniform(
  //     tensor_w1.host_view(),
  //     1,
  //     ElementInputB(1),
  //     ElementInputB(-1),
  //     0);
  
  // // Fill tensor B on host with uniform-distribution random data
  // cutlass::reference::host::TensorFillRandomUniform(
  //   tensor_w2.host_view(),
  //   1,
  //   ElementInputB(1),
  //   ElementInputB(-1),
  //   0);

  // Fill tensor C on host with zeros
  cutlass::reference::host::TensorFill(
    tensor_y1.host_view());
  cutlass::reference::host::TensorFill(
    tensor_y2.host_view());

  // Fill tensor C for reference on host with zeros
  cutlass::reference::host::TensorFill(
    tensor_ref_y1.host_view());
  cutlass::reference::host::TensorFill(
    tensor_ref_y2.host_view());
  
  // Copy data from host to GPU
  tensor_x.sync_device();
  tensor_w1.sync_device();
  tensor_w2.sync_device();
  tensor_y1.sync_device();
  tensor_y2.sync_device();
  tensor_ref_y1.sync_device();
  tensor_ref_y2.sync_device();

  // mode (kCrossCorrelation or kConvolution)
  cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

  // Construct Conv2dProblemSize with user defined output size
  cutlass::conv::Conv2dProblemSize problem_size(
      options.input_size,
      options.filter_size,
      options.padding,
      options.conv_stride,
      options.dilation,
      options.output_size(),
      mode,
      options.split_k_slices);
  //
  // Optional reference check
  //

  int warmup = 5;
  int epochs = 20;
  double elapsedTime = 0;
  double conv1Time = 0;
  double conv2Time = 0;
  
  OverlapHandle baselineHandle;
  runConvolution<true, ImplicitGemm1, ImplicitGemm2>
    (problem_size, options, &streams[0], baselineHandle, tensor_x, tensor_w1, tensor_w2, tensor_y1, tensor_y2, NULL, elapsedTime, conv1Time, conv2Time, 1);

  if (options.reference_check) {
    std::cout << "Verification on host...\n";

    // Compute with reference implementation
    cutlass::reference::host::Conv2dFprop<
      ElementInputA,
      LayoutInputA,
      ElementInputB,
      LayoutInputB,
      ElementOutput,
      LayoutOutput,
      ElementComputeEpilogue,
      ElementAccumulator,
      cutlass::NumericConverter<ElementOutput, ElementComputeEpilogue>
    >(
      problem_size,
      tensor_x.host_ref(),
      tensor_w1.host_ref(),
      tensor_y1.host_ref(),
      tensor_ref_y1.host_ref(),
      options.alpha,
      options.beta
    );

    cutlass::reference::host::Conv2dFprop<
      ElementInputA,
      LayoutInputA,
      ElementInputB,
      LayoutInputB,
      ElementOutput,
      LayoutOutput,
      ElementComputeEpilogue,
      ElementAccumulator,
      cutlass::NumericConverter<ElementOutput, ElementComputeEpilogue>
    >(
      problem_size,
      tensor_y1.host_ref(),
      tensor_w2.host_ref(),
      tensor_y2.host_ref(),
      tensor_ref_y2.host_ref(),
      options.alpha,
      options.beta
    );
    // Check if output from CUTLASS kernel and reference kernel are equal or not
    tensor_y1.sync_host(); tensor_y2.sync_host();
    bool passed = cutlass::reference::host::TensorEquals(
      tensor_y1.host_view(),
      tensor_ref_y1.host_view());

    if (!passed) {
      result.reference_check = cutlass::Status::kErrorInternal;
      std::cout << "ERROR - First conv incorrect.\n";
      return result;
    } else {
      result.reference_check = cutlass::Status::kSuccess;
      std::cout << "First Passed.\n";
    }

    passed = cutlass::reference::host::TensorEquals(
      tensor_y2.host_view(),
      tensor_ref_y2.host_view());
    
    if (!passed) {
      result.reference_check = cutlass::Status::kErrorInternal;
      std::cout << "ERROR - second conv incorrect.\n";
      return result;
    } else {
      result.reference_check = cutlass::Status::kSuccess;
      std::cout << "Second Passed.\n";
    }
  }
  if (true) {
  runConvolution<true, ImplicitGemm1, ImplicitGemm2>(problem_size, options, &streams[0], baselineHandle, tensor_x, tensor_w1, tensor_w2, tensor_y1, tensor_y2, NULL, elapsedTime, conv1Time, conv2Time, warmup);
  elapsedTime = 0;
  conv1Time = 0;
  conv2Time = 0;
  printf("START-BASELINE:\n");
  runConvolution<true, ImplicitGemm1, ImplicitGemm2>(problem_size, options, &streams[0], baselineHandle, tensor_x, tensor_w1, tensor_w2, tensor_y1, tensor_y2, NULL, elapsedTime, conv1Time, conv2Time, epochs);
  
  printf("END-BASELINE: {Total: %lf, Conv1: %lf, Conv2: %lf} micro seconds\n", elapsedTime/epochs, conv1Time/epochs, conv2Time/epochs);
  }
  auto gemm_problem_size = cutlass::conv::implicit_gemm_problem_size(cutlass::conv::Operator::kFprop, problem_size);
  printf("gemm problem size: {%d, %d, %d}\n", gemm_problem_size.m(), gemm_problem_size.n(), gemm_problem_size.k());
  printf("Number of thread blocks for both convs: {%d, %d, %d}\n", (gemm_problem_size.m()+ThreadblockShape::kM-1)/ThreadblockShape::kM,gemm_problem_size.n()/ThreadblockShape::kN, options.split_k_slices);
  OverlapHandle overlapHandle(gemm_problem_size.m(), gemm_problem_size.n(), 1, 1);
  if (options.rowSyncOrTileSync) 
    overlapHandle.waitValue = overlapHandle.ySize/ThreadblockShape::kN;
  else
    overlapHandle.waitValue =  1;
  
  overlapHandle.allocTileStatusMap(ThreadblockShape::kM, ThreadblockShape::kN, 1);
  // double overlapTime = 0;
  uint* kernelExecuted;
  CUDA_CHECK(cudaMalloc(&kernelExecuted, sizeof(uint) * 128));
  CUDA_CHECK(cudaMemset(kernelExecuted, 0, sizeof(uint) * 128));
  
  cutlass::reference::host::TensorFill(
    tensor_y1.host_view());
  cutlass::reference::host::TensorFill(
    tensor_y2.host_view());
  
  tensor_y1.sync_device();
  tensor_y2.sync_device();
      
  runConvolution<false, ImplicitGemm1, ImplicitGemm2>
    (problem_size, options, &streams[0], overlapHandle, tensor_x, tensor_w1, tensor_w2, tensor_y1, tensor_y2, kernelExecuted, elapsedTime, conv1Time, conv2Time, 1);
  
  if (options.reference_check) {
    // Check if output from CUTLASS kernel and reference kernel are equal or not
    tensor_y1.sync_host(); tensor_y2.sync_host();
    bool passed = cutlass::reference::host::TensorEquals(
      tensor_y1.host_view(),
      tensor_ref_y1.host_view());

    if (!passed) {
      result.reference_check = cutlass::Status::kErrorInternal;
      std::cout << "ERROR - First conv incorrect.\n";
      return result;
    } else {
      result.reference_check = cutlass::Status::kSuccess;
      std::cout << "First Passed.\n";
    }

    passed = cutlass::reference::host::TensorEquals(
      tensor_y2.host_view(),
      tensor_ref_y2.host_view());
    
    if (!passed) {
      result.reference_check = cutlass::Status::kErrorInternal;
      std::cout << "ERROR - second conv incorrect.\n";
      return result;
    } else {
      result.reference_check = cutlass::Status::kSuccess;
      std::cout << "Second Passed.\n";
    }
  }

  runConvolution<false, ImplicitGemm1, ImplicitGemm2>
    (problem_size, options, &streams[0], overlapHandle, tensor_x, tensor_w1, tensor_w2, tensor_y1, tensor_y2, kernelExecuted, elapsedTime, conv1Time, conv2Time, warmup);
  elapsedTime = 0;
  conv1Time = 0;
  conv2Time = 0;
  printf("START-OVERLAP:\n");
  runConvolution<false, ImplicitGemm1, ImplicitGemm2>
    (problem_size, options, &streams[0], overlapHandle, tensor_x, tensor_w1, tensor_w2, tensor_y1, tensor_y2, kernelExecuted, elapsedTime, conv1Time, conv2Time, epochs);
  printf("END-OVERLAP {Total: %lf, Conv1: %lf, Conv2: %lf} micro seconds\n", elapsedTime/epochs, conv1Time/epochs, conv2Time/epochs);

  return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  // Turing Tensor Core operations exposed with mma.sync are first available in CUDA 10.2.
  //
  // CUTLASS must be compiled with CUDA 10.2 Toolkit to run these examples.
  if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2))) {
    std::cerr << "Turing Tensor Core operations must be compiled with CUDA 10.2 Toolkit or later." << std::endl;
    return 0;
  }

  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

  // if (!(props.major > 7 || (props.major == 7 && props.minor >= 5))) {
  //   std::cerr << "Turing Tensor Ops must be run on a machine with compute capability at least 75."
  //             << std::endl;
  //   return 0;
  // }

  Options options;
  
  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }
  // Execute one problem size
  if (!options.valid()) {
    std::cerr << "Invalid problem." << std::endl;
    return -1;
  }

  Result result = profile_convolution(options);

  Result::print_header(std::cout, options) << std::endl;
  result.print(std::cout, 1, options) << std::endl;

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////



