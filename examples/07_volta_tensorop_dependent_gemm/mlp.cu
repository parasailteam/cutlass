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

#include<cuSync.h>

#ifdef ROWSYNC 
  using ProdCuStage = CuStage<CuStageType::Producer, RowMajor, RowSync>;
  using ConsCuStage = CuStage<CuStageType::Consumer, RowMajor, RowSync>;
  using Sync = RowSync;
#elif TILEBATCH
  using ProdCuStage = CuStage<CuStageType::Producer, RowMajor, TileSync<8>>;
  using ConsCuStage = CuStage<CuStageType::Consumer, RowMajor, TileSync<8>>;
  using Sync = TileSync<8>;
#elif TILESYNC
  using ProdCuStage = CuStage<CuStageType::Producer, RowMajor, TileSync<1>>;
  using ConsCuStage = CuStage<CuStageType::Consumer, RowMajor, TileSync<1>>;
  using Sync = TileSync<1>;
#elif BATCHEDROW
  using ProdCuStage = CuStage<CuStageType::Producer, RowMajor, BatchedRowSync>;
  using ConsCuStage = CuStage<CuStageType::Consumer, RowMajor, BatchedRowSync>;
  using Sync = BatchedRowSync;
#else
  #error "Unknown Synchronization"
#endif

#include "common.h"

//Tile sizes of all GeMMs
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 128, 32>;  
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>; 
using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;  

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

using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmHorizontalThreadblockSwizzle;

//First GeMM in MLP is fused with GELU
using EpilogueOp1 = cutlass::epilogue::thread::LinearCombinationGELU<
    ElementOutput,                                        
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator, 
    ElementComputeEpilogue,                              
    cutlass::epilogue::thread::ScaleType::NoBetaScaling>;

//Second GeMM in MLP performs no extra fused computations 
using EpilogueOp2 = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                        
    128 / cutlass::sizeof_bits<ElementOutput>::value,     
    ElementAccumulator,
    ElementComputeEpilogue>;

template<typename EpilogueOp, bool splitK>
class BaseMLPGemm : public cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, 
                                                       ElementInputB, LayoutInputB,
                                                       ElementOutput, LayoutOutput,
                                                        ElementAccumulator, MMAOp,
                                                        SmArch, ShapeMMAThreadBlock,
                                                        ShapeMMAWarp, ShapeMMAOp,
                                                        EpilogueOp, SwizzleThreadBlock, 2, 8, 8, splitK> {};
// Baseline GeMMs
using Gemm1 = BaseMLPGemm<EpilogueOp2, false>;
using Gemm2 = BaseMLPGemm<EpilogueOp2, false>;

//Baseline GeMMs with SplitK enabled
using GemmSplitK1 = BaseMLPGemm<EpilogueOp2, true>;
using GemmSplitK2 = BaseMLPGemm<EpilogueOp2, true>;

//CuSync GeMMs
using CuSyncImpl = CuSync<ProdCuStage, ConsCuStage>;


template<typename CuStage, typename EpilogueOp, bool splitK>
class CuSyncMLPGemm : public cutlass::gemm::device::CuSyncGemm<CuStage, ElementInputA, LayoutInputA, 
                                                       ElementInputB, LayoutInputB,
                                                       ElementOutput, LayoutOutput,
                                                        ElementAccumulator, MMAOp,
                                                        SmArch, ShapeMMAThreadBlock,
                                                        ShapeMMAWarp, ShapeMMAOp,
                                                        EpilogueOp, 
                                                        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 
                                                        2, 8, 8, splitK> {};

using OverlapGemm1 = CuSyncMLPGemm<ProdCuStage, EpilogueOp2, false>;
using OverlapGemm2 = CuSyncMLPGemm<ConsCuStage, EpilogueOp2, false>;

using OverlapGemmSplitK1 = CuSyncMLPGemm<ProdCuStage, EpilogueOp2, true>;
using OverlapGemmSplitK2 = CuSyncMLPGemm<ConsCuStage, EpilogueOp2, true>;

using HostTensor = cutlass::HostTensor<ElementInputA, LayoutInputA>;

struct MLPParameters {
  HostTensor a;
  HostTensor b;
  HostTensor c;
  HostTensor d;
  HostTensor e;

  HostTensor ref_c;
  HostTensor ref_e;
  bool checkResults;

  cutlass::gemm::GemmCoord gemm_size1;
  cutlass::gemm::GemmCoord gemm_size2;
  ElementComputeEpilogue alpha;
  ElementComputeEpilogue beta;

  MLPParameters(int problem[4], bool check) {
    alpha = ElementComputeEpilogue(1.0);
    beta = ElementComputeEpilogue(0.0);
    gemm_size1 = cutlass::gemm::GemmCoord(problem[0], problem[1], problem[2]);
    gemm_size2 = cutlass::gemm::GemmCoord(problem[0], problem[3], problem[1]);
    a = HostTensor(gemm_size1.mk());
    b = HostTensor(gemm_size1.kn());
    c = HostTensor(gemm_size1.mn());
    d = HostTensor(gemm_size2.kn());
    e = HostTensor(gemm_size2.mn());
    ref_c = HostTensor(gemm_size1.mn());
    ref_e = HostTensor(gemm_size2.mn());
    checkResults = check;
  }

  void initIns() {
    if (checkResults) {
      memset_random2(a.host_data(), ElementOutput(0.05), ElementOutput(0.2), a.size());
      memset_random2(b.host_data(), ElementOutput(0.01), ElementOutput(0.2), b.size());
      memset_random2(d.host_data(), ElementOutput(0.01), ElementOutput(0.05), d.size());
    } else {
      cutlass::reference::host::TensorFill(a.host_view(), ElementOutput(0.05));
      cutlass::reference::host::TensorFill(b.host_view(), ElementOutput(0.5));
      cutlass::reference::host::TensorFill(d.host_view(), ElementOutput(0.01));
    }
    // Copy data from host to GPU
    a.sync_device();
    b.sync_device();
    d.sync_device();
  }
  
  void initOuts() {
    cutlass::reference::host::TensorFill(c.host_view());
    cutlass::reference::host::TensorFill(e.host_view());
      
    c.sync_device();
    e.sync_device();
  }

  void initRefs() {
    cutlass::reference::host::TensorFill(ref_e.host_view());
    cutlass::reference::host::TensorFill(ref_c.host_view());
      
    ref_e.sync_device();
    ref_c.sync_device();
  }
};

/** Reference MLP for correctness check **/
cudaError_t referenceMLP(MLPParameters& mlpParams) {
  gpumatmul<ElementOutput, ElementAccumulator>(mlpParams.gemm_size1.m(), 
                                               mlpParams.gemm_size1.n(), 
                                               mlpParams.gemm_size1.k(),
                                               mlpParams.a.device_data(), 
                                               mlpParams.b.device_data(), 
                                               mlpParams.ref_c.host_data());
  CUDA_CHECK(cudaMemcpy(mlpParams.ref_c.device_data(), mlpParams.ref_c.host_data(), 
             sizeof(ElementOutput) * mlpParams.ref_c.size(), cudaMemcpyHostToDevice));
  gpumatmul<ElementOutput, ElementAccumulator>(mlpParams.gemm_size2.m(),
                                               mlpParams.gemm_size2.n(),
                                               mlpParams.gemm_size2.k(), 
                                               mlpParams.ref_c.device_data(),
                                               mlpParams.d.device_data(), 
                                               mlpParams.ref_e.host_data());
  return cudaSuccess;
}

cudaError_t checkMLPResults(MLPParameters& mlpParams) {
  ElementOutput* hostC = new ElementOutput[mlpParams.ref_c.size()];
  CUDA_CHECK(cudaMemcpy(hostC, mlpParams.c.device_data(), 
                        mlpParams.c.size() * sizeof(ElementOutput), 
                        cudaMemcpyDeviceToHost));
  printf("Checking first GeMM\n");
  bool eq = equals(mlpParams.ref_c.size(), mlpParams.ref_c.host_data(), hostC, 1e-2);
  if (eq == false) {
    printf("First GeMM not correct\n");
    return cudaErrorUnknown;
  }
  printf("First GeMM passed\n");
  ElementOutput* hostE = new ElementOutput[mlpParams.ref_e.size()];
  CUDA_CHECK(cudaMemcpy(hostE, mlpParams.e.device_data(), 
                        mlpParams.e.size() * sizeof(ElementOutput), 
                        cudaMemcpyDeviceToHost));
  printf("Checking second GeMM\n");
  eq = equals(mlpParams.ref_e.size(), mlpParams.ref_e.host_data(), hostE, 1e-1);
  if (eq == false) {
    printf("Second GeMM not correct \n");
    return cudaErrorUnknown;
  }

  printf("Second GeMM passed\n");

  return cudaSuccess;
}

/*Baseline MLP*/
template<typename GemmTy1, typename GemmTy2>
cudaError_t runBaseline(int split_k1, int split_k2, 
                        MLPParameters& mlpParams,
                        cudaStream_t stream,
                        double& execTime, double& matmul1Time, double& softmaxTime, double& matmul2Time,
                        int iters = 100) {
  //Setup first GeMM
  typename GemmTy1::Arguments args1 {
    mlpParams.gemm_size1,
    mlpParams.a.device_ref(), 
    mlpParams.b.device_ref(),
    mlpParams.c.device_ref(),
    mlpParams.c.device_ref(),
    {mlpParams.alpha, mlpParams.beta},
    split_k1};

  size_t workspace_size = GemmTy1::get_workspace_size(args1);
  cutlass::device_memory::allocation<uint8_t> workspace1(workspace_size);
  GemmTy1 gemm_op1;
  cutlass::Status status = gemm_op1.can_implement(args1);
  CUTLASS_CHECK(status);
  status = gemm_op1.initialize(args1, workspace1.get());
  CUTLASS_CHECK(status);

  //Setup Second GeMM
  typename GemmTy2::Arguments args2{ 
    mlpParams.gemm_size2, 
    mlpParams.c.device_ref(), 
    mlpParams.d.device_ref(), 
    mlpParams.e.device_ref(), 
    mlpParams.e.device_ref(), 
    {mlpParams.alpha, mlpParams.beta},         
    split_k2};
  
  GemmTy2 gemm_op2;
  workspace_size = GemmTy2::get_workspace_size(args2);
  cutlass::device_memory::allocation<uint8_t> workspace2(workspace_size);
  status = gemm_op2.can_implement(args2);
  CUTLASS_CHECK(status);
  status = gemm_op2.initialize(args2, workspace2.get());
  CUTLASS_CHECK(status);
  
  execTime = 0;
  
  //Run kernels
  for (int r = 0; r < iters; r++) {    
    double start = timeInMicroSeconds();
    status = gemm_op1(args1, workspace1.get(), stream);
    CUTLASS_CHECK(status);
    
    if (status != cutlass::Status::kSuccess) {
      return cudaErrorUnknown;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    double middle1 = timeInMicroSeconds();
    double iterMatMul1 = middle1-start;
    matmul1Time += iterMatMul1;
    status = gemm_op2(args2, workspace2.get(), stream);
    CUTLASS_CHECK(status);

    if (status != cutlass::Status::kSuccess) {
      return cudaErrorUnknown;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    double middle3 = timeInMicroSeconds();
    double iterMatmul2 = middle3-middle1;
    matmul2Time += iterMatmul2;
    double end = timeInMicroSeconds();
    if (iters > 10)
      printf("{\"Total\": %lf, \"matmul1Time\": %lf, \"matmul2Time\": %lf}\n",end-start,iterMatMul1, iterMatmul2);
    execTime += end-start;
  }

  return cudaSuccess;
}

template<typename GemmTy1, typename GemmTy2, typename GemmSplitKTy1, typename GemmSplitKTy2>
cudaError_t runBaseline(int split_k1, int split_k2, 
                        MLPParameters& mlpParams,
                        cudaStream_t stream,
                        double& execTime,
                        double& matmul1Time,
                        double& softmaxTime,
                        double& matmul2Time,
                        int iters = 100) {
  cudaError_t result;
  execTime = 0;
  matmul1Time = 0;
  softmaxTime = 0;
  matmul2Time = 0;
  if (split_k1 == 1 && split_k2 == 1) {
    result = runBaseline<GemmTy1, GemmTy2>(split_k1, split_k2, mlpParams, stream, execTime, matmul1Time, softmaxTime, matmul2Time, iters);
  } else if (split_k1 > 1 && split_k2 == 1) {
    result = runBaseline<GemmSplitKTy1, GemmTy2>(split_k1, split_k2, mlpParams, stream, execTime, matmul1Time, softmaxTime, matmul2Time, iters);
  } else if (split_k1 == 1 && split_k2 > 1) {
    result = runBaseline<GemmTy1, GemmSplitKTy2>(split_k1, split_k2, mlpParams, stream, execTime, matmul1Time, softmaxTime, matmul2Time, iters);
  } else {
    result = runBaseline<GemmSplitKTy1, GemmSplitKTy2>(split_k1, split_k2, mlpParams, stream, execTime, matmul1Time, softmaxTime, matmul2Time, iters);
  }

  return result;
}

/*CuSync GeMMs in MLP*/
template<typename GemmTy1, typename GemmTy2>
cudaError_t runCuSync(int split_k1, int split_k2,
                           MLPParameters& mlpParams,
                           CuSyncImpl& handle,
                           cudaStream_t producer_stream, cudaStream_t consumer_stream,
                           double& execTime,
                           int iters = 100) {
  typename GemmTy1::Arguments args1{handle.prod(),
                                     mlpParams.gemm_size1,
                                     mlpParams.a.device_ref(),
                                     mlpParams.b.device_ref(),
                                     mlpParams.c.device_ref(),
                                     mlpParams.c.device_ref(),
                                     {mlpParams.alpha, mlpParams.beta},         
                                     split_k1};
  GemmTy1 gemm_op1;
  size_t workspace_size = GemmTy1::get_workspace_size(args1);
  cutlass::device_memory::allocation<uint8_t> workspace1(workspace_size);
  cutlass::Status status = gemm_op1.can_implement(args1);
  CUTLASS_CHECK(status);
  status = gemm_op1.initialize(args1, workspace1.get());
  CUTLASS_CHECK(status);

  typename GemmTy2::Arguments args2{handle.cons(),
                                    mlpParams.gemm_size2,  
                                    mlpParams.c.device_ref(),
                                    mlpParams.d.device_ref(),
                                    mlpParams.e.device_ref(),
                                    mlpParams.e.device_ref(),
                                    {mlpParams.alpha, mlpParams.beta},
                                    split_k2};

  GemmTy2 gemm_op2;
  workspace_size = GemmTy2::get_workspace_size(args2);
  cutlass::device_memory::allocation<uint8_t> workspace2(workspace_size);
  status = gemm_op2.can_implement(args2);
  CUTLASS_CHECK(status);
  status = gemm_op2.initialize(args2, workspace2.get());
  CUTLASS_CHECK(status);

  execTime = 0;
  
  for (int r = 0; r < iters; r++) {
    gemm_op2.params_.custage.iter += 1;
    gemm_op1.params_.custage.iter += 1;
    
    double start = timeInMicroSeconds();
    status = gemm_op1.run(true, NULL, producer_stream);
    CUTLASS_CHECK(status);

    // CUDA_CHECK(cudaDeviceSynchronize());
    handle.invokeWaitKernel(consumer_stream);
    status = gemm_op2.run(true, NULL, consumer_stream);
    CUTLASS_CHECK(status);
    CUDA_CHECK(cudaDeviceSynchronize());
    double end = timeInMicroSeconds();
    if (iters > 10)
      printf("{\"Total\": %lf}\n",end-start);
    execTime += end-start;
  }

  return cudaSuccess;
}

template<typename GemmTy1, typename GemmTy2, typename GemmSplitKTy1, typename GemmSplitKTy2>
cudaError_t runCuSync(int split_k1, int split_k2, MLPParameters& mlpParams,
                      CuSyncImpl& handle,
                      cudaStream_t producer_stream, cudaStream_t consumer_stream,
                      double& execTime, int iters = 100) {
  cudaError_t result;
  execTime = 0;

  if (split_k1 == 1 && split_k2 == 1) {
    result = runCuSync<GemmTy1, GemmTy2>(split_k1, split_k2, mlpParams, handle, producer_stream, consumer_stream, execTime, iters);
  } else if (split_k1 > 1 && split_k2 == 1) {
    result = runCuSync<GemmSplitKTy1, GemmTy2>(split_k1, split_k2, mlpParams, handle, producer_stream, consumer_stream, execTime, iters);
  } else if (split_k1 == 1 && split_k2 > 1) {
    result = runCuSync<GemmTy1, GemmSplitKTy2>(split_k1, split_k2, mlpParams, handle, producer_stream, consumer_stream, execTime, iters);
  } else {
    result = runCuSync<GemmSplitKTy1, GemmSplitKTy2>(split_k1, split_k2, mlpParams, handle, producer_stream, consumer_stream, execTime, iters);
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

  cudaStream_t producer_stream;
  cudaStream_t consumer_stream;
  CUDA_CHECK(cudaStreamCreate(&producer_stream));
  CUDA_CHECK(cudaStreamCreate(&consumer_stream));
  
  printf("problem[0] %d problem[1] %d problem[2] %d problem[3] %d\n", problem[0], problem[1], problem[2], problem[3]);
  printf("doChecking=%d split_k1_slices=%d split_k2_slices=%d\n", doChecking, split_k1, split_k2);

  MLPParameters mlpParams(problem, doChecking);
  mlpParams.initIns();
  mlpParams.initOuts();
  mlpParams.initRefs();

  dim3 gridDim = {DIVUP(mlpParams.gemm_size1.m(), ShapeMMAThreadBlock::kM), 
                  DIVUP(mlpParams.gemm_size1.n(), ShapeMMAThreadBlock::kN), 
                  split_k1};
  dim3 tileSize = {ShapeMMAThreadBlock::kM, ShapeMMAThreadBlock::kN, 1};
  
#if ROWSYNC
  using Sync = RowSync;
  RowSync sync(gridDim.y);
#elif TILEBATCH
  using Sync = TileSync<8>;
  Sync sync;
#elif TILESYNC
  using Sync = TileSync<1>;
  Sync sync;
#elif BATCHEDROW
  using Sync = BatchedRowSync;
  BatchedRowSync sync(gridDim.y, 1);
#else
  #error "Unknown Policy"
#endif
  ProdCuStage prod(gridDim, tileSize, sync);
  ConsCuStage cons({DIVUP(mlpParams.gemm_size2.m(), ShapeMMAThreadBlock::kM), 
                    DIVUP(mlpParams.gemm_size2.n(), ShapeMMAThreadBlock::kN), 
                    split_k2}, tileSize, sync);
  prod.iter = cons.iter = 0;
  CuSyncImpl cuSyncHandle(prod, cons);
  cudaError_t result;
  int epochs = 20;
  int warmup = 10;

  if (doChecking) {
    result = referenceMLP(mlpParams);
    if (result != cudaSuccess) {
      return 1;
    }
  }

  double baselineTime = 0;
  double matmul1Time = 0;
  double softmaxTime = 0;
  double matmul2Time = 0;

  if (true) {
    result = runBaseline<Gemm1, Gemm2, GemmSplitK1, GemmSplitK2>(split_k1, split_k2, mlpParams, producer_stream, baselineTime, matmul1Time, softmaxTime, matmul2Time, 1);

    CUDA_CHECK(cudaDeviceSynchronize());

    if (doChecking) {
      result = checkMLPResults(mlpParams);
      if (result != cudaSuccess) {
        return 1;
      }
    }

    result = runBaseline<Gemm1, Gemm2, GemmSplitK1, GemmSplitK2>(split_k1, split_k2, mlpParams, producer_stream, baselineTime, matmul1Time, softmaxTime, matmul2Time, warmup);

    CUDA_CHECK(cudaDeviceSynchronize());
    printf("START-BASELINE:\n");
    // double startTime = convertTimeValToDouble(getTimeOfDay());    
    result = runBaseline<Gemm1, Gemm2, GemmSplitK1, GemmSplitK2>(split_k1, split_k2, mlpParams, producer_stream, baselineTime, matmul1Time, softmaxTime, matmul2Time, epochs);
    CUDA_CHECK(result);

    printf("END-BASELINE:\n");
    printf("cutlass-baseline elapsedtime %lf microseconds\n", baselineTime/(float)epochs);
  }

  if (doChecking) {
    mlpParams.initOuts();
  }
  
  int highestPriority;
  int lowestPriority;
  CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lowestPriority, &highestPriority));
  CUDA_CHECK(cudaStreamCreateWithPriority(&consumer_stream, 0, lowestPriority));

  printf("gridDim.x %d gridDim.y %d\n", gridDim.x, gridDim.y);
  dim3 grid2Dim = {DIVUP(mlpParams.gemm_size2.m(), ShapeMMAThreadBlock::kM), 
                   DIVUP(mlpParams.gemm_size2.n(), ShapeMMAThreadBlock::kN),
                   split_k2};

  double overlapTime = 0;
  cuSyncHandle.iter = 0;
  cuSyncHandle.prod().iter = cuSyncHandle.cons().iter = 0;
  if (true) {
    result = runCuSync<OverlapGemm1, OverlapGemm2, OverlapGemmSplitK1, OverlapGemmSplitK2>(split_k1, split_k2, mlpParams, cuSyncHandle, producer_stream, consumer_stream, overlapTime, 1);

    CUDA_CHECK(cudaDeviceSynchronize());
    if (doChecking) {
      result = checkMLPResults(mlpParams);
      if (result != cudaSuccess) {
        return 1;
      }
    }
    //warmup
    result = runCuSync<OverlapGemm1, OverlapGemm2, OverlapGemmSplitK1, OverlapGemmSplitK2>(split_k1, split_k2, mlpParams, cuSyncHandle, producer_stream, consumer_stream, overlapTime, warmup);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("START-OVERLAPPED:\n");
    
    result = runCuSync<OverlapGemm1, OverlapGemm2, OverlapGemmSplitK1, OverlapGemmSplitK2>(split_k1, split_k2, mlpParams, cuSyncHandle, producer_stream, consumer_stream, overlapTime, epochs);
    
    CUDA_CHECK(result);
    printf("END-OVERLAPPED:\n");
    
    CUDA_CHECK(cudaStreamSynchronize(consumer_stream));
    CUDA_CHECK(cudaStreamSynchronize(producer_stream));
    
    printf("overlapped elapsedtime %lf microseconds\n", overlapTime/(float)epochs);
  }

  return 0;
}