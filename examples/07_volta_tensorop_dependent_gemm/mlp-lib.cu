#include <iostream>
#include <sstream>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/cusyncgemm.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "helper.h"
#include <curand_kernel.h>

#include <time.h>
#include <sys/time.h>

#include<cuSync.h>

#ifdef ROWSYNC
  using ProdCuStage = CuStage<CuStageType::Producer, RowMajor, RowSync>;
  using MiddleCuStage = CuStage<CuStageType::Producer | CuStageType::Consumer, RowMajor, RowSync>;
  using ConsCuStage = CuStage<CuStageType::Consumer, RowMajor, RowSync>;
  using Sync = RowSync;
#elif defined(TILEBATCH)
  using ProdCuStage = CuStage<CuStageType::Producer, RowMajor, TileSync<2>>;
  using MiddleCuStage = CuStage<CuStageType::Producer | CuStageType::Consumer, RowMajor, TileSync<2>>;
  using ConsCuStage = CuStage<CuStageType::Consumer, RowMajor, TileSync<2>>;
  using Sync = TileSync<2>;
#elif defined(TILESYNC)
  using ProdCuStage = CuStage<CuStageType::Producer, RowMajor, TileSync<1>>;
  using MiddleCuStage = CuStage<CuStageType::Producer | CuStageType::Consumer, RowMajor, TileSync<1>>;
  using ConsCuStage = CuStage<CuStageType::Consumer, RowMajor, TileSync<1>>;
  using Sync = TileSync<1>;
#elif defined(BATCHEDROW)
  using ProdCuStage = CuStage<CuStageType::Producer, RowMajor, BatchedRowSync>;
  using ConsCuStage = CuStage<CuStageType::Consumer, RowMajor, BatchedRowSync>;
  using Sync = BatchedRowSync;
#else
  #error "Unknown Synchronization"
#endif

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

// #include "common.h"

#ifndef EVAL_TILE_SIZES
//Tile sizes of all GeMMs
using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 256, 32>;  
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 128, 32>;
#else
//<eval tiles>
using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;  
using ShapeMMAWarp = cutlass::gemm::GemmShape<128, 64, 32>;
//</eval tiles>
#endif

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
                                                        EpilogueOp, 
                                                        cutlass::gemm::threadblock::GemmHorizontalThreadblockSwizzle, 
                                                        2, 8, 8, splitK> {};
// Baseline GeMMs
using Gemm1 = BaseMLPGemm<EpilogueOp2, false>;
using Gemm2 = BaseMLPGemm<EpilogueOp2, false>;

//Baseline GeMMs with SplitK enabled
using GemmSplitK1 = BaseMLPGemm<EpilogueOp2, true>;
using GemmSplitK2 = BaseMLPGemm<EpilogueOp2, true>;

//CuSync GeMMs
using CuSyncImpl = CuSync<ProdCuStage, ConsCuStage>;

using CuSyncImpl1 = CuSync<ProdCuStage, MiddleCuStage>;
using CuSyncImpl2 = CuSync<MiddleCuStage, ConsCuStage>;

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

using CuSyncGemm1 = CuSyncMLPGemm<ProdCuStage, EpilogueOp2, false>;
using CuSyncGemmMiddle = CuSyncMLPGemm<MiddleCuStage, EpilogueOp2, false>;
using CuSyncGemm2 = CuSyncMLPGemm<ConsCuStage, EpilogueOp2, false>;

using CuSyncGemmSplitK1 = CuSyncMLPGemm<ProdCuStage, EpilogueOp2, true>;
using CuSyncGemmSplitKMiddle = CuSyncMLPGemm<MiddleCuStage, EpilogueOp2, true>;
using CuSyncGemmSplitK2 = CuSyncMLPGemm<ConsCuStage, EpilogueOp2, true>;

using HostTensor = cutlass::HostTensor<ElementInputA, LayoutInputA>;

enum MLPType {
  GPT3,
  LLaMa    
};

struct MLPParameters {
  HostTensor x; //[B, H]
  HostTensor w1; //[H, 4H/8] in GPT-3 and [H, H/3] in LLaMa
  //xw1 = GeLU(x * w1)
  HostTensor xw1; //[B, 4 H / 8]
  HostTensor w2; //[4H/8, H] in GPT-3 and [H/3, H] in LLaMa
  //xw12 = xw1 * w2
  HostTensor xw12; //[B, H]

  //For LLaMa only
  HostTensor v; //[H, H/3] in LLaMa
  HostTensor xv; //[B, H/3] in LLaMa
  
  HostTensor ref_xw1;
  HostTensor ref_xw12;

  //For LLaMa only
  HostTensor ref_xv;

  bool checkResults;

  cutlass::gemm::GemmCoord gemm_size1;
  cutlass::gemm::GemmCoord gemm_size2;
  ElementComputeEpilogue alpha;
  ElementComputeEpilogue beta;

  std::string model;

  MLPParameters() {
    
  }
  MLPParameters(std::string model_, uint batch, bool check) {
    alpha = ElementComputeEpilogue(1.0);
    beta = ElementComputeEpilogue(0.0);
    model = model_;

    if (model == "gpt3") {
      gemm_size1 = cutlass::gemm::GemmCoord(batch, 4*12288/8, 12288);
      gemm_size2 = cutlass::gemm::GemmCoord(batch, 12288, 4*12288/8);
    } else if (model=="llama") {
      int d = ((8192/3 + 127)/128)*128;
      gemm_size1 = cutlass::gemm::GemmCoord(batch, d, 8192);
      gemm_size2 = cutlass::gemm::GemmCoord(batch, 8192, d);
    }
    std::cout << "GeMM 1 Size: " << gemm_size1.m() << ", " << 
      gemm_size1.n() << ", " << gemm_size1.k() << std::endl;
    std::cout << "GeMM 2 Size: " << gemm_size2.m() << ", " << 
      gemm_size2.n() << ", " << gemm_size2.k() << std::endl;
    
    x = HostTensor(gemm_size1.mk());
    w1 = HostTensor(gemm_size1.kn());
    xw1 = HostTensor(gemm_size1.mn());
    w2 = HostTensor(gemm_size2.kn());
    xw12 = HostTensor(gemm_size2.mn());
    ref_xw1 = HostTensor(gemm_size1.mn());
    ref_xw12 = HostTensor(gemm_size2.mn());

    if (model == "llama") {
      v = HostTensor(gemm_size1.kn());
      xv = HostTensor(gemm_size1.mn());
      ref_xv = HostTensor(gemm_size1.mn());
    }
    checkResults = check;
  }

  void initIns() {
    // if (checkResults) {
    //   memset_random2(x.host_data(), ElementOutput(0.05), ElementOutput(0.2), x.size());
    //   memset_random2(w1.host_data(), ElementOutput(0.01), ElementOutput(0.2), w1.size());
    //   memset_random2(w2.host_data(), ElementOutput(0.01), ElementOutput(0.05), w2.size());
    //   if (model == "llama") {
    //     memset_random2(v.host_data(), ElementOutput(0.01), ElementOutput(0.2), v.size());
    //   }
    // } else {
    //   cutlass::reference::host::TensorFill(x.host_view(), ElementOutput(0.05));
    //   cutlass::reference::host::TensorFill(w1.host_view(), ElementOutput(0.5));
    //   cutlass::reference::host::TensorFill(w2.host_view(), ElementOutput(0.01));
    //   if (model == "llama") {
    //     cutlass::reference::host::TensorFill(v.host_view(), ElementOutput(0.5));
    //   }
    // }
    // Copy data from host to GPU
    // x.sync_device();
    // w1.sync_device();
    // w2.sync_device();
    // if (model == "llama") {
    //   v.sync_device();
    // }
  }
  
  void initOuts() {
    cutlass::reference::host::TensorFill(xw1.host_view());
    cutlass::reference::host::TensorFill(xw12.host_view());
      
    xw1.sync_device();
    xw12.sync_device();
    if (model == "llama") {
      cutlass::reference::host::TensorFill(xv.host_view());
      xv.sync_device();
    }
  }

  void initRefs() {
    cutlass::reference::host::TensorFill(ref_xw12.host_view());
    cutlass::reference::host::TensorFill(ref_xw1.host_view());

    ref_xw12.sync_device();
    ref_xw1.sync_device();
    if (model == "llama") {
      cutlass::reference::host::TensorFill(ref_xv.host_view());
      ref_xv.sync_device(); 
    }
  }

  bool isGPT3() {return model == "gpt3";}
  bool isLLaMa() {return model == "llama";}
};

/*LLaMA Baseline MLP*/
template<typename GemmTy1, typename GemmTy2>
cudaError_t runBaselineLLaMA(int split_k1, int split_k2, 
                             MLPParameters& mlpParams,
                             cudaStream_t stream1,
                             cudaStream_t stream2,
                             double& execTime, double& matmul1Time, 
                             double& matmul2Time, double& matmul3Time,
                             int iters = 100) {
  //Setup XW1 GeMM
  typename GemmTy1::Arguments argsXW1{
    mlpParams.gemm_size1,
    mlpParams.x.device_ref(), 
    mlpParams.w1.device_ref(),
    mlpParams.xw1.device_ref(),
    mlpParams.xw1.device_ref(),
    {mlpParams.alpha, mlpParams.beta},
    split_k1};

  size_t workspace_size = GemmTy1::get_workspace_size(argsXW1);
  cutlass::device_memory::allocation<uint8_t> workspace1(workspace_size);
  GemmTy1 gemm_opXW1;
  cutlass::Status status = gemm_opXW1.can_implement(argsXW1);
  CUTLASS_CHECK(status);
  status = gemm_opXW1.initialize(argsXW1, workspace1.get());
  CUTLASS_CHECK(status);
  
  //Setup XV GeMM
  typename GemmTy1::Arguments argsXV{
    mlpParams.gemm_size1,
    mlpParams.x.device_ref(), 
    mlpParams.v.device_ref(),
    mlpParams.xv.device_ref(),
    mlpParams.xv.device_ref(),
    {mlpParams.alpha, mlpParams.beta},
    split_k1};
  workspace_size = GemmTy1::get_workspace_size(argsXV);
  cutlass::device_memory::allocation<uint8_t> workspace2(workspace_size);
  GemmTy1 gemm_opXV;
  status = gemm_opXV.can_implement(argsXV);
  CUTLASS_CHECK(status);
  status = gemm_opXV.initialize(argsXV, workspace2.get());
  CUTLASS_CHECK(status);

  //Setup XW12 GeMM
  typename GemmTy2::Arguments argsXW12{
    mlpParams.gemm_size2, 
    mlpParams.xw1.device_ref(), 
    mlpParams.w2.device_ref(), 
    mlpParams.xw12.device_ref(), 
    mlpParams.xw12.device_ref(), 
    {mlpParams.alpha, mlpParams.beta},         
    split_k2};
  
  GemmTy2 gemm_opXW12;
  workspace_size = GemmTy2::get_workspace_size(argsXW12);
  cutlass::device_memory::allocation<uint8_t> workspace3(workspace_size);
  status = gemm_opXW12.can_implement(argsXW12);
  CUTLASS_CHECK(status);
  status = gemm_opXW12.initialize(argsXW12, workspace3.get());
  CUTLASS_CHECK(status);
  
  execTime = 0; 

  //Run kernels
  for (int r = 0; r < iters; r++) {    
    double start = timeInMicroSeconds();
    status = gemm_opXW1(stream1);
    CUTLASS_CHECK(status);
    CUDA_CHECK(cudaDeviceSynchronize());
    double middle1 = timeInMicroSeconds();
    double iterMatMul1 = middle1-start;
    matmul1Time += iterMatMul1;

    status = gemm_opXV(stream1);
    CUTLASS_CHECK(status);
    CUDA_CHECK(cudaDeviceSynchronize());
    double middle2 = timeInMicroSeconds();
    double iterMatMul2 = middle2-middle1;
    matmul2Time += iterMatMul2;

    status = gemm_opXW12(stream1);
    CUTLASS_CHECK(status);
    CUDA_CHECK(cudaDeviceSynchronize());
    double middle3 = timeInMicroSeconds();
    double iterMatmul3 = middle3-middle2;
    matmul3Time += iterMatmul3;
    double end = timeInMicroSeconds();
    if (iters > 10)
      printf("{\"Total\": %lf, \"matmul1Time\": %lf, \"matmul2Time\": %lf, \"matmul3Time\": %lf}\n",end-start, iterMatMul1, iterMatMul2, iterMatmul3);
    execTime += end-start;
  }

  return cudaSuccess;
}

cudaError_t runBaselineLLaMA(int split_k1, int split_k2, 
                        MLPParameters& mlpParams,
                        cudaStream_t stream1,
                        cudaStream_t stream2,
                        double& execTime,
                        double& matmul1Time,
                        double& matmul2Time,
                        double& matmul3Time,
                        int iters = 100) {
  cudaError_t result;
  execTime = 0;
  matmul1Time = 0;
  matmul2Time = 0;
  matmul3Time = 0;
  if (split_k1 == 1 && split_k2 == 1) {
    result = runBaselineLLaMA<Gemm1, Gemm2>(split_k1, split_k2, mlpParams, stream1, stream2, execTime, matmul1Time, matmul2Time, matmul3Time, iters);
  } else if (split_k1 > 1 && split_k2 == 1) {
    result = runBaselineLLaMA<GemmSplitK1, Gemm2>(split_k1, split_k2, mlpParams, stream1, stream2, execTime, matmul1Time, matmul2Time, matmul3Time, iters);
  } else if (split_k1 == 1 && split_k2 > 1) {
    result = runBaselineLLaMA<Gemm1, GemmSplitK2>(split_k1, split_k2, mlpParams, stream1, stream2, execTime, matmul1Time, matmul2Time, matmul3Time, iters);
  } else {
    result = runBaselineLLaMA<GemmSplitK1, GemmSplitK2>(split_k1, split_k2, mlpParams, stream1, stream2, execTime, matmul1Time, matmul2Time, matmul3Time, iters);
  }

  return result;
}

MLPParameters llamaMLPParams;

extern "C"
void initMLPParams(const void* ptr, size_t size) {
  printf("ptr %p size %ld\n", ptr, size);
}