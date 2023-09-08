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
#include "cutlass/tensor_ref.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/epilogue/thread/linear_combination_silu.h"

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
using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256, 128, 32>;  
using ShapeMMAWarp = cutlass::gemm::GemmShape<128, 64, 32>;
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
// using LayoutInputB = cutlass::layout::RowMajor;
// using LayoutOutput = cutlass::layout::RowMajor;

//Use FP-16 Tensor Cores
using MMAOp = cutlass::arch::OpClassTensorOp;

using SmArch = cutlass::arch::Sm70;

//First GeMM in MLP is fused with GELU
using EpilogueOp1 = cutlass::epilogue::thread::LinearCombinationSilu<
    ElementOutput,                                        
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator, 
    ElementComputeEpilogue,                              
    cutlass::epilogue::thread::ScaleType::NoBetaScaling>;

using EpilogueOp2 = cutlass::epilogue::thread::LinearCombinationSwiGLU<
    ElementOutput,                                        
    128 / cutlass::sizeof_bits<ElementOutput>::value,     
    ElementAccumulator,
    ElementComputeEpilogue,
    cutlass::epilogue::thread::ScaleType::SwishScaling>;

//Third GeMM in MLP performs no extra fused computations 
using EpilogueOp3 = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                        
    128 / cutlass::sizeof_bits<ElementOutput>::value,     
    ElementAccumulator,
    ElementComputeEpilogue>;

template<typename EpilogueOp, bool splitK>
class BaseMLPGemm : public cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, 
                                                       ElementInputA, LayoutInputA,
                                                       ElementInputA, LayoutInputA,
                                                        ElementAccumulator, MMAOp,
                                                        SmArch, ShapeMMAThreadBlock,
                                                        ShapeMMAWarp, ShapeMMAOp,
                                                        EpilogueOp, 
                                                        cutlass::gemm::threadblock::GemmHorizontalThreadblockSwizzle, 
                                                        2, 8, 8, splitK> {};
// Baseline GeMMs
using Gemm1 = BaseMLPGemm<EpilogueOp2, false>;
using Gemm2 = BaseMLPGemm<EpilogueOp2, false>;
using Gemm3 = BaseMLPGemm<EpilogueOp3, false>;

//Baseline GeMMs with SplitK enabled
using GemmSplitK1 = BaseMLPGemm<EpilogueOp2, true>;
using GemmSplitK2 = BaseMLPGemm<EpilogueOp2, true>;
using GemmSplitK3 = BaseMLPGemm<EpilogueOp3, true>;

//CuSync GeMMs
using CuSyncImpl = CuSync<ProdCuStage, ConsCuStage>;

using CuSyncImpl1 = CuSync<ProdCuStage, MiddleCuStage>;
using CuSyncImpl2 = CuSync<MiddleCuStage, ConsCuStage>;

template<typename CuStage, typename EpilogueOp, bool splitK>
class CuSyncMLPGemm : public cutlass::gemm::device::CuSyncGemm<CuStage, ElementInputA, LayoutInputA, 
                                                       ElementInputA, LayoutInputA,
                                                       ElementInputA, LayoutInputA,
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
using TensorRef = cutlass::TensorRef<ElementInputA, LayoutInputA>;

enum MLPType {
  GPT3,
  LLaMa    
};

template<typename GemmTy1, typename GemmTy2, typename GemmTy3>
struct MLPParameters {
  TensorRef x; //[B, H]
  TensorRef w1; //[H, 4H/8] in GPT-3 and [H, H/3] in LLaMa
  //xw1 = GeLU(x * w1)
  TensorRef xw1; //[B, 4 H / 8]
  TensorRef w2; //[4H/8, H] in GPT-3 and [H/3, H] in LLaMa
  //xw12 = xw1 * w2
  TensorRef xw12; //[B, H]

  //For LLaMa only
  TensorRef v; //[H, H/3] in LLaMa
  TensorRef xv; //[B, H/3] in LLaMa
  
  TensorRef ref_xw1;
  TensorRef ref_xw12;

  //For LLaMa only
  TensorRef ref_xv;

  bool checkResults;

  cutlass::gemm::GemmCoord gemm_size1;
  cutlass::gemm::GemmCoord gemm_size2;
  ElementComputeEpilogue alpha;
  ElementComputeEpilogue beta;

  std::string model;

  GemmTy1 gemm1;
  GemmTy2 gemm2;
  GemmTy3 gemm3;

  typename GemmTy1::Arguments argsXW1;
  typename GemmTy2::Arguments argsXV;
  typename GemmTy3::Arguments argsXW12;

  MLPParameters() {
    
  }

  MLPParameters(std::string model_, uint batch, const ElementInputA* w1Ptr, const ElementInputA* vPtr, const ElementInputA* w2Ptr) {
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
    auto extent_ = LayoutInputA::TensorCoord();
    auto layout_ = LayoutInputA::packed(extent_);
    x = TensorRef();
    w1 = TensorRef((cutlass::half_t*)w1Ptr, layout_);
    w2 = TensorRef((cutlass::half_t*)w2Ptr, layout_);
    v = TensorRef((cutlass::half_t*)vPtr, layout_);
    checkResults = false;

    //Initialize GeMMs

    //Setup XW1 GeMM;
    int split_k1 = 1; 
    int split_k2 = 1;
    argsXW1 = typename GemmTy1::Arguments {gemm_size1,
                                           x, 
                                           w1,
                                           xw1,
                                           xw1,
                                           {alpha, beta},
                                           split_k1};

    size_t workspace_size = GemmTy1::get_workspace_size(argsXW1);
    cutlass::device_memory::allocation<uint8_t> workspace1(workspace_size);
    cutlass::Status status = gemm1.can_implement(argsXW1);
    CUTLASS_CHECK(status);
    status = gemm1.initialize(argsXW1, workspace1.get());
    CUTLASS_CHECK(status);
    
    //Setup XV GeMM
    argsXV = typename GemmTy2::Arguments {gemm_size1,
                                          x, 
                                          v,
                                          xw1,
                                          xv,
                                          {alpha, ElementComputeEpilogue(1.0f)},
                                          split_k1};
    workspace_size = GemmTy2::get_workspace_size(argsXV);
    cutlass::device_memory::allocation<uint8_t> workspace2(workspace_size);
    status = gemm2.can_implement(argsXV);
    CUTLASS_CHECK(status);
    status = gemm2.initialize(argsXV, workspace2.get());
    CUTLASS_CHECK(status);

    //Setup XW12 GeMM
    argsXW12 = typename GemmTy3::Arguments {gemm_size2, 
                                            xv, 
                                            w2, 
                                            xw12, 
                                            xw12, 
                                            {alpha, beta},         
                                            split_k2};
    
    workspace_size = GemmTy3::get_workspace_size(argsXW12);
    cutlass::device_memory::allocation<uint8_t> workspace3(workspace_size);
    status = gemm3.can_implement(argsXW12);
    CUTLASS_CHECK(status);
    status = gemm3.initialize(argsXW12, workspace3.get());
    CUTLASS_CHECK(status);
  }

  void setInput(ElementInputA* xPtr) {
    this->x = TensorRef(xPtr, LayoutInputA());
    gemm1.updateA(this->x);
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

  void setIntermediate(ElementInputA* silu, ElementInputA* xv) {
    this->xv = TensorRef(xv, LayoutInputA());
    this->xw1 = TensorRef(silu, LayoutInputA());

    gemm1.updateC(this->xw1);
    gemm1.updateD(this->xw1);

    gemm2.updateA(this->x);
    gemm2.updateC(this->xw1);
    gemm2.updateD(this->xv);

    gemm3.updateA(this->xv);
  }

  void setOutput(ElementInputA* out) {
    this->xw12 = TensorRef(out, LayoutInputA());

    gemm3.updateC(this->xw12);
    gemm3.updateD(this->xw12);
  }

  bool isGPT3() {return model == "gpt3";}
  bool isLLaMa() {return model == "llama";}
};

/*LLaMA Baseline MLP*/
template<typename GemmTy1, typename GemmTy2, typename GemmTy3>
cudaError_t runBaselineLLaMA(int split_k1, int split_k2, 
                             MLPParameters<GemmTy1, GemmTy2, GemmTy3>& mlpParams,
                             cudaStream_t stream1,
                             cudaStream_t stream2,
                             double& execTime, double& matmul1Time, 
                             double& matmul2Time, double& matmul3Time,
                             int iters = 100) {  
  execTime = 0; 
                     
  //Run kernels
  for (int r = 0; r < iters; r++) {    
    double start = timeInMicroSeconds();
    auto status = mlpParams.gemm1(stream1);
    CUTLASS_CHECK(status);
    CUDA_CHECK(cudaDeviceSynchronize());
    double middle1 = timeInMicroSeconds();
    double iterMatMul1 = middle1-start;
    matmul1Time += iterMatMul1;

    status = mlpParams.gemm2(stream1);
    CUTLASS_CHECK(status);
    CUDA_CHECK(cudaDeviceSynchronize());
    double middle2 = timeInMicroSeconds();
    double iterMatMul2 = middle2-middle1;
    matmul2Time += iterMatMul2;

    status = mlpParams.gemm3(stream1);
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

// cudaError_t runBaselineLLaMA(int split_k1, int split_k2, 
//                         MLPParameters& mlpParams,
//                         cudaStream_t stream1,
//                         cudaStream_t stream2,
//                         double& execTime,
//                         double& matmul1Time,
//                         double& matmul2Time,
//                         double& matmul3Time,
//                         int iters = 100) {
//   cudaError_t result;
//   execTime = 0;
//   matmul1Time = 0;
//   matmul2Time = 0;
//   matmul3Time = 0;
//   if (split_k1 == 1 && split_k2 == 1) {
//     result = runBaselineLLaMA<Gemm1, Gemm2, Gemm3>(split_k1, split_k2, mlpParams, stream1, stream2, execTime, matmul1Time, matmul2Time, matmul3Time, iters);
//   } else if (split_k1 > 1 && split_k2 == 1) {
//     result = runBaselineLLaMA<GemmSplitK1, GemmSplitK2, Gemm3>(split_k1, split_k2, mlpParams, stream1, stream2, execTime, matmul1Time, matmul2Time, matmul3Time, iters);
//   } else if (split_k1 == 1 && split_k2 > 1) {
//     result = runBaselineLLaMA<Gemm1, GemmSplitK2, Gemm3>(split_k1, split_k2, mlpParams, stream1, stream2, execTime, matmul1Time, matmul2Time, matmul3Time, iters);
//   } else {
//     result = runBaselineLLaMA<GemmSplitK1, GemmSplitK2, GemmSplitK3>(split_k1, split_k2, mlpParams, stream1, stream2, execTime, matmul1Time, matmul2Time, matmul3Time, iters);
//   }

//   return result;
// }

extern "C"
MLPParameters<Gemm1, Gemm2, Gemm3>* initMLPParams(const void* w1, const void* v, const void* w2, const uint batch) {
  const size_t H = 8192;
  printf("w1 %p v %p w2 %p\n", w1, v, w2);

  MLPParameters<Gemm1, Gemm2, Gemm3>* llamaMLPParams = new MLPParameters<Gemm1, Gemm2, Gemm3>(std::string("llama"), batch, 
                                 (const ElementInputA*)w1, 
                                 (const ElementInputA*)v,
                                 (const ElementInputA*)w2);
  return llamaMLPParams;
}

extern "C"
void runLLAMA(MLPParameters<Gemm1, Gemm2, Gemm3>* llamaMLPParams, const void* x, const void* silu, const void* xv, const void* out) {
  double times = 0;
  llamaMLPParams->setInput((ElementInputA*)x);
  llamaMLPParams->setIntermediate((ElementInputA*)silu, (ElementInputA*)xv);
  llamaMLPParams->setOutput((ElementInputA*)out);
  runBaselineLLaMA<Gemm1, Gemm2, Gemm3>(1, 1, *llamaMLPParams, 0, 0, times, times, times, times, 1);
}