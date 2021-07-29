/***************************************************************************************************
 * Copyright (c) 2017-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Template for a pipelined GEMM kernel. Does not compute batching or support split-K.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/device_kernel.h"
#include "nccl.h"
#define SCCL_MAX_ITER 65536
#define COMPUTE_FLAG_INDEX(__BID__,__GRIDOFFSET_ITER__,__STEP__) \
   SCCL_MAX_ITER*SCCL_MAX_NUM_STEPS*__BID__ + (__GRIDOFFSET_ITER__ * SCCL_MAX_NUM_STEPS + __STEP__)

struct ChunkInfo {
  int numTiles;
  int status;
  int flagsIndex;
};

// static_assert(sizeof(struct ChunkInfo) == 0x20, "Size of Chunk Info must be a power of 2.");

#define ALIGN_UP(x,y) (((x) + (y) - 1)/(y))*(y)

#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/gemm/kernel/scclgemm.h"

#include "cutlass/gemm/kernel/default_scclgemm.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"

#include <vector>
#include <tuple>
#include <set>
#include <map>
#include <utility>
#include <functional>

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

/*! Gemm device-level operator. This is an interface to efficient CUTLASS GEMM kernels that may
  be invoked from host code.

  The contributions of this class are:
    
    1. At compile time, it maps data types and high-level structural parameters onto 
       specific CUTLASS components.

    2. At runtime, it maps logical arguments to GEMM problems to kernel parameters.

    3. At runtime, it launches kernels on the device.

  The intent is to provide a convenient mechanism for interacting with most plausible GEMM
  configurations for each supported architecture. Consequently, not all parameters are exposed
  to the top-level interface. Rather, sensible defaults at each level of the CUTLASS hierarchy
  are selected to tradeoff simplicity of the interface with flexibility. We expect 
  most configurations to be specified at this level. Applications with more exotic requirements 
  may construct their kernels of interest using CUTLASS components at the threadblock, warp, 
  and thread levels of abstraction.

  CUTLASS exposes computations using the functor design pattern in which objects compose some
  internal state with an overloaded function call operator. This enables decoupling of
  initialization from execution, possibly reducing overhead during steady state phases of
  application execution.

  CUTLASS device-level operators expose an Arguments structure encompassing each logical
  input to the computation. This is distinct from the kernel-level Params structure pattern
  which contains application-specific precomputed state needed by the device code.

  Example of a CUTLASS GEMM operator implementing the functionality of cuBLAS's SGEMM NN
  is as follows:

    //
    // Instantiate the CUTLASS GEMM operator.
    //

    cutlass::gemm::device::Gemm<
      float,
      cutlass::layout::ColumnMajor,
      float,
      cutlass::layout::ColumnMajor,
      float,
      cutlass::layout::ColumnMajor
    > gemm_op;

    //
    // Launch the GEMM operation on the device
    //

    cutlass::Status status = gemm_op({
      {m, n, k},                          // GemmCoord problem_size,
      {A, lda},                           // TensorRef<float, layout::ColumnMajor> ref_A,
      {B, ldb},                           // TensorRef<float, layout::ColumnMajor> ref_B,
      {C, ldc},                           // TensorRef<float, layout::ColumnMajor> ref_C,
      {D, ldd},                           // TensorRef<float, layout::ColumnMajor> ref_D,
      {alpha, beta}                       // EpilogueOutputOp::Params epilogue_op_params
    });


  A simplified view of the template is listed below.

    template <
      /// Element type for A matrix operand
      typename ElementA,
      
      /// Layout type for A matrix operand
      typename LayoutA,
      
      /// Element type for B matrix operand
      typename ElementB,
      
      /// Layout type for B matrix operand
      typename LayoutB,
      
      /// Element type for C and D matrix operands
      typename ElementC,
      
      /// Layout type for C and D matrix operands
      typename LayoutC,
      
      /// Element type for internal accumulation
      typename ElementAccumulator,

      /// Operator class tag
      typename OperatorClass,
      
      /// Tag indicating architecture to tune for.  This is the minimum SM that
      /// supports the intended feature. The device kernel can be built
      /// targeting any SM larger than this number.
      typename ArchTag,
      
      /// Threadblock-level tile size (concept: GemmShape)
      typename ThreadblockShape,
      
      /// Warp-level tile size (concept: GemmShape)
      typename WarpShape,
      
      /// Warp-level tile size (concept: GemmShape)
      typename InstructionShape,
      
      /// Epilogue output operator
      typename EpilogueOutputOp,
      
      /// Threadblock-level swizzling operator
      typename ThreadblockSwizzle,
      
      /// Number of stages used in the pipelined mainloop
      int Stages
    >
    class Gemm;
*/
template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Element type for internal accumulation
    typename ElementAccumulator_ = ElementC_,
    /// Operator class tag
    typename OperatorClass_ = arch::OpClassSimt,
    /// Tag indicating architecture to tune for
    typename ArchTag_ = arch::Sm70,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle_ =
        typename threadblock::GemmIdentityThreadblockSwizzle<>,
    /// Number of stages used in the pipelined mainloop
    int Stages =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kStages,
    /// Access granularity of A matrix in units of elements
    int AlignmentA =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kAlignmentA,
    /// Access granularity of B matrix in units of elements
    int AlignmentB =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kAlignmentB,
    /// If true, kernel supports split-K with serial reduction
    bool SplitKSerial = false,
    /// Operation performed by GEMM
    typename Operator_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::Operator>
class SCCLGemm {
 public:

  using ElementA = ElementA_;
  using LayoutA = LayoutA_;
  using TensorRefA = TensorRef<ElementA const, LayoutA>;
  using ElementB = ElementB_;
  using LayoutB = LayoutB_;
  using TensorRefB = TensorRef<ElementB const, LayoutB>;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using TensorRefC = TensorRef<ElementC const, LayoutC>;
  using TensorRefD = TensorRef<ElementC, LayoutC>;
  using ElementAccumulator = ElementAccumulator_;
  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using EpilogueOutputOp = EpilogueOutputOp_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using Operator = Operator_;
  static int const kStages = Stages;
  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentB = AlignmentB;
  static int const kAlignmentC = EpilogueOutputOp::kCount;
  static bool const kSplitKSerial = SplitKSerial;
  static ComplexTransform const kTransformA = ComplexTransform::kNone;
  static ComplexTransform const kTransformB = ComplexTransform::kNone;

  /// Define the kernel
  using GemmKernel = typename kernel::DefaultSCCLGemm<
    ElementA,
    LayoutA,
    kAlignmentA,
    ElementB,
    LayoutB,
    kAlignmentB,
    ElementC,
    LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    kStages,
    kSplitKSerial,
    Operator
  >::GemmKernel;

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    GemmCoord problem_size;
    TensorRef<ElementA const, LayoutA> ref_A;
    TensorRef<ElementB const, LayoutB> ref_B;
    TensorRef<ElementC const, LayoutC> ref_C;
    TensorRef<ElementC, LayoutC> ref_D;
    scclFlag* scclFlags;
    int flagsPerBlock;
    std::vector<std::vector<NCCLChunk>> ncclChunks;
    typename EpilogueOutputOp::Params epilogue;
    int split_k_slices;

    //
    // Methods
    //

    /// Default ctor
    
    Arguments(): problem_size(0, 0, 0), split_k_slices(1) {

    }

    /// Constructs an Arguments structure 
    
    Arguments(
      GemmCoord problem_size_,
      TensorRef<ElementA const, LayoutA> ref_A_,
      TensorRef<ElementB const, LayoutB> ref_B_,
      TensorRef<ElementC const, LayoutC> ref_C_,
      TensorRef<ElementC, LayoutC> ref_D_,
      scclFlag* scclFlags,
      int flagsPerBlock,
      std::vector<std::vector<NCCLChunk>>& ncclChunks,
      typename EpilogueOutputOp::Params epilogue_ = 
        typename EpilogueOutputOp::Params(),
      int split_k_slices = 1
    ):
      problem_size(problem_size_),
      ref_A(ref_A_),
      ref_B(ref_B_),
      ref_C(ref_C_),
      ref_D(ref_D_),
      scclFlags(scclFlags),
      flagsPerBlock(flagsPerBlock),
      ncclChunks(ncclChunks),
      epilogue(epilogue_),
      split_k_slices(split_k_slices) {

    }
  };

private:

  /// Kernel parameters object
  typename GemmKernel::Params params_;
  typename GemmKernel::Workspace workspace_;
  int *tileOrder;
  int workIndex;

  //Workspace Buffer Offsets
  size_t semaphoreOff;
  size_t tileIdxOff;
  size_t tileOrderOff;
  size_t chunksForTileOff;
  size_t chunkInfosOff;

public:

  /// Constructs the GEMM.
  SCCLGemm() { }

  /// Determines whether the GEMM can execute the given problem.
  static Status can_implement(Arguments const &args) {

    if (!kSplitKSerial && args.split_k_slices > 1) {
      return Status::kErrorInvalidProblem;
    }

    Status status = GemmKernel::can_implement(
      args.problem_size,
      args.ref_A.non_const_ref(),
      args.ref_B.non_const_ref(),
      args.ref_C.non_const_ref(),
      args.ref_D
    );

    if (status != Status::kSuccess) {
      return status;
    }

    return Status::kSuccess;
  }

  size_t get_max_chunks_for_tile(Arguments const& args) {
    int maxChunksForTile = 0;
    int maxChunkRows = 0;
    int maxChunkCols = 0;
    for (auto& blockChunks : args.ncclChunks) {
      for (auto& chunk : blockChunks) {
        maxChunkRows = max(maxChunkRows, chunk.chunk.rows);
        maxChunkCols = max(maxChunkCols, chunk.chunk.cols);
      }
    }

    if (maxChunkRows < ThreadblockShape::kM) {
      maxChunksForTile = ThreadblockShape::kM/maxChunkRows;
    } else if (maxChunkRows % ThreadblockShape::kM == 0) {
      maxChunksForTile = 1;
    } else {
      maxChunksForTile = 2;
    }

    if (maxChunkCols < ThreadblockShape::kN) {
      maxChunksForTile *= ThreadblockShape::kN/maxChunkCols;
    } else if (maxChunkCols % ThreadblockShape::kN == 0) {
      maxChunksForTile *= 1;
    } else {
      maxChunksForTile *= 2;
    }
    
    return maxChunksForTile;
  }
  /// Gets the workspace size
  size_t get_workspace_size(Arguments const &args) {
    
    size_t bytes = 0;

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord tiled_shape = threadblock_swizzle.get_tiled_shape(
      args.problem_size, 
      {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
      args.split_k_slices);
    
    semaphoreOff = 0;

    if (kSplitKSerial && args.split_k_slices > 1) {

      bytes += sizeof(int) * size_t(tiled_shape.m()) * size_t(tiled_shape.n());
    }

    size_t numTotalChunks = 0;
    for (int bid = 0; bid < args.ncclChunks.size(); bid++) {
      numTotalChunks += args.ncclChunks[bid].size();
    }

    bytes = ALIGN_UP(bytes, 128);
    
    tileIdxOff = bytes;
    bytes += sizeof(int);
    bytes = ALIGN_UP(bytes, 128);
    
    tileOrderOff = bytes;
    bytes += tiled_shape.m() * tiled_shape.n() * 2 * sizeof(int);
    bytes = ALIGN_UP(bytes, 128);

    chunkInfosOff = bytes;
    bytes += numTotalChunks * sizeof(ChunkInfo);
    bytes = ALIGN_UP(bytes, 128);

    chunksForTileOff = bytes;
    bytes += tiled_shape.m() * tiled_shape.n() * get_max_chunks_for_tile(args) * sizeof(int);
    bytes = ALIGN_UP(bytes, 128);

    return bytes;
  }

  #define MIN(x,y) (((x) < (y)) ? (x) : (y))

#define COMPUTE_FLAG(__WORKINDEX__,__GRIDOFFSET_ITER__,__STEP__) \
   SCCL_MAX_ITER*SCCL_MAX_NUM_STEPS*(uint64_t)__WORKINDEX__ + ((uint64_t)__GRIDOFFSET_ITER__ * SCCL_MAX_NUM_STEPS + (uint64_t)__STEP__)
  /// Initializes GEMM state from arguments.
  Status initialize(Arguments const &args, int _workIndex = 0, void *workspacePtr = nullptr, cudaStream_t stream = nullptr) {

    if (workspacePtr == nullptr) 
      return Status::kErrorInternal;

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord grid_shape = threadblock_swizzle.get_tiled_shape(
      args.problem_size, 
      {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
      args.split_k_slices);
    
    workIndex = _workIndex;
    
    //Do Chunk to Tile mapping
    const int rows = args.problem_size.m();
    const int cols = args.problem_size.n();
    
    int numTotalChunks = 0;
    int maxChunks = 0;
    for (int bid = 0; bid < args.ncclChunks.size(); bid++) {
      numTotalChunks += args.ncclChunks[bid].size();
      maxChunks = max((int)args.ncclChunks[bid].size(), maxChunks);
    }

    std::vector<ChunkInfo> chunkInfos = std::vector<ChunkInfo>(numTotalChunks);
    std::set<std::pair<int, int>> chunkTBs;
    std::vector<int> tileOrderAsPair;
    std::map<int, std::set<int>> tileToChunks;
    int gpuid;
    cudaGetDevice(&gpuid);
    //Schedule tiles of first chunk of all blocks.
    //Then second chunk of all blocks, and so on.
    for (auto chunk = 0; chunk < maxChunks; chunk++) {
      for (int bid = 0; bid < args.ncclChunks.size(); bid++) {
        if (chunk >= args.ncclChunks[bid].size())
          continue;
        auto ncclChunk = args.ncclChunks[bid][chunk];
        auto block = ncclChunk.chunk;
        int cy = block.startRow;
        int cx = block.startCol;
        int m = block.rows;
        int n = block.cols;

        int chunkIndex = cy/m * (cols/n) + cx/n;
        //For a chunk get all tiles required to obtain this chunk
        int startTy = (cy/ ThreadblockShape::kM) * ThreadblockShape::kM;
        int tiles = 0;
        int combinedChunks = 1;
        for (int ty = startTy; ty < min(cy + m, rows); ty += ThreadblockShape::kM) {
          for (int tx = cx; tx < min(cx + n, cols); tx += ThreadblockShape::kN) {
            int tileIndex = ty/ThreadblockShape::kM * (cols/ThreadblockShape::kN) + tx/ThreadblockShape::kN;
            if (tileToChunks[tileIndex].count(chunkIndex/combinedChunks) == 0) {
              tileToChunks[tileIndex].insert(chunkIndex/combinedChunks);
            }

            if (chunkTBs.count(std::make_pair(ty,tx)) == 0) {
              //Each tile should be processed only once
              chunkTBs.insert(std::make_pair(ty,tx));
              tileOrderAsPair.push_back(ty/ThreadblockShape::kM);
              tileOrderAsPair.push_back(tx/ThreadblockShape::kN);
            }

            tiles++;
          }
        } 
        // if (gpuid == 0) printf("gpuid %d chunkIndex %d (%d, %d):(%dx%d) tiles %d index %d\n", gpuid, chunkIndex, cy, cx, m, n, tiles, ncclChunk.bid *  args.flagsPerBlock + ncclChunk.iter);
        chunkInfos[chunkIndex/combinedChunks] = { tiles, 0, ncclChunk.bid * args.flagsPerBlock + ncclChunk.iter};
      }
    }

    int numTiles = grid_shape.m()*grid_shape.n();

    if (tileOrderAsPair.size() != 2*numTiles) {
      std::cerr << "number of tiles scheduled " << tileOrderAsPair.size() << " != " << " 2 * total tiles " << (2 * numTiles) << std::endl;
      return Status::kErrorInternal;
    }

    int maxChunksForTile = 0;

    for (auto v : tileToChunks) {
      maxChunksForTile = std::max(maxChunksForTile, (int)v.second.size());
    }

    std::vector<int> hChunksForTile = std::vector<int>(maxChunksForTile * numTiles, 0);

    for (auto it : tileToChunks) {
      int i = 0;
      for (int c : it.second) {
        hChunksForTile[it.first * maxChunksForTile + i] = c;
        i++;
      }
      for (; i < maxChunksForTile; i++) {
        hChunksForTile[it.first * maxChunksForTile + i] = -1;
      }
    }

    //Transfer Host data to Device in the workspace
    int* dTileIdx = (int*)(((char*)workspacePtr) + tileIdxOff);
    int* dThreadBlockToTileMap = (int*)(((char*)workspacePtr) + tileOrderOff);
    ChunkInfo* dChunkInfo = (ChunkInfo*)(((char*)workspacePtr) + chunkInfosOff);
    int* dChunksForTile = (int*)(((char*)workspacePtr) + chunksForTileOff);

    CUDACHECK(cudaMemset(dTileIdx, 0, sizeof(int)));
    CUDACHECK(cudaMemcpy(dThreadBlockToTileMap, &tileOrderAsPair[0], 
                         numTiles * 2 * sizeof(int), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(dChunkInfo, &chunkInfos[0], numTotalChunks * sizeof(ChunkInfo), 
                         cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(dChunksForTile, &hChunksForTile[0], hChunksForTile.size() * sizeof(int), 
                         cudaMemcpyHostToDevice));

    workspace_ = {dTileIdx, dThreadBlockToTileMap, maxChunksForTile, dChunksForTile, dChunkInfo, gpuid};

    if (kSplitKSerial) {
      if (args.split_k_slices > 1) {
        if (!workspacePtr) {
          return Status::kErrorWorkspaceNull;
        }
        std::cerr << "Do not support split_k_slices > 1 " << std::endl;
        return Status::kErrorInvalidProblem;
        size_t bytes = get_workspace_size(args);
      
        cudaError_t result = cudaMemsetAsync(workspacePtr, 0, bytes, stream);

        if (result != cudaSuccess) {
          return Status::kErrorInternal;
        }
      }
    }
    else {
      if (args.split_k_slices > 1) {
        return Status::kErrorInvalidProblem;
      }
    }

    // Initialize the Params structure
    params_ = typename GemmKernel::Params{
      args.problem_size,
      grid_shape,
      args.ref_A.non_const_ref(),
      args.ref_B.non_const_ref(),
      args.ref_C.non_const_ref(),
      args.ref_D,
      args.scclFlags,
      _workIndex,
      workspace_,
      args.epilogue,
    };

    return Status::kSuccess;
  }

  /// Lightweight update given a subset of arguments
  Status update(Arguments const &args, void *workspace = nullptr) {
    
    if (kSplitKSerial && args.split_k_slices > 1) {  
      if (!workspace) {
        return Status::kErrorWorkspaceNull;
      }
    }

    params_.ref_A.reset(args.ref_A.non_const_ref().data());
    params_.ref_B.reset(args.ref_B.non_const_ref().data());
    params_.ref_C.reset(args.ref_C.non_const_ref().data());
    params_.ref_D.reset(args.ref_D.data());
    params_.output_op = args.epilogue;
    params_.semaphore = static_cast<int *>(workspace);

    return Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  Status run(cudaStream_t stream = nullptr) {

    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
    dim3 block(GemmKernel::kThreadCount, 1, 1);

    params_.currWorkIndex = workIndex;
    workIndex += 1;

    cudaError_t result;

    int smem_size = int(sizeof(typename GemmKernel::SharedStorage));
    if (smem_size >= (48 << 10)) {
      result = cudaFuncSetAttribute(Kernel<GemmKernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);

      if (result != cudaSuccess) {
        return Status::kErrorInternal;
      }

      result = cudaFuncSetAttribute(
          Kernel<GemmKernel>,
          cudaFuncAttributePreferredSharedMemoryCarveout, 100);

      if (result != cudaSuccess) {
        return Status::kErrorInternal;
      }
    }

    cutlass::Kernel<GemmKernel><<<grid, block, smem_size, stream>>>(params_);

    result = cudaGetLastError();

    return result == cudaSuccess ? Status::kSuccess : Status::kErrorInternal;
  }

  /// Runs the kernel using initialized state.
  Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }

  /// Runs the kernel using initialized state.
  Status operator()(
    Arguments const &args, 
    int workIndex = 0,
    void *workspace = nullptr, 
    cudaStream_t stream = nullptr) {
    
    Status status = initialize(args, workIndex, workspace, stream);
    
    if (status == Status::kSuccess) {
      status = run(stream);
    }

    return status;
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Parital specialization for column-major output exchanges problem size and operand.
template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Element type for internal accumulation
    typename ElementAccumulator_,
    /// Operator class tag
    typename OperatorClass_,
    /// Tag indicating architecture to tune for
    typename ArchTag_,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape_,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_,
    /// Epilogue output operator
    typename EpilogueOutputOp_,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle_,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Access granularity of A matrix in units of elements
    int AlignmentA,
    /// Access granularity of B matrix in units of elements
    int AlignmentB,
    /// If true, kernel supports split-K as a serial reduction
    bool SplitKSerial,
    /// Operation performed by GEMM
    typename Operator_>
class SCCLGemm<ElementA_, LayoutA_, ElementB_, LayoutB_, ElementC_,
           layout::ColumnMajor,  // partially specialized on LayoutC
           ElementAccumulator_, OperatorClass_, ArchTag_, ThreadblockShape_,
           WarpShape_, InstructionShape_, EpilogueOutputOp_,
           ThreadblockSwizzle_, Stages, AlignmentA, AlignmentB, SplitKSerial,
           Operator_> {
 public:

  using ElementA = ElementA_;
  using LayoutA = LayoutA_;
  using TensorRefA = TensorRef<ElementA const, LayoutA>;
  using ElementB = ElementB_;
  using LayoutB = LayoutB_;
  using TensorRefB = TensorRef<ElementB const, LayoutB>;
  using ElementC = ElementC_;
  using LayoutC = layout::ColumnMajor;
  using TensorRefC = TensorRef<ElementC const, LayoutC>;
  using TensorRefD = TensorRef<ElementC, LayoutC>;
  using ElementAccumulator = ElementAccumulator_;
  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using EpilogueOutputOp = EpilogueOutputOp_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using Operator = Operator_;
  static int const kStages = Stages;
  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentB = AlignmentB;
  static ComplexTransform const kTransformA = ComplexTransform::kNone;
  static ComplexTransform const kTransformB = ComplexTransform::kNone;
  static bool const kSplitKSerial = SplitKSerial;

  using UnderlyingOperator = SCCLGemm< 
    ElementB,
    typename layout::LayoutTranspose<LayoutB>::type,
    ElementA,
    typename layout::LayoutTranspose<LayoutA>::type,
    ElementC,
    layout::RowMajor,    
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    kAlignmentB,
    kAlignmentA,
    SplitKSerial,
    Operator
  >;

  using UnderlyingArguments = typename UnderlyingOperator::Arguments;
  using GemmKernel = typename UnderlyingOperator::GemmKernel;
  static int const kAlignmentC = UnderlyingOperator::kAlignmentC;

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    GemmCoord problem_size;
    TensorRef<ElementA const, LayoutA> ref_A;
    TensorRef<ElementB const, LayoutB> ref_B;
    TensorRef<ElementC const, LayoutC> ref_C;
    TensorRef<ElementC, LayoutC> ref_D;
    scclFlag* scclFlags;
    std::vector<NCCLChunk> ncclChunks;
    int flagsPerBlock;
    typename EpilogueOutputOp::Params epilogue;
    int split_k_slices;

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments() { }

    /// Constructs an Arguments structure 
    CUTLASS_HOST_DEVICE
    Arguments(
      GemmCoord problem_size_,
      TensorRef<ElementA const, LayoutA> ref_A_,
      TensorRef<ElementB const, LayoutB> ref_B_,
      TensorRef<ElementC const, LayoutC> ref_C_,
      TensorRef<ElementC, LayoutC> ref_D_,
      scclFlag* scclFlags,
      int flagsPerBlock,
      std::vector<NCCLChunk>& ncclChunks,
      typename EpilogueOutputOp::Params epilogue_ = 
        typename EpilogueOutputOp::Params(),
      int split_k_slices = 1
    ):
      problem_size(problem_size_),
      ref_A(ref_A_),
      ref_B(ref_B_),
      ref_C(ref_C_),
      ref_D(ref_D_),
      scclFlags(scclFlags),
      flagsPerBlock(flagsPerBlock),
      ncclChunks(ncclChunks),
      epilogue(epilogue_),
      split_k_slices(split_k_slices) { }
  };

private:

  UnderlyingOperator underlying_operator_;

public:

  /// Constructs the GEMM.
  SCCLGemm() { }

  /// Helper to construct a transposed equivalent for the underying GEMM operator
  static UnderlyingArguments to_underlying_arguments(Arguments const &args) {
    return UnderlyingArguments(
      {args.problem_size.n(), args.problem_size.m(), args.problem_size.k()},
      {args.ref_B.data(), args.ref_B.stride(0)},
      {args.ref_A.data(), args.ref_A.stride(0)},
      {args.ref_C.data(), args.ref_C.stride(0)},
      {args.ref_D.data(), args.ref_D.stride(0)},
      args.scclFlags,
      args.flagsPerBlock,
      args.ncclChunks,
      args.epilogue,
      args.split_k_slices
    );
  }

  /// Determines whether the GEMM can execute the given problem.
  static Status can_implement(Arguments const &args) {

    return UnderlyingOperator::can_implement(to_underlying_arguments(args));
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args) {
    
    return UnderlyingOperator::get_workspace_size(to_underlying_arguments(args));
  }

  /// Initializes GEMM state from arguments.
  Status initialize(Arguments const &args, int workIndex, void *workspace = nullptr, cudaStream_t stream = nullptr) {

    return underlying_operator_.initialize(to_underlying_arguments(args), workIndex, workspace, stream);
  }

  /// Lightweight update given a subset of arguments
  Status update(Arguments const &args, void *workspace = nullptr) {

    return underlying_operator_.update(to_underlying_arguments(args), workspace);
  }

  /// Runs the kernel using initialized state.
  Status run(cudaStream_t stream = nullptr) {

    return underlying_operator_.run(stream);
  }

  /// Runs the kernel using initialized state.
  Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }

  /// Runs the kernel using initialized state.
  Status operator()(
    Arguments const &args, 
    int workIndex = 0,
    void *workspace = nullptr, 
    cudaStream_t stream = nullptr) {
    
    Status status = initialize(args, workIndex, workspace, stream);
    
    if (status == Status::kSuccess) {
      status = run(stream);
    }

    return status;
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
