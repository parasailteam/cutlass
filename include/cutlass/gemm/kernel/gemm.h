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

/*! \file
    \brief Template for a pipelined GEMM kernel. Does not compute batching or support split-K.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/semaphore.h"
#include "cutlass/arch/arch.h"

#include "cutlass/overlap_handle.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Mma_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Epilogue_,             ///! Epilogue
  typename ThreadblockSwizzle_,   ///! Threadblock swizzling function
  bool SplitKSerial               ///! If true, code supporting split-K via serial reduction is enabled.
>
struct Gemm {

  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using OutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static bool const kSplitKSerial = SplitKSerial;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  /// Parameters structure
  struct Params {
    OverlapHandle overlap_handle;
    cutlass::gemm::GemmCoord problem_size;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    int swizzle_log_tile;
    typename Mma::IteratorA::Params params_A;
    typename Mma::IteratorA::TensorRef ref_A;
    typename Mma::IteratorB::Params params_B;
    typename Mma::IteratorB::TensorRef ref_B;
    typename Epilogue::OutputTileIterator::Params params_C;
    typename Epilogue::OutputTileIterator::TensorRef ref_C;
    typename Epilogue::OutputTileIterator::Params params_D;
    typename Epilogue::OutputTileIterator::TensorRef ref_D;
    typename OutputOp::Params output_op;
    int *semaphore;
    int gemm_k_size;
    // For gather+scatter operations
    int const *gather_A_indices;
    int const *gather_B_indices;
    int const *scatter_D_indices;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params(): swizzle_log_tile(0), semaphore(0), gemm_k_size(0) { }

    CUTLASS_HOST_DEVICE
    Params(
      OverlapHandle overlap_handle,
      cutlass::gemm::GemmCoord const & problem_size,
      cutlass::gemm::GemmCoord const & grid_tiled_shape,
      typename Mma::IteratorA::TensorRef ref_A,
      typename Mma::IteratorB::TensorRef ref_B,
      typename Epilogue::OutputTileIterator::TensorRef ref_C,
      typename Epilogue::OutputTileIterator::TensorRef ref_D,
      typename OutputOp::Params output_op = typename OutputOp::Params(),
      int *workspace = nullptr,
      int const *gather_A_indices = nullptr,
      int const *gather_B_indices = nullptr,
      int const *scatter_D_indices = nullptr
    ):
      overlap_handle(overlap_handle),
      problem_size(problem_size),
      grid_tiled_shape(grid_tiled_shape),
      swizzle_log_tile(ThreadblockSwizzle().get_log_tile(grid_tiled_shape)),
      params_A(ref_A.layout()),
      ref_A(ref_A),
      params_B(ref_B.layout()),
      ref_B(ref_B),
      params_C(ref_C.layout()),
      ref_C(ref_C),
      params_D(ref_D.layout()),
      ref_D(ref_D),
      output_op(output_op),
      gather_A_indices(gather_A_indices),
      gather_B_indices(gather_B_indices),
      scatter_D_indices(scatter_D_indices) {
      // printf("128: swizzle_log_tile %d\n", swizzle_log_tile);
      int total_gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;
      int gemm_k_iterations = (total_gemm_k_iterations + grid_tiled_shape.k() - 1) / grid_tiled_shape.k();
      
      gemm_k_size = gemm_k_iterations * Mma::Shape::kK;

    semaphore = workspace;
    }
  };

  /// Shared memory storage structure
  union SharedStorage {
    struct {
      int linear_id;
      int block_idx_x;
      int block_idx_y;
    } tbInfo;
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  Gemm() { } 

  /// Determines whether kernel satisfies alignment
  CUTLASS_HOST_DEVICE
  static Status can_implement(
    cutlass::gemm::GemmCoord const & problem_size,
    typename Mma::IteratorA::TensorRef ref_A,
    typename Mma::IteratorB::TensorRef ref_B,
    typename Epilogue::OutputTileIterator::TensorRef ref_C,
    typename Epilogue::OutputTileIterator::TensorRef ref_D) {

    static int const kAlignmentA = (platform::is_same<typename Mma::IteratorA::Layout,
                                                      layout::ColumnMajorInterleaved<32>>::value)
                                   ? 32
                                   : (platform::is_same<typename Mma::IteratorA::Layout,
                                                        layout::ColumnMajorInterleaved<64>>::value)
                                     ? 64
                                     : Mma::IteratorA::AccessType::kElements;
    static int const kAlignmentB =  (platform::is_same<typename Mma::IteratorB::Layout,
                                                       layout::RowMajorInterleaved<32>>::value)
                                   ? 32
                                   : (platform::is_same<typename Mma::IteratorB::Layout,
                                                        layout::RowMajorInterleaved<64>>::value)
                                     ? 64
                                     : Mma::IteratorB::AccessType::kElements;
    static int const kAlignmentC = (platform::is_same<typename Epilogue::OutputTileIterator::Layout,
                                                      layout::ColumnMajorInterleaved<32>>::value)
                                   ? 32
                                   : (platform::is_same<typename Epilogue::OutputTileIterator::Layout,
                                                        layout::ColumnMajorInterleaved<64>>::value)
                                     ? 64
                                     : Epilogue::OutputTileIterator::kElementsPerAccess;

    if (!TensorRef_aligned(ref_A, kAlignmentA)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_B, kAlignmentB)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_C, kAlignmentC)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_D, kAlignmentC)) {
      return Status::kErrorMisalignedOperand;
    }

    return Status::kSuccess;
  }

  CUTLASS_DEVICE
  void operator()(Params &params, SharedStorage &shared_storage) {
    // Compute threadblock location
    ThreadblockSwizzle threadblock_swizzle;
    cutlass::gemm::GemmCoord threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);
    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //   printf("params.grid_tiled_shape.m() %d params.grid_tiled_shape.n() %d A %p B %p C %p D %p\n",
    //   params.grid_tiled_shape.m(), params.grid_tiled_shape.n(), params.ref_A.data(), params.ref_B.data(), params.ref_C.data(), params.ref_D.data());
    // }
    // Early exit if CTA is out of range
    if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
      params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {
      return;
    }

    // if (threadIdx.x == 0) printf("Mma::Shape::kM %d Mma::Shape::kN %d\n", Mma::Shape::kM, Mma::Shape::kN);
    // Compute initial location in logical coordinates
    cutlass::MatrixCoord tb_offset_A{
      threadblock_tile_offset.m() * Mma::Shape::kM,
      threadblock_tile_offset.k() * params.gemm_k_size,
    };

    cutlass::MatrixCoord tb_offset_B{
      threadblock_tile_offset.k() * params.gemm_k_size,
      threadblock_tile_offset.n() * Mma::Shape::kN
    };

    // Problem size is a function of threadblock index in the K dimension
    int problem_size_k = min(
      params.problem_size.k(), 
      (threadblock_tile_offset.k() + 1) * params.gemm_k_size);

    // Compute threadblock-scoped matrix multiply-add
    int gemm_k_iterations = (problem_size_k - tb_offset_A.column() + Mma::Shape::kK - 1) / Mma::Shape::kK;

    // Compute position within threadblock
    int thread_idx = threadIdx.x;

    // Construct iterators to A and B operands
    typename Mma::IteratorA iterator_A(
      params.params_A,
      params.ref_A.data(),
      {params.problem_size.m(), problem_size_k},
      thread_idx,
      tb_offset_A,
      params.gather_A_indices);

    typename Mma::IteratorB iterator_B(
      params.params_B,
      params.ref_B.data(),
      {problem_size_k, params.problem_size.n()},
      thread_idx,
      tb_offset_B,
      params.gather_B_indices);

    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    int lane_idx = threadIdx.x % 32;

    //
    // Main loop
    //

    // Construct thread-scoped matrix multiply
    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    typename Mma::FragmentC accumulators;

    accumulators.clear();

    if (!kSplitKSerial || gemm_k_iterations > 0) {
      // Compute threadblock-scoped matrix multiply-add
      mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);
    }

    //
    // Epilogue
    //

    OutputOp output_op(params.output_op);

    //
    // Masked tile iterators constructed from members
    //

    threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    //assume identity swizzle
    MatrixCoord threadblock_offset(
      threadblock_tile_offset.m() * Mma::Shape::kM,
      threadblock_tile_offset.n() * Mma::Shape::kN
    );

    int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

    // Construct the semaphore.
    Semaphore semaphore(params.semaphore + block_idx, thread_idx);

    // If performing a reduction via split-K, fetch the initial synchronization
    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
      
      // Fetch the synchronization lock initially but do not block.
      semaphore.fetch();

      // Indicate which position in a serial reduction the output operator is currently updating
      output_op.set_k_partition(threadblock_tile_offset.k(), params.grid_tiled_shape.k());
    }

    // Tile iterator loading from source tensor.
    typename Epilogue::OutputTileIterator iterator_C(
      params.params_C,
      params.ref_C.data(),
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset,
      params.scatter_D_indices
    );

    // Tile iterator writing to destination tensor.
    typename Epilogue::OutputTileIterator iterator_D(
      params.params_D,
      params.ref_D.data(),
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset,
      params.scatter_D_indices
    );

    Epilogue epilogue(
      shared_storage.epilogue, 
      thread_idx, 
      warp_idx, 
      lane_idx);

    // Wait on the semaphore - this latency may have been covered by iterator construction
    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
        
      // For subsequent threadblocks, the source matrix is held in the 'D' tensor.
      if (threadblock_tile_offset.k()) {
        iterator_C = iterator_D;
      }

      semaphore.wait(threadblock_tile_offset.k());

    }

    // Execute the epilogue operator to update the destination tensor.
    epilogue(output_op, iterator_D, accumulators, iterator_C); 
    
    //
    // Release the semaphore
    //

    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
      
      int lock = 0;
      if (params.grid_tiled_shape.k() == threadblock_tile_offset.k() + 1) {

        // The final threadblock resets the semaphore for subsequent grids.
        lock = 0;
      }
      else {
        // Otherwise, the semaphore is incremented
        lock = threadblock_tile_offset.k() + 1;
      }

      semaphore.release(lock);
    }
  }

  /// Executes one GEMM
  CUTLASS_DEVICE
  void run_overlap_gemm(Params &params, SharedStorage &shared_storage, bool isProducerOrConsumer, int firstBlockIdxX, int lastBlockIdxX, volatile int* kernelAllocated) {
    //Convert 1-D thread block id to 2-D
    
    //In column major, y-dim is M,
    //Using compile time constants to avoid expensive divide and mod
    const uint grid_dim_x = lastBlockIdxX - firstBlockIdxX;//(gridDim.x >= params.grid_tiled_shape.m()) ? params.grid_tiled_shape.m() : gridDim.x;
    const uint grid_dim_y = 1;//(grid_dim_x >= gridDim.x) ? 1 : gridDim.x / grid_dim_x;
    uint start_block_idx_y = 0;
    uint start_block_idx_x = 0;
    
    int totalTBs = params.grid_tiled_shape.m()*params.grid_tiled_shape.n();
    if (threadIdx.x == 0) {
      if (isProducerOrConsumer) {
        shared_storage.tbInfo.linear_id = atomicAdd(params.overlap_handle.numProducerTBs, 1) - (params.overlap_handle.iter-1)*totalTBs;
        shared_storage.tbInfo.block_idx_x = params.overlap_handle.blockIndexOrder[shared_storage.tbInfo.linear_id*2];//blockIdx.y;//params.grid_tiled_shape.m();
        shared_storage.tbInfo.block_idx_y = params.overlap_handle.blockIndexOrder[shared_storage.tbInfo.linear_id*2 + 1];//firstBlockIdxX + blockIdx.x;//blockIdx.x % params.grid_tiled_shape.m();
        // printf("400: %d %d %d\n", shared_storage.tbInfo.linear_id, shared_storage.tbInfo.block_idx_x, shared_storage.tbInfo.block_idx_y);
        if (shared_storage.tbInfo.linear_id == 0) {
          *kernelAllocated = params.overlap_handle.iter;
        } 
      } else {
        // atomicAdd(params.overlap_handle.numConsumerTBs, 1);// - (params.overlap_handle.iter-1)*totalTBs;
      } 
    }

    if (isProducerOrConsumer) {
      __syncthreads();
      // const uint mTbs = gridDim.x;//35584/128;
      start_block_idx_x = shared_storage.tbInfo.block_idx_x; // shared_storage.block_idx_y = params.overlap_handle.blockIndexOrder[shared_storage.linear_id*2];//blockIdx.y;//params.grid_tiled_shape.m();
      start_block_idx_y = shared_storage.tbInfo.block_idx_y;// shared_storage.block_idx_x = params.overlap_handle.blockIndexOrder[shared_storage.linear_id*2 + 1];//firstBlockIdxX + blockIdx.x;//blockIdx.x % params.grid_tiled_shape.m();
    } else {
      start_block_idx_y = blockIdx.y;
      start_block_idx_x = blockIdx.x;
    }

    uint block_idx_y = start_block_idx_y;
    uint block_idx_x = start_block_idx_x;
    const int remaining_block_idx = (gridDim.x/(3*80))*(3*80);
    
    // for (uint block_idx_y = start_block_idx_y; block_idx_y < params.grid_tiled_shape.n(); block_idx_y += grid_dim_y) 
    {
    // for (uint block_idx_x = start_block_idx_x; block_idx_x < lastBlockIdxX; block_idx_x += grid_dim_x) 
    {
    const uint block_idx_z = 0;
    ThreadblockSwizzle threadblock_swizzle;
    
    // Compute threadblock location
    cutlass::gemm::GemmCoord threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile, block_idx_x, block_idx_y, block_idx_z);

    // Early exit if CTA is out of range
    if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
      params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {

      return;
    }
    // if (isProducerOrConsumer && threadIdx.x == 0) {
    //   printf("%d %d\n", start_block_idx_x, start_block_idx_y);
    // }
    if (isProducerOrConsumer == false) {
      // Wait for tile of this thread block to be processed by other kernel
      // if (threadIdx.x == 0)
      // if (threadIdx.x == 0)
        // printf("Waiting %d %d %d\n", block_idx_y, block_idx_x, params.overlap_handle.isBlockRemaining[block_idx_y * gridDim.y + block_idx_x]);
        // if (params.overlap_handle.isBlockRemaining[block_idx_x] == 1) 
        {
          // for (int col = 0; col < params.overlap_handle.xSize; col += 128)
            //TODO: Can combine all into one
          params.overlap_handle.waitOnTiles(block_idx_x, 0, 0, 1, params.overlap_handle.ySize/128);
          // printf("426: Waiting %d %d\n", block_idx_y, block_idx_x);
        }
    }

    // if (threadIdx.x == 0) printf("Mma::Shape::kM %d Mma::Shape::kN %d\n", Mma::Shape::kM, Mma::Shape::kN);
    // Compute initial location in logical coordinates
    cutlass::MatrixCoord tb_offset_A{
      threadblock_tile_offset.m() * Mma::Shape::kM,
      threadblock_tile_offset.k() * params.gemm_k_size,
    };

    cutlass::MatrixCoord tb_offset_B{
      threadblock_tile_offset.k() * params.gemm_k_size,
      threadblock_tile_offset.n() * Mma::Shape::kN
    };

    // Problem size is a function of threadblock index in the K dimension
    int problem_size_k = min(
      params.problem_size.k(),
      (threadblock_tile_offset.k() + 1) * params.gemm_k_size);

    // Compute threadblock-scoped matrix multiply-add
    int gemm_k_iterations = (problem_size_k - tb_offset_A.column() + Mma::Shape::kK - 1) / Mma::Shape::kK;

    // Compute position within threadblock
    int thread_idx = threadIdx.x;

    // Construct iterators to A and B operands
    typename Mma::IteratorA iterator_A(
      params.params_A,
      params.ref_A.data(),
      {params.problem_size.m(), problem_size_k},
      thread_idx,
      tb_offset_A,
      params.gather_A_indices);

    typename Mma::IteratorB iterator_B(
      params.params_B,
      params.ref_B.data(),
      {problem_size_k, params.problem_size.n()},
      thread_idx,
      tb_offset_B,
      params.gather_B_indices);

    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    int lane_idx = threadIdx.x % 32;

    //
    // Main loop
    //

    // Construct thread-scoped matrix multiply
    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    typename Mma::FragmentC accumulators;

    accumulators.clear();

    if (!kSplitKSerial || gemm_k_iterations > 0) {
      // Compute threadblock-scoped matrix multiply-add
      mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);
    }

    //
    // Epilogue
    //

    OutputOp output_op(params.output_op);

    //
    // Masked tile iterators constructed from members
    //

    threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile, block_idx_x, block_idx_y, block_idx_z);

    //assume identity swizzle
    MatrixCoord threadblock_offset(
      threadblock_tile_offset.m() * Mma::Shape::kM,
      threadblock_tile_offset.n() * Mma::Shape::kN
    );

    int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

    // Construct the semaphore.
    Semaphore semaphore(params.semaphore + block_idx, thread_idx);

    // If performing a reduction via split-K, fetch the initial synchronization
    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
      
      // Fetch the synchronization lock initially but do not block.
      semaphore.fetch();

      // Indicate which position in a serial reduction the output operator is currently updating
      output_op.set_k_partition(threadblock_tile_offset.k(), params.grid_tiled_shape.k());
    }

    // Tile iterator loading from source tensor.
    typename Epilogue::OutputTileIterator iterator_C(
      params.params_C,
      params.ref_C.data(),
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset,
      params.scatter_D_indices
    );

    // Tile iterator writing to destination tensor.
    typename Epilogue::OutputTileIterator iterator_D(
      params.params_D,
      params.ref_D.data(),
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset,
      params.scatter_D_indices
    );

    Epilogue epilogue(
      shared_storage.epilogue, 
      thread_idx, 
      warp_idx, 
      lane_idx);

    // Wait on the semaphore - this latency may have been covered by iterator construction
    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
        
      // For subsequent threadblocks, the source matrix is held in the 'D' tensor.
      if (threadblock_tile_offset.k()) {
        iterator_C = iterator_D;
      }

      semaphore.wait(threadblock_tile_offset.k());

    }

    // Execute the epilogue operator to update the destination tensor.
    epilogue(output_op, iterator_D, accumulators, iterator_C); 
    
    //
    // Release the semaphore
    //

    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
      
      int lock = 0;
      if (params.grid_tiled_shape.k() == threadblock_tile_offset.k() + 1) {

        // The final threadblock resets the semaphore for subsequent grids.
        lock = 0;
      }
      else {
        // Otherwise, the semaphore is incremented
        lock = threadblock_tile_offset.k() + 1;
      }

      semaphore.release(lock);
    }

    // //Tile of this thread block is processed
    if (isProducerOrConsumer)
      // if (threadIdx.x == 0)
        // if (params.overlap_handle.isBlockRemaining[block_idx_x] == 1)
          params.overlap_handle.setTileStatus(block_idx_x, 0, 0, 1);

  }}}
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

