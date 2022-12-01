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

/*
  This example demonstrates how to call a CUTLASS GEMM kernel and provides a naive reference
  matrix multiply kernel to verify its correctness.

  The CUTLASS Gemm template is instantiated in the function CutlassSgemmNN. This is kernel computes
  the general matrix product (GEMM) using single-precision floating-point arithmetic and assumes
  all matrices have column-major layout.

  The threadblock tile size is chosen as 128x128x8 which offers good performance for large matrices.
  See the CUTLASS Parallel for All blog post for more exposition on the tunable parameters available
  in CUTLASS.

  https://devblogs.nvidia.com/cutlass-linear-algebra-cuda/

  Aside from defining and launching the SGEMM kernel, this example does not use any other components
  or utilities within CUTLASS. Such utilities are demonstrated elsewhere in other examples and are
  prevalent in the CUTLASS unit tests.

  This example has delibrately been kept similar to the basic_gemm example from cutass-1.3 to 
  highlight the minimum amount of differences needed to transition to cutlass-2.0.

  Cutlass-1.3 sgemm: https://github.com/NVIDIA/cutlass/blob/master/examples/00_basic_gemm/basic_gemm.cu
*/

// Standard Library includes
#include <iostream>
#include <sstream>
#include <vector>

#include "cutlass/overlap_handle.h"

// Helper methods to check for errors
#include "helper.h"

//
// CUTLASS includes needed for single-precision GEMM kernel
//

// Defines cutlass::gemm::device::Gemm, the generic Gemm computation template class.
#include "cutlass/gemm/device/gemm.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// This function defines a CUTLASS GEMM kernel instantiation, constructs its parameters object,
// and launches it on the CUDA device.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t CutlassSgemmNN(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc,
  int M2, int N2, int K2,
  float* D,
  int ldd,
  float* E,
  int lde,
  OverlapHandle& handle,
  cudaStream_t producer_stream, cudaStream_t consumer_stream,
  int iters = 100) {

  // Define type definition for single-precision CUTLASS GEMM with column-major
  // input matrices and 128x128x8 threadblock tile size (chosen by default).
  //
  // To keep the interface manageable, several helpers are defined for plausible compositions
  // including the following example for single-precision GEMM. Typical values are used as
  // default template arguments. See `cutlass/gemm/device/default_gemm_configuration.h` for more details.
  //
  // To view the full gemm device API interface, see `cutlass/gemm/device/gemm.h`

  using ColumnMajor = cutlass::layout::ColumnMajor;

  using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                  ColumnMajor,  // Layout of A matrix
                                                  float,        // Data-type of B matrix
                                                  ColumnMajor,  // Layout of B matrix
                                                  float,        // Data-type of C matrix
                                                  ColumnMajor>; // Layout of C matrix
  
  CutlassGemm gemm_operator;

  // Define a CUTLASS GEMM type

  // Construct the CUTLASS GEMM arguments object.
  //
  // One of CUTLASS's design patterns is to define gemm argument objects that are constructible
  // in host code and passed to kernels by value. These may include pointers, strides, scalars,
  // and other arguments needed by Gemm and its components.
  //
  // The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
  // arguments to kernels and (2.) minimized initialization overhead on kernel entry.
  //
  
  for (int r = 0; r < iters; r++) {
    handle.iter += 1;
    handle.producerOrConsumer_ = true;
    //C = A * B
    CutlassGemm::Arguments args(handle,
                                {M, N, K},  // Gemm Problem dimensions
                                {A, lda},    // Tensor-ref for source matrix A
                                {B, ldb},    // Tensor-ref for source matrix B
                                {C, ldc},    // Tensor-ref for source matrix C
                                {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                {alpha, beta}); // Scalars used in the Epilogue

    //
    // Launch the CUTLASS GEMM kernel.
    //
    
    cutlass::Status status = gemm_operator(args, handle.enable(), NULL, producer_stream);
    assert(M == M2);
    assert(N == K2);
    
    handle.producerOrConsumer_ = false;

    //E = C * D
    CutlassGemm::Arguments args2(handle,
      {M2, N2, K2},  // Gemm Problem dimensions
      {C, ldc},    // Tensor-ref for source matrix A
      {D, ldd},    // Tensor-ref for source matrix B
      {E, lde},    // Tensor-ref for source matrix C
      {E, lde},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
      {alpha, beta}); // Scalars used in the Epilogue

    //
    // Launch the CUTLASS GEMM kernel.
    //

    status = gemm_operator(args2, handle.enable(), NULL, consumer_stream);

    //
    // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
    //

    if (status != cutlass::Status::kSuccess) {
      return cudaErrorUnknown;
    }

    CUDA_CHECK(cudaStreamSynchronize(consumer_stream));
  }

  // Return success, if no errors were encountered.
  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// The source code after this point in the file is generic CUDA using the CUDA Runtime API
// and simple CUDA kernels to initialize matrices and compute the general matrix product.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel to initialize a matrix with small integers.
__global__ void InitializeMatrix_kernel(
  float *matrix,
  int rows,
  int columns,
  int seed = 0) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < rows && j < columns) {
    int offset = i + j * rows;

    // Generate arbitrary elements.
    int const k = 16807;
    int const m = 16;
    float value = float(((offset + seed) * k % m) - m / 2);

    matrix[offset] = value;
  }
}

/// Simple function to initialize a matrix to arbitrary small integers.
cudaError_t InitializeMatrix(float *matrix, int rows, int columns, int seed = 0) {

  dim3 block(16, 16);
  dim3 grid(
    (rows + block.x - 1) / block.x,
    (columns + block.y - 1) / block.y
  );

  InitializeMatrix_kernel<<< grid, block >>>(matrix, rows, columns, seed);

  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocates device memory for a matrix then fills with arbitrary small integers.
cudaError_t AllocateMatrix(float **matrix, int rows, int columns, int seed = 0) {
  cudaError_t result;

  size_t sizeof_matrix = sizeof(float) * rows * columns;

  // Allocate device memory.
  result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Clear the allocation.
  result = cudaMemset(*matrix, 0, sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to clear matrix device memory: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Initialize matrix elements to arbitrary small integers.
  result = InitializeMatrix(*matrix, rows, columns, seed);

  if (result != cudaSuccess) {
    std::cerr << "Failed to initialize matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Naive reference GEMM computation.
__global__ void ReferenceGemm_kernel(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    float accumulator = 0;

    for (int k = 0; k < K; ++k) {
      accumulator += A[i + k * lda] * B[k + j * ldb];
    }

    C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
  }
}

/// Reference GEMM computation.
cudaError_t ReferenceGemm(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc) {

  dim3 block(16, 16);
  dim3 grid(
    (M + block.x - 1) / block.x,
    (N + block.y - 1) / block.y
  );

  ReferenceGemm_kernel<<< grid, block >>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

  return cudaGetLastError();
}

cudaError_t CheckResults(int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C_cutlass,
  float *C_reference,
  int ldc,
  int M2, int N2, int K2,
  float* D,
  int ldd,
  float* E_cutlass,
  float* E_reference,
  int lde) {
  
  cudaError_t result;
  size_t sizeof_C = sizeof(float) * ldc * N;
  size_t sizeof_E = sizeof(float) * lde * N2;

  // Launch reference GEMM
  result = ReferenceGemm(M, N, K, alpha, A, lda, B, ldb, beta, C_reference, ldc);
  
  if (result != cudaSuccess) {
    std::cerr << "Reference GEMM kernel failed: "
      << cudaGetErrorString(result) << std::endl;

    return result;
  }

  result = ReferenceGemm(M2, N2, K2, alpha, C_reference, ldc, D, ldd, beta, E_reference, lde);

  // Copy to host and verify equivalence for C = A * B
  {
    std::vector<float> host_cutlass(ldc * N, 0);
    std::vector<float> host_reference(ldc * N, 0);

    result = cudaMemcpy(host_cutlass.data(), C_cutlass, sizeof_C, cudaMemcpyDeviceToHost);

    if (result != cudaSuccess) {
      std::cerr << "Failed to copy CUTLASS GEMM results: "
        << cudaGetErrorString(result) << std::endl;

    
      return result;
    }

    result = cudaMemcpy(host_reference.data(), C_reference, sizeof_C, cudaMemcpyDeviceToHost);

    if (result != cudaSuccess) {
      std::cerr << "Failed to copy Reference GEMM results: "
        << cudaGetErrorString(result) << std::endl;

    
      return result;
    }
    //
    // Test for bit equivalence of results.
    //

    if (host_cutlass != host_reference) {
      std::cerr << "CUTLASS results incorrect for C = A * B" << std::endl;

      return cudaErrorUnknown;
    }
  }

  // Copy to host and verify equivalence for E = C * D
  {
    std::vector<float> host_cutlass(lde * N2, 0);
    std::vector<float> host_reference(lde * N2, 0);

    result = cudaMemcpy(host_cutlass.data(), E_cutlass, sizeof_E, cudaMemcpyDeviceToHost);

    if (result != cudaSuccess) {
      std::cerr << "Failed to copy CUTLASS GEMM results: "
        << cudaGetErrorString(result) << std::endl;

    
      return result;
    }

    result = cudaMemcpy(host_reference.data(), E_reference, sizeof_E, cudaMemcpyDeviceToHost);

    if (result != cudaSuccess) {
      std::cerr << "Failed to copy Reference GEMM results: "
        << cudaGetErrorString(result) << std::endl;

      return result;
    }

    if (host_cutlass != host_reference) {
      std::cerr << "CUTLASS results incorrect for E = C * D" << std::endl;

      return cudaErrorUnknown;
    }
  }

  std::cout << "Passed" << std::endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
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

/// Allocate several matrices in GPU device memory and call a single-precision
/// CUTLASS GEMM kernel.
cudaError_t TestCutlassGemm(int M, int N, int K, int L, float alpha, float beta) {
  cudaError_t result;

  //
  // Define several matrices to be used as operands to GEMM kernels.
  //

  // Compute leading dimensions for each matrix.
  int lda = M;
  int ldb = K;
  int ldc = M;
  int ldd = N;
  int lde = M;

  // Compute size in bytes of the C matrix.
  size_t sizeof_C = sizeof(float) * ldc * N;
  size_t sizeof_E = sizeof(float) * lde * L;

  // Define pointers to matrices in GPU device memory.
  float *A;
  float *B;
  float *C_cutlass;
  float *C_reference;
  float *D;
  float *E_cutlass;
  float* E_reference;

  //
  // Allocate matrices in GPU device memory with arbitrary seeds.
  //

  result = AllocateMatrix(&A, M, K, 0);

  if (result !=  cudaSuccess) {
    return result;
  }

  result = AllocateMatrix(&B, K, N, 17);

  if (result !=  cudaSuccess) {
    cudaFree(A);
    return result;
  }

  result = AllocateMatrix(&C_cutlass, M, N, 101);

  if (result != cudaSuccess) {
    cudaFree(A);
    cudaFree(B);
    return result;
  }

  result = AllocateMatrix(&C_reference, M, N, 101);

  if (result != cudaSuccess) {
    cudaFree(A);
    cudaFree(B);
    cudaFree(C_cutlass);
    return result;
  }
  
  result = cudaMemcpy(C_reference, C_cutlass, sizeof_C, cudaMemcpyDeviceToDevice);

  result = AllocateMatrix(&D, N, L, 17);

  if (result !=  cudaSuccess) {
    cudaFree(A);
    return result;
  }

  result = AllocateMatrix(&E_cutlass, M, L, 101);

  if (result != cudaSuccess) {
    return result;
  }

  result = AllocateMatrix(&E_reference, M, L, 101);

  if (result != cudaSuccess) {
    return result;
  }
  
  result = cudaMemcpy(E_reference, E_cutlass, sizeof_E, cudaMemcpyDeviceToDevice);

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy C_cutlass matrix to C_reference: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  //
  // Launch Baseline Dependant GEMMs.
  //
  int epochs = 100;
  int warmup = 10;
  cudaStream_t producer_stream;
  OverlapHandle baselineHandle;
  baselineHandle.setGridDims(0, 0, 0);
  CUDA_CHECK(cudaStreamCreate(&producer_stream));
  
  cudaEvent_t start;
  cudaEvent_t end;
  float baselineTime = 0;
  
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));

  {
    result = CutlassSgemmNN(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc, M, L, N, D, ldd, E_cutlass, lde, baselineHandle, producer_stream, producer_stream, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (result != cudaSuccess) {
      std::cerr << "CUTLASS GEMM kernel failed: "
        << cudaGetErrorString(result) << std::endl;
      return result;
    }

    //
    // Verify.
    //
    CheckResults(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, C_reference, ldc, M, L, N, D, ldd, E_cutlass, E_reference, lde);
    //warmup
    result = CutlassSgemmNN(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc, M, L, N, D, ldd, E_cutlass, lde, baselineHandle, producer_stream, producer_stream, warmup);
    CUDA_CHECK(cudaDeviceSynchronize());

    // if (result != cudaSuccess) {
    //   std::cerr << "CUTLASS GEMM kernel failed: "
    //     << cudaGetErrorString(result) << std::endl;
    //   return result;
    // }

    CUDA_CHECK(cudaEventRecord(start, producer_stream));
    
    result = CutlassSgemmNN(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc, M, L, N, D, ldd, E_cutlass, lde, baselineHandle, producer_stream, producer_stream, epochs);
    if (result != cudaSuccess) {
      std::cerr << "CUTLASS GEMM kernel failed: "
        << cudaGetErrorString(result) << std::endl;
      return result;
    }
    CUDA_CHECK(cudaEventRecord(end, producer_stream));
    CUDA_CHECK(cudaEventSynchronize(end));
    CUDA_CHECK(cudaEventElapsedTime(&baselineTime, start, end));
    printf("baseline elapsedtime %f milliseconds\n", baselineTime/(float)epochs);
  }

  //Launch overlapped gemms
  CUDA_CHECK(cudaMemset(C_cutlass, 0, sizeof_C));
  CUDA_CHECK(cudaMemset(C_reference, 0, sizeof_C));
  CUDA_CHECK(cudaMemset(E_cutlass, 0, sizeof_E));
  CUDA_CHECK(cudaMemset(E_reference, 0, sizeof_E));
  cudaStream_t consumer_stream;
  OverlapHandle overlapHandle(N, M, 1, 1);
  overlapHandle.setGridDims(1,(M/128 >= 80) ? 79 : 0, 1);
  CUDA_CHECK(cudaStreamCreate(&consumer_stream));
  overlapHandle.allocTileStatusMap(128, 128, 1);
  double overlapTime = 0;
  {
    result = CutlassSgemmNN(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc, M, L, N, D, ldd, E_cutlass, lde, overlapHandle, producer_stream, consumer_stream, 1);
    if (result != cudaSuccess) {
      std::cerr << "CUTLASS GEMM kernel failed: "
        << cudaGetErrorString(result) << std::endl;
      return result;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CheckResults(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, C_reference, ldc, M, L, N, D, ldd, E_cutlass, E_reference, lde);

    //warmup
    result = CutlassSgemmNN(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc, M, L, N, D, ldd, E_cutlass, lde, overlapHandle, producer_stream, consumer_stream, warmup);
    if (result != cudaSuccess) {
      std::cerr << "CUTLASS GEMM kernel failed: "
        << cudaGetErrorString(result) << std::endl;
      return result;
    }

    double startTime = convertTimeValToDouble(getTimeOfDay());
    result = CutlassSgemmNN(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc, M, L, N, D, ldd, E_cutlass, lde, overlapHandle, producer_stream, consumer_stream, epochs);
    if (result != cudaSuccess) {
      std::cerr << "CUTLASS GEMM kernel failed: "
        << cudaGetErrorString(result) << std::endl;
      return result;
    }
    // CUDA_CHECK(cudaStreamSynchronize(consumer_stream));
    // CUDA_CHECK(cudaStreamSynchronize(producer_stream));
    double endTime = convertTimeValToDouble(getTimeOfDay());
    overlapTime = (endTime - startTime)/1e3; //convert from microseconds to milliseconds
    // printf("612: endTime %lf startTime %lf elapsed %lf\n",  endTime, startTime, endTime - startTime);
    printf("overlapped elapsedtime %lf milliseconds\n", overlapTime/(float)epochs);
  }

  printf("speedup %lf\n", baselineTime/(double)overlapTime);

  //
  // Free device memory allocations.
  //

  cudaFree(C_reference);
  cudaFree(C_cutlass);
  cudaFree(B);
  cudaFree(A);

  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Entry point to basic_gemm example.
//
// usage:
//
//   00_basic_gemm <M> <N> <K> <alpha> <beta>
//
int main(int argc, const char *arg[]) {

  //
  // Parse the command line to obtain GEMM dimensions and scalar values.
  //

  // GEMM problem dimensions.
  int problem[4] = { 128, 128, 128, 128 };

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

  printf("problem[0] %d problem[1] %d problem[2] %d problem[3] %d\n", problem[0], problem[1], problem[2], problem[3]);
  cudaError_t result = TestCutlassGemm(
    problem[0],     // GEMM M dimension
    problem[1],     // GEMM N dimension
    problem[2],     // GEMM K dimension
    problem[3],     // 2nd GEMM L dimension
    scalars[0],     // alpha
    scalars[1]     // beta
  );


  // Exit.
  return result == cudaSuccess ? 0 : -1;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
