CUTLASS=../../
CUSYNC=../../../cusync/src/include/

INCLUDES=-I$(CUTLASS)/include -I$(CUTLASS)/examples/common -I$(CUTLASS)/tools/util/include -I$(CUSYNC)

all: mlp-rowsync mlp-tilesync

streamk: streamk.cu
	/usr/local/cuda-11.2/bin/nvcc  -DCUTLASS_ENABLE_CUBLAS=1 -DCUTLASS_NAMESPACE=cutlass $(INCLUDES) -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu -DNDEBUG ./streamk.cu -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -O3

streamk-eval: streamk-eval.cu
	/usr/local/cuda-11.2/bin/nvcc  -DCUTLASS_ENABLE_CUBLAS=1 -DCUTLASS_NAMESPACE=cutlass $(INCLUDES) -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu -DNDEBUG ./streamk-eval.cu -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -O3


mlp-batchedrow: mlp.cu common.h
	/usr/local/cuda-11.2/bin/nvcc  -DCUTLASS_ENABLE_CUBLAS=1 -DCUTLASS_NAMESPACE=cutlass $(INCLUDES) -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu -DNDEBUG ./mlp.cu -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -O3 -DBATCHEDROW

mlp-tilebatchsync: mlp.cu common.h
	/usr/local/cuda-11.2/bin/nvcc  -DCUTLASS_ENABLE_CUBLAS=1 -DCUTLASS_NAMESPACE=cutlass $(INCLUDES) -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu -DNDEBUG ./mlp.cu -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -O3 -DTILEBATCH -DCUSTOM_ORDER

mlp-rowsync: mlp.cu common.h
	/usr/local/cuda-11.2/bin/nvcc  -DCUTLASS_ENABLE_CUBLAS=1 -DCUTLASS_NAMESPACE=cutlass $(INCLUDES) -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu -DNDEBUG ./mlp.cu -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -O3 -DROWSYNC

mlp-tilesync: mlp.cu common.h
	/usr/local/cuda-11.2/bin/nvcc $(INCLUDES) -DCUTLASS_ENABLE_CUBLAS=1 -DCUTLASS_NAMESPACE=cutlass  -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu -DNDEBUG ./mlp.cu -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -O3 -DTILESYNC

mlp-eval: mlp-eval.cu common.h
	/usr/local/cuda-11.2/bin/nvcc $(INCLUDES) -DCUTLASS_ENABLE_CUBLAS=1 -DCUTLASS_NAMESPACE=cutlass  -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu -DNDEBUG ./mlp-eval.cu -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -O3

attention-rowsync: attention.cu
	/usr/local/cuda-11.2/bin/nvcc  -DCUTLASS_ENABLE_CUBLAS=1 -DCUTLASS_NAMESPACE=cutlass $(INCLUDES) -O3   -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu -DNDEBUG ./attention.cu -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -DROWSYNC

attention-tilesync: attention.cu
	/usr/local/cuda-11.2/bin/nvcc  -DCUTLASS_ENABLE_CUBLAS=1 -DCUTLASS_NAMESPACE=cutlass $(INCLUDES) -O3   -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu -DNDEBUG ./attention.cu -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -DTILESYNC

attention-gpt3-stridedsync: attention.cu
	/usr/local/cuda-11.2/bin/nvcc  -DCUTLASS_ENABLE_CUBLAS=1 -DCUTLASS_NAMESPACE=cutlass $(INCLUDES) -O3   -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu -DNDEBUG ./attention.cu -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -DSTRIDEDSYNC -DGPT3

attention-llama-stridedsync: attention.cu
	/usr/local/cuda-11.2/bin/nvcc  -DCUTLASS_ENABLE_CUBLAS=1 -DCUTLASS_NAMESPACE=cutlass $(INCLUDES) -O3   -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu -DNDEBUG ./attention.cu -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -DSTRIDEDSYNC -DLLaMA

attention-gpt3-eval: attention-eval.cu
	/usr/local/cuda-11.2/bin/nvcc  -DCUTLASS_ENABLE_CUBLAS=1 -DCUTLASS_NAMESPACE=cutlass $(INCLUDES) -O3   -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu -DNDEBUG ./attention-eval.cu -o attention-eval -Xptxas -v -lcublas -Xcompiler=-fopenmp -DGPT3

attention-llama-eval: attention-eval.cu
	/usr/local/cuda-11.2/bin/nvcc  -DCUTLASS_ENABLE_CUBLAS=1 -DCUTLASS_NAMESPACE=cutlass $(INCLUDES) -O3   -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu -DNDEBUG ./attention-eval.cu -o attention-eval -Xptxas -v -lcublas -Xcompiler=-fopenmp -DLLaMA

clean:
	rm -f mlp-rowsync mlp-tilesync attention-rowsync attention-tilesync
