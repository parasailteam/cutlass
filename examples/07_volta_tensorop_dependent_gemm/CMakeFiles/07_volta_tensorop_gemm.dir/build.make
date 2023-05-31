# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/parasail/parasail-cutlass

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/parasail/parasail-cutlass

# Include any dependencies generated for this target.
include examples/07_volta_tensorop_gemm/CMakeFiles/07_volta_tensorop_gemm.dir/depend.make

# Include the progress variables for this target.
include examples/07_volta_tensorop_gemm/CMakeFiles/07_volta_tensorop_gemm.dir/progress.make

# Include the compile flags for this target's objects.
include examples/07_volta_tensorop_gemm/CMakeFiles/07_volta_tensorop_gemm.dir/flags.make

examples/07_volta_tensorop_gemm/CMakeFiles/07_volta_tensorop_gemm.dir/volta_tensorop_gemm.cu.o: examples/07_volta_tensorop_gemm/CMakeFiles/07_volta_tensorop_gemm.dir/flags.make
examples/07_volta_tensorop_gemm/CMakeFiles/07_volta_tensorop_gemm.dir/volta_tensorop_gemm.cu.o: examples/07_volta_tensorop_gemm/volta_tensorop_gemm.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/parasail/parasail-cutlass/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object examples/07_volta_tensorop_gemm/CMakeFiles/07_volta_tensorop_gemm.dir/volta_tensorop_gemm.cu.o"
	cd /home/parasail/parasail-cutlass/examples/07_volta_tensorop_gemm && /usr/local/cuda-10.2/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/parasail/parasail-cutlass/examples/07_volta_tensorop_gemm/volta_tensorop_gemm.cu -o CMakeFiles/07_volta_tensorop_gemm.dir/volta_tensorop_gemm.cu.o

examples/07_volta_tensorop_gemm/CMakeFiles/07_volta_tensorop_gemm.dir/volta_tensorop_gemm.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/07_volta_tensorop_gemm.dir/volta_tensorop_gemm.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

examples/07_volta_tensorop_gemm/CMakeFiles/07_volta_tensorop_gemm.dir/volta_tensorop_gemm.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/07_volta_tensorop_gemm.dir/volta_tensorop_gemm.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target 07_volta_tensorop_gemm
07_volta_tensorop_gemm_OBJECTS = \
"CMakeFiles/07_volta_tensorop_gemm.dir/volta_tensorop_gemm.cu.o"

# External object files for target 07_volta_tensorop_gemm
07_volta_tensorop_gemm_EXTERNAL_OBJECTS =

examples/07_volta_tensorop_gemm/07_volta_tensorop_gemm: examples/07_volta_tensorop_gemm/CMakeFiles/07_volta_tensorop_gemm.dir/volta_tensorop_gemm.cu.o
examples/07_volta_tensorop_gemm/07_volta_tensorop_gemm: examples/07_volta_tensorop_gemm/CMakeFiles/07_volta_tensorop_gemm.dir/build.make
examples/07_volta_tensorop_gemm/07_volta_tensorop_gemm: /usr/lib/x86_64-linux-gnu/libcublas.so
examples/07_volta_tensorop_gemm/07_volta_tensorop_gemm: /usr/lib/x86_64-linux-gnu/libcublasLt.so
examples/07_volta_tensorop_gemm/07_volta_tensorop_gemm: examples/07_volta_tensorop_gemm/CMakeFiles/07_volta_tensorop_gemm.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/parasail/parasail-cutlass/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable 07_volta_tensorop_gemm"
	cd /home/parasail/parasail-cutlass/examples/07_volta_tensorop_gemm && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/07_volta_tensorop_gemm.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/07_volta_tensorop_gemm/CMakeFiles/07_volta_tensorop_gemm.dir/build: examples/07_volta_tensorop_gemm/07_volta_tensorop_gemm

.PHONY : examples/07_volta_tensorop_gemm/CMakeFiles/07_volta_tensorop_gemm.dir/build

examples/07_volta_tensorop_gemm/CMakeFiles/07_volta_tensorop_gemm.dir/clean:
	cd /home/parasail/parasail-cutlass/examples/07_volta_tensorop_gemm && $(CMAKE_COMMAND) -P CMakeFiles/07_volta_tensorop_gemm.dir/cmake_clean.cmake
.PHONY : examples/07_volta_tensorop_gemm/CMakeFiles/07_volta_tensorop_gemm.dir/clean

examples/07_volta_tensorop_gemm/CMakeFiles/07_volta_tensorop_gemm.dir/depend:
	cd /home/parasail/parasail-cutlass && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/parasail/parasail-cutlass /home/parasail/parasail-cutlass/examples/07_volta_tensorop_gemm /home/parasail/parasail-cutlass /home/parasail/parasail-cutlass/examples/07_volta_tensorop_gemm /home/parasail/parasail-cutlass/examples/07_volta_tensorop_gemm/CMakeFiles/07_volta_tensorop_gemm.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/07_volta_tensorop_gemm/CMakeFiles/07_volta_tensorop_gemm.dir/depend
