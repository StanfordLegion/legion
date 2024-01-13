# Copyright 2023 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined, aborting build)
endif

# Flags for directing the runtime makefile what to include
DEBUG           ?= 1		# Include debugging symbols
OUTPUT_LEVEL    ?= LEVEL_DEBUG	# Compile time logging level
USE_CUDA        ?= 0		# Include CUDA support (requires CUDA)
USE_GASNET      ?= 0		# Include GASNet support (requires GASNet)
USE_HDF         ?= 0		# Include HDF5 support (requires HDF5)
ALT_MAPPERS     ?= 0		# Include alternative mappers (not recommended)
USE_OPENMP      ?= 1
USE_OPENMP_SYSTEM_RUNTIME ?= 1
BLAS_IMPL       ?= OPENBLAS
BLAS_DIR ?= /scratch2/wwu/OpenBLAS/install-omp
#BLAS_DIR ?= /scratch2/wwu/anaconda3/envs/omp
# BLAS_DIR ?= /scratch2/wwu/anaconda3/envs/blis

ifeq ($(strip $(BLAS_DIR)),"")
    echo "Please specify the BLAS_DIR"
    exit 1
endif

# Put the binary file name here
OUTFILE		?= concurrent_omp_dgemm 
# List all the application source files here
GEN_SRC		?= concurrent_omp_dgemm.cc		# .cc files
GEN_GPU_SRC	?=				# .cu files

# You can modify these variables, some will be appended to by the runtime makefile
CC_FLAGS	?=
NVCC_FLAGS	?=
GASNET_FLAGS	?=
ifeq ($(strip $(BLAS_IMPL)),OPENBLAS)
INC_FLAGS	?= -I$(BLAS_DIR)/include -DBLAS_USE_OPENBLAS
LD_FLAGS	?= -L$(BLAS_DIR)/lib -lopenblas
else ifeq ($(strip $(BLAS_IMPL)),MKL)
INC_FLAGS	?= -I$(BLAS_DIR)/include -DBLAS_USE_MKL
LD_FLAGS	?= -L$(BLAS_DIR)/lib -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
else ifeq ($(strip $(BLAS_IMPL)),BLIS)
INC_FLAGS	?= -I$(BLAS_DIR)/include/blis -DBLAS_USE_BLIS
LD_FLAGS	?= -L$(BLAS_DIR)/lib -lblis
endif

###########################################################################
#
#   Don't change anything below here
#   
###########################################################################

include $(LG_RT_DIR)/runtime.mk

