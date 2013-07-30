# Copyright 2013 Stanford University
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


# If using the general low-level runtime
# select a GASNET conduit to use
CONDUIT = ibv
#CONDUIT = gemini
#CONDUIT = mpi
#CONDUIT = udp

# If using the general low-level runtime
# select a target GPU architecture
GPU_ARCH = fermi
#GPU_ARCH = kepler
#GPU_ARCH = k20

ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined, aborting build)
endif

# Handle some of the common machines we frequent

ifeq ($(shell uname -n),sapling-head)
GASNET = /usr/local/gasnet-1.20.0-openmpi
MPI = /usr/local/openmpi-1.6.4
CUDA = /usr/local/cuda-5.0
CONDUIT = ibv
GPU_ARCH = fermi
endif
ifeq ($(shell uname -n),n0000)
GASNET = /usr/local/gasnet-1.20.0-openmpi
MPI = /usr/local/openmpi-1.6.4
CUDA = /usr/local/cuda-5.0
CONDUIT = ibv
GPU_ARCH = fermi
endif
ifeq ($(shell uname -n),n0001)
GASNET = /usr/local/gasnet-1.20.0-openmpi
MPI = /usr/local/openmpi-1.6.4
CUDA = /usr/local/cuda-5.0
CONDUIT = ibv
GPU_ARCH = fermi
endif
ifeq ($(shell uname -n),n0002)
GASNET = /usr/local/gasnet-1.20.0-openmpi
MPI = /usr/local/openmpi-1.6.4
CUDA = /usr/local/cuda-5.0
CONDUIT = ibv
GPU_ARCH = fermi
endif
ifeq ($(shell uname -n),n0003)
GASNET = /usr/local/gasnet-1.20.0-openmpi
MPI = /usr/local/openmpi-1.6.4
CUDA = /usr/local/cuda-5.0
CONDUIT = ibv
GPU_ARCH = fermi
endif
ifeq ($(findstring nics.utk.edu,$(shell uname -n)),nics.utk.edu)
GASNET = /nics/d/home/sequoia/gasnet-1.20.2-openmpi
MPI = /sw/kfs/openmpi/1.6.1/centos6.2_intel2011_sp1.11.339
CUDA = /sw/kfs/cuda/4.2/linux_binary
CONDUIT = ibv
GPU_ARCH = fermi
endif
ifeq ($(findstring titan,$(shell uname -n)),titan)
#GASNET = /sw/xk6/gasnet/1.20.2/cle4.1_gnu4.7.2_gnumpi_fast
GASNET = /ccs/home/mebauer/gasnet-1.20.2-build
CUDA = /opt/nvidia/cudatoolkit/5.0.35.102
CONDUIT = gemini
GPU_ARCH = k20
LD_FLAGS += -L/opt/cray/ugni/4.0-1.0401.5928.9.5.gem/lib64/ 
LD_FLAGS += -L/opt/cray/pmi/4.0.1-1.0000.9421.73.3.gem/lib64/
endif

INC_FLAGS	+= -I$(LG_RT_DIR)
LD_FLAGS	+= -lrt -lpthread

# Falgs for running in the general low-level runtime
ifndef SHARED_LOWLEVEL

ifndef CUDA
$(error CUDA variable is not defined, aborting build)
endif

ifndef GASNET
$(error GASNET variable is not defined, aborting build)
endif

# General CUDA variables
INC_FLAGS	+= -I$(CUDA)/include 
ifdef DEBUG
NVCC_FLAGS	+= -DDEBUG_LOW_LEVEL -DDEBUG_HIGH_LEVEL -g
#NVCC_FLAGS	+= -G
else
NVCC_FLAGS	+= -O2
endif
LD_FLAGS	+= -L$(CUDA)/lib64 -lcudart -Xlinker -rpath=$(CUDA)/lib64
# CUDA arch variables
ifeq ($(GPU_ARCH),fermi)
NVCC_FLAGS	+= -arch=compute_20 -code=sm_20
NVCC_FLAGS	+= -DFERMI_ARCH
endif
ifeq ($(GPU_ARCH),kepler)
NVCC_FLAGS	+= -arch=compute_30 -code=sm_30
NVCC_FLAGS	+= -DKEPLER_ARCH
endif
ifeq ($(GPU_ARCH),k20)
NVCC_FLAGS	+= -arch=compute_35 -code=sm_35
NVCC_FLAGS	+= -DK20_ARCH
endif
NVCC_FLAGS	+= -Xptxas "-v -abi=no"

# General GASNET variables
INC_FLAGS	+= -I$(GASNET)/include
LD_FLAGS	+= -L$(GASNET)/lib -lrt -lm
# GASNET conduit variables
ifeq ($(CONDUIT),ibv)
INC_FLAGS 	+= -I$(GASNET)/include/ibv-conduit
CC_FLAGS	+= -DGASNET_CONDUIT_IBV
LD_FLAGS	+= -lgasnet-ibv-par -libverbs
endif
ifeq ($(CONDUIT),gemini)
INC_FLAGS	+= -I$(GASNET)/include/gemini-conduit
CC_FLAGS	+= -DGASNET_CONDUIT_GEMINI
LD_FLAGS	+= -lgasnet-gemini-par -lugni -lpmi -lhugetlbfs
endif
ifeq ($(CONDUIT),mpi)
INC_FLAGS	+= -I$(GASNET)/include/mpi-conduit
CC_FLAGS	+= -DGASNET_CONDUIT_MPI
LD_FLAGS	+= -lgasnet-mpi-par -lammpi -lmpi
endif
ifeq ($(CONDUIT),udp)
INC_FLAGS	+= -I$(GASNET)/include/udp-conduit
CC_FLAGS	+= -DGASNET_CONDUIT_UDP
LD_FLAGS	+= -lgasnet-udp-par -lamudp
endif

#Extra options for MPI
ifdef MPI
CC 	:= mpicc
CXX	:= mpicxx
GCC	:= $(CXX)
LD_FLAGS+= -L$(MPI)/lib -lmpi
endif

endif # ifndef SHARED_LOWLEVEL


ifdef DEBUG
CC_FLAGS	+= -DDEBUG_LOW_LEVEL -DDEBUG_HIGH_LEVEL -g #-ggdb -Wall
else
CC_FLAGS	+= -O2 
endif


# Manage the output setting
CC_FLAGS	+= -DCOMPILE_TIME_MIN_LEVEL=$(OUTPUT_LEVEL)

#CC_FLAGS += -DUSE_MASKED_COPIES

# Set the source files
ifndef SHARED_LOWLEVEL
LOW_RUNTIME_SRC	+= $(LG_RT_DIR)/lowlevel.cc $(LG_RT_DIR)/lowlevel_gpu.cc
LOW_RUNTIME_SRC += $(LG_RT_DIR)/activemsg.cc $(LG_RT_DIR)/lowlevel_dma.cc
GPU_RUNTIME_SRC +=
else
CC_FLAGS	+= -DSHARED_LOWLEVEL
LOW_RUNTIME_SRC	+= $(LG_RT_DIR)/shared_lowlevel.cc 
endif

# If you want to go back to using the shared mapper, comment out the next line
# and uncomment the one after that
MAPPER_SRC	+= $(LG_RT_DIR)/default_mapper.cc $(LG_RT_DIR)/mapping_utilities.cc
#MAPPER_SRC	+= $(LG_RT_DIR)/shared_mapper.cc
ifdef ALT_MAPPERS
MAPPER_SRC	+= $(LG_RT_DIR)/alt_mappers.cc
endif

HIGH_RUNTIME_SRC += $(LG_RT_DIR)/legion.cc $(LG_RT_DIR)/legion_ops.cc $(LG_RT_DIR)/region_tree.cc $(LG_RT_DIR)/legion_logging.cc 
