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
# select a target GPU architecture
GPU_ARCH = fermi
#GPU_ARCH = kepler
#GPU_ARCH = k20

ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined, aborting build)
endif

# defaults for GASNet if they have not been supplied by some other source. this
# is done because a user's configuration (in their project's Makefile, for
# example) may be unconditionally overwritten :-\.
ifndef CONDUIT
CONDUIT = udp
endif
ifndef GASNET
GASNET = $(LG_RT_DIR)/gasnet/release
endif

# Handle some of the common machines we frequent

ifeq ($(shell uname -n),sapling-head)
CC_FLAGS += -march=native
GASNET=/usr/local/gasnet-1.20.0-openmpi
MPI=/usr/local/openmpi-1.6.4
CUDA=/usr/local/cuda-5.0
CONDUIT=ibv
GPU_ARCH=fermi
endif
ifeq ($(shell uname -n),n0000)
CC_FLAGS += -march=native
GASNET=/usr/local/gasnet-1.20.0-openmpi
MPI=/usr/local/openmpi-1.6.4
CUDA=/usr/local/cuda-5.0
CONDUIT=ibv
GPU_ARCH=fermi
endif
ifeq ($(shell uname -n),n0001)
CC_FLAGS += -march=native
GASNET=/usr/local/gasnet-1.20.0-openmpi
MPI=/usr/local/openmpi-1.6.4
CUDA=/usr/local/cuda-5.0
CONDUIT=ibv
GPU_ARCH=fermi
endif
ifeq ($(shell uname -n),n0002)
CC_FLAGS += -march=native
GASNET=/usr/local/gasnet-1.20.0-openmpi
MPI=/usr/local/openmpi-1.6.4
CUDA=/usr/local/cuda-5.0
CONDUIT=ibv
GPU_ARCH=fermi
endif
ifeq ($(shell uname -n),n0003)
CC_FLAGS += -march=native
GASNET=/usr/local/gasnet-1.20.0-openmpi
MPI=/usr/local/openmpi-1.6.4
CUDA=/usr/local/cuda-5.0
CONDUIT=ibv
GPU_ARCH=fermi
endif
ifeq ($(findstring nics.utk.edu,$(shell uname -n)),nics.utk.edu)
CC_FLAGS += -march=native
GASNET=/nics/d/home/sequoia/gasnet-1.20.2-openmpi
MPI=/sw/kfs/openmpi/1.6.1/centos6.2_intel2011_sp1.11.339
CUDA=/sw/kfs/cuda/4.2/linux_binary
CONDUIT=ibv
GPU_ARCH=fermi
endif
ifeq ($(findstring titan,$(shell uname -n)),titan)
GCC=CC
CC_FLAGS += -march=bdver1 -DGASNETI_BUG1389_WORKAROUND=1
GASNET = ${GASNET_ROOT}
#GASNET=/ccs/home/mebauer/gasnet-1.20.2-build
CUDA=${CUDATOOLKIT_HOME}
CONDUIT=gemini
GPU_ARCH=k20
LD_FLAGS += ${CRAY_UGNI_POST_LINK_OPTS}
LD_FLAGS += ${CRAY_PMI_POST_LINK_OPTS}
endif

INC_FLAGS	+= -I$(LG_RT_DIR)
ifneq ($(shell uname -s),Darwin)
LD_FLAGS	+= -lrt -lpthread
else
LD_FLAGS	+= -lpthread
endif

# Falgs for running in the general low-level runtime
ifeq ($(strip $(SHARED_LOWLEVEL)),0)

# general low-level uses CUDA by default
USE_CUDA ?= 1
ifeq ($(strip $(USE_CUDA)),1)
  ifndef CUDA
    $(error CUDA variable is not defined, aborting build)
  endif
endif

ifndef GASNET
$(error GASNET variable is not defined, aborting build)
endif

# General CUDA variables
ifeq ($(strip $(USE_CUDA)),1)
CC_FLAGS        += -DUSE_CUDA
INC_FLAGS	+= -I$(CUDA)/include 
ifeq ($(strip $(DEBUG)),1)
NVCC_FLAGS	+= -DDEBUG_LOW_LEVEL -DDEBUG_HIGH_LEVEL -g
#NVCC_FLAGS	+= -G
else
NVCC_FLAGS	+= -O2
endif
LD_FLAGS	+= -L$(CUDA)/lib64 -lcudart -lcuda -Xlinker -rpath=$(CUDA)/lib64
# CUDA arch variables
ifeq ($(strip $(GPU_ARCH)),fermi)
NVCC_FLAGS	+= -arch=compute_20 -code=sm_20
NVCC_FLAGS	+= -DFERMI_ARCH
endif
ifeq ($(strip $(GPU_ARCH)),kepler)
NVCC_FLAGS	+= -arch=compute_30 -code=sm_30
NVCC_FLAGS	+= -DKEPLER_ARCH
endif
ifeq ($(strip $(GPU_ARCH)),k20)
NVCC_FLAGS	+= -arch=compute_35 -code=sm_35
NVCC_FLAGS	+= -DK20_ARCH
endif
NVCC_FLAGS	+= -Xptxas "-v" #-abi=no"
endif

# General GASNET variables
INC_FLAGS	+= -I$(GASNET)/include
LD_FLAGS	+= -L$(GASNET)/lib -lrt -lm
# GASNET conduit variables
ifeq ($(strip $(CONDUIT)),ibv)
INC_FLAGS 	+= -I$(GASNET)/include/ibv-conduit
CC_FLAGS	+= -DGASNET_CONDUIT_IBV
LD_FLAGS	+= -lgasnet-ibv-par -libverbs
endif
ifeq ($(strip $(CONDUIT)),gemini)
INC_FLAGS	+= -I$(GASNET)/include/gemini-conduit
CC_FLAGS	+= -DGASNET_CONDUIT_GEMINI
LD_FLAGS	+= -lgasnet-gemini-par -lugni -lpmi -lhugetlbfs
endif
ifeq ($(strip $(CONDUIT)),mpi)
INC_FLAGS	+= -I$(GASNET)/include/mpi-conduit
CC_FLAGS	+= -DGASNET_CONDUIT_MPI
LD_FLAGS	+= -lgasnet-mpi-par -lammpi -lmpi
endif
ifeq ($(strip $(CONDUIT)),udp)
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

endif # ifeq SHARED_LOWLEVEL


ifeq ($(strip $(DEBUG)),1)
CC_FLAGS	+= -DDEBUG_LOW_LEVEL -DDEBUG_HIGH_LEVEL -ggdb #-ggdb -Wall
else
CC_FLAGS	+= -O2 #-ggdb
endif


# Manage the output setting
CC_FLAGS	+= -DCOMPILE_TIME_MIN_LEVEL=$(OUTPUT_LEVEL)

#CC_FLAGS += -DUSE_MASKED_COPIES

# Set the source files
ifeq ($(strip $(SHARED_LOWLEVEL)),0)
LOW_RUNTIME_SRC	+= $(LG_RT_DIR)/lowlevel.cc
ifeq ($(strip $(USE_CUDA)),1)
LOW_RUNTIME_SRC += $(LG_RT_DIR)/lowlevel_gpu.cc
endif
LOW_RUNTIME_SRC += $(LG_RT_DIR)/activemsg.cc $(LG_RT_DIR)/lowlevel_dma.cc
GPU_RUNTIME_SRC +=
else
CC_FLAGS	+= -DSHARED_LOWLEVEL
LOW_RUNTIME_SRC	+= $(LG_RT_DIR)/shared_lowlevel.cc 
endif

# If you want to go back to using the shared mapper, comment out the next line
# and uncomment the one after that
MAPPER_SRC	+= $(LG_RT_DIR)/default_mapper.cc \
		   $(LG_RT_DIR)/shim_mapper.cc \
		   $(LG_RT_DIR)/mapping_utilities.cc
#MAPPER_SRC	+= $(LG_RT_DIR)/shared_mapper.cc
ifeq ($(strip $(ALT_MAPPERS)),1)
MAPPER_SRC	+= $(LG_RT_DIR)/alt_mappers.cc
endif

HIGH_RUNTIME_SRC += $(LG_RT_DIR)/legion.cc \
		    $(LG_RT_DIR)/legion_ops.cc \
		    $(LG_RT_DIR)/legion_tasks.cc \
		    $(LG_RT_DIR)/legion_trace.cc \
		    $(LG_RT_DIR)/legion_spy.cc \
		    $(LG_RT_DIR)/region_tree.cc \
		    $(LG_RT_DIR)/runtime.cc \
		    $(LG_RT_DIR)/garbage_collection.cc
