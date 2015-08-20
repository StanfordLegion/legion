# Copyright 2015 Stanford University, NVIDIA Corporation
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
GPU_ARCH ?= fermi
#GPU_ARCH ?= kepler
#GPU_ARCH ?= k20

# if CUDA is not set, but CUDATOOLKIT_HOME is, use that
ifdef CUDATOOLKIT_HOME
CUDA ?= $(CUDATOOLKIT_HOME)
endif

ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined, aborting build)
endif

# defaults for GASNet
CONDUIT ?= udp
ifdef GASNET_ROOT
GASNET ?= $(GASNET_ROOT)
else
GASNET ?= $(LG_RT_DIR)/gasnet/release
endif

# Handle some of the common machines we frequent

# machine architecture (generally "native" unless cross-compiling)
MARCH ?= native

ifeq ($(shell uname -n),sapling)
CONDUIT=ibv
GPU_ARCH=fermi
endif
ifeq ($(shell uname -n),n0000)
CONDUIT=ibv
GPU_ARCH=fermi
endif
ifeq ($(shell uname -n),n0001)
CONDUIT=ibv
GPU_ARCH=fermi
endif
ifeq ($(shell uname -n),n0002)
CONDUIT=ibv
GPU_ARCH=fermi
endif
ifeq ($(shell uname -n),n0003)
CONDUIT=ibv
GPU_ARCH=fermi
endif
ifeq ($(findstring nics.utk.edu,$(shell uname -n)),nics.utk.edu)
GASNET=/nics/d/home/sequoia/gasnet-1.20.2-openmpi
MPI=/sw/kfs/openmpi/1.6.1/centos6.2_intel2011_sp1.11.339
CUDA=/sw/kfs/cuda/4.2/linux_binary
CONDUIT=ibv
GPU_ARCH=fermi
endif
ifeq ($(findstring titan,$(shell uname -n)),titan)
GCC=CC
F90=ftn
# without this, lapack stuff will link, but generate garbage output - thanks Cray!
LAPACK_LIBS=-L/opt/acml/5.3.1/gfortran64_fma4/lib -Wl,-rpath=/opt/acml/5.3.1/gfortran64_fma4/lib -lacml
MARCH=bdver1
CC_FLAGS += -DGASNETI_BUG1389_WORKAROUND=1
CUDA=${CUDATOOLKIT_HOME}
CONDUIT=gemini
GPU_ARCH=k20
LD_FLAGS += ${CRAY_UGNI_POST_LINK_OPTS}
LD_FLAGS += ${CRAY_PMI_POST_LINK_OPTS}
endif
ifeq ($(findstring daint,$(shell uname -n)),daint)
GCC=CC
F90=ftn
# Cray's magic wrappers automatically provide LAPACK goodness?
LAPACK_LIBS=
MARCH=corei7-avx
CC_FLAGS += -DGASNETI_BUG1389_WORKAROUND=1
CUDA=${CUDATOOLKIT_HOME}
CONDUIT=aries
GPU_ARCH=k20
LD_FLAGS += ${CRAY_UGNI_POST_LINK_OPTS}
LD_FLAGS += ${CRAY_PMI_POST_LINK_OPTS}
endif

ifneq (${MARCH},)
  CC_FLAGS += -march=${MARCH}
endif

INC_FLAGS	+= -I$(LG_RT_DIR) -I$(LG_RT_DIR)/realm
ifneq ($(shell uname -s),Darwin)
LD_FLAGS	+= -lrt -lpthread
else
LD_FLAGS	+= -lpthread
endif

USE_LIBDL = 1
ifeq ($(strip $(USE_LIBDL)),1)
#CC_FLAGS += -rdynamic
LD_FLAGS += -ldl -rdynamic
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

# General CUDA variables
ifeq ($(strip $(USE_CUDA)),1)
CC_FLAGS        += -DUSE_CUDA
NVCC_FLAGS      += -DUSE_CUDA
INC_FLAGS	+= -I$(CUDA)/include 
ifeq ($(strip $(DEBUG)),1)
NVCC_FLAGS	+= -DDEBUG_LOW_LEVEL -DDEBUG_HIGH_LEVEL -g
#NVCC_FLAGS	+= -G
else
NVCC_FLAGS	+= -O2
endif
ifneq ($(shell uname -s),Darwin)
LD_FLAGS	+= -L$(CUDA)/lib64 -lcuda -Xlinker -rpath=$(CUDA)/lib64
else
LD_FLAGS	+= -L$(CUDA)/lib -lcuda
endif
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

# general low-level uses GASNet by default
USE_GASNET ?= 1
ifeq ($(strip $(USE_GASNET)),1)
  ifndef GASNET
    $(error GASNET variable is not defined, aborting build)
  endif

  # General GASNET variables
  INC_FLAGS	+= -I$(GASNET)/include
  ifneq ($(shell uname -s),Darwin)
    LD_FLAGS	+= -L$(GASNET)/lib -lrt -lm
  else
    LD_FLAGS	+= -L$(GASNET)/lib -lm
  endif 
  CC_FLAGS	+= -DUSE_GASNET
  # newer versions of gasnet seem to need this
  CC_FLAGS	+= -DGASNETI_BUG1389_WORKAROUND=1

  # GASNET conduit variables
  ifeq ($(strip $(CONDUIT)),ibv)
    INC_FLAGS 	+= -I$(GASNET)/include/ibv-conduit
    CC_FLAGS	+= -DGASNET_CONDUIT_IBV
    LD_FLAGS	+= -lgasnet-ibv-par -libverbs
    # GASNet needs MPI for interop support
    USE_MPI	= 1
  endif
  ifeq ($(strip $(CONDUIT)),gemini)
    INC_FLAGS	+= -I$(GASNET)/include/gemini-conduit
    CC_FLAGS	+= -DGASNET_CONDUIT_GEMINI
    LD_FLAGS	+= -lgasnet-gemini-par -lugni -lpmi -lhugetlbfs
    # GASNet needs MPI for interop support
    USE_MPI	= 1
  endif
  ifeq ($(strip $(CONDUIT)),aries)
    INC_FLAGS   += -I$(GASNET)/include/aries-conduit
    CC_FLAGS    += -DGASNET_CONDUIT_ARIES
    LD_FLAGS    += -lgasnet-aries-par -lugni -lpmi -lhugetlbfs
    # GASNet needs MPI for interop support
    USE_MPI	= 1
  endif
  ifeq ($(strip $(CONDUIT)),mpi)
    INC_FLAGS	+= -I$(GASNET)/include/mpi-conduit
    CC_FLAGS	+= -DGASNET_CONDUIT_MPI
    LD_FLAGS	+= -lgasnet-mpi-par -lammpi -lmpi
    USE_MPI	= 1
  endif
  ifeq ($(strip $(CONDUIT)),udp)
    INC_FLAGS	+= -I$(GASNET)/include/udp-conduit
    CC_FLAGS	+= -DGASNET_CONDUIT_UDP
    LD_FLAGS	+= -lgasnet-udp-par -lamudp
  endif

endif

# general low-level doesn't use HDF by default
USE_HDF ?= 0
ifeq ($(strip $(USE_HDF)), 1)
  CC_FLAGS      += -DUSE_HDF
  LD_FLAGS      += -lhdf5
endif

SKIP_MACHINES= titan% daint%
#Extra options for MPI support in GASNet
ifeq ($(strip $(USE_MPI)),1)
  # Skip any machines on this list list
  ifeq ($(filter-out $(SKIP_MACHINES),$(shell uname -n)),$(shell uname -n))
    CC		:= mpicc
    CXX		:= mpicxx
    GCC		:= $(CXX)
    F90         := mpif90
    LD_FLAGS	+= -L$(MPI)/lib -lmpi
    LAPACK_LIBS ?= -lblas
  endif
endif

endif # ifeq SHARED_LOWLEVEL


ifeq ($(strip $(DEBUG)),1)
CC_FLAGS	+= -DDEBUG_LOW_LEVEL -DDEBUG_HIGH_LEVEL -ggdb #-ggdb -Wall
else
CC_FLAGS	+= -O2 -fno-strict-aliasing #-ggdb
endif


# Manage the output setting
CC_FLAGS	+= -DCOMPILE_TIME_MIN_LEVEL=$(OUTPUT_LEVEL)

# demand warning-free compilation
CC_FLAGS        += -Wall -Wno-strict-overflow -Werror

#CC_FLAGS += -DUSE_MASKED_COPIES

LOW_RUNTIME_SRC	?=
HIGH_RUNTIME_SRC?=
GPU_RUNTIME_SRC	?=
MAPPER_SRC	?=
ASM_SRC		?=

# Set the source files
ifeq ($(strip $(SHARED_LOWLEVEL)),0)
LOW_RUNTIME_SRC	+= $(LG_RT_DIR)/lowlevel.cc $(LG_RT_DIR)/lowlevel_disk.cc
ifeq ($(strip $(USE_CUDA)),1)
LOW_RUNTIME_SRC += $(LG_RT_DIR)/lowlevel_gpu.cc
endif
ifeq ($(strip $(USE_GASNET)),1)
LOW_RUNTIME_SRC += $(LG_RT_DIR)/activemsg.cc
endif
LOW_RUNTIME_SRC += $(LG_RT_DIR)/lowlevel_dma.cc \
	           $(LG_RT_DIR)/realm/threads.cc \
	           $(LG_RT_DIR)/realm/tasks.cc \
	           $(LG_RT_DIR)/realm/metadata.cc \
		   $(LG_RT_DIR)/realm/event_impl.cc \
		   $(LG_RT_DIR)/realm/rsrv_impl.cc \
		   $(LG_RT_DIR)/realm/proc_impl.cc \
		   $(LG_RT_DIR)/realm/mem_impl.cc \
		   $(LG_RT_DIR)/realm/inst_impl.cc \
		   $(LG_RT_DIR)/realm/idx_impl.cc \
		   $(LG_RT_DIR)/realm/machine_impl.cc \
		   $(LG_RT_DIR)/realm/runtime_impl.cc
GPU_RUNTIME_SRC +=
else
CC_FLAGS	+= -DSHARED_LOWLEVEL
LOW_RUNTIME_SRC	+= $(LG_RT_DIR)/shared_lowlevel.cc 
endif
LOW_RUNTIME_SRC += $(LG_RT_DIR)/realm/logging.cc \
		   $(LG_RT_DIR)/realm/profiling.cc \
		   $(LG_RT_DIR)/realm/operation.cc \
		   $(LG_RT_DIR)/realm/timers.cc

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
		    $(LG_RT_DIR)/legion_c.cc \
		    $(LG_RT_DIR)/legion_ops.cc \
		    $(LG_RT_DIR)/legion_tasks.cc \
		    $(LG_RT_DIR)/legion_trace.cc \
		    $(LG_RT_DIR)/legion_spy.cc \
		    $(LG_RT_DIR)/legion_profiling.cc \
		    $(LG_RT_DIR)/region_tree.cc \
		    $(LG_RT_DIR)/runtime.cc \
		    $(LG_RT_DIR)/garbage_collection.cc

# General shell commands
SHELL	:= /bin/sh
SH	:= sh
RM	:= rm -f
LS	:= ls
MKDIR	:= mkdir
MV	:= mv
CP	:= cp
SED	:= sed
ECHO	:= echo
TOUCH	:= touch
MAKE	:= make
ifndef GCC
GCC	:= g++
endif
ifndef NVCC
NVCC	:= $(CUDA)/bin/nvcc
endif
SSH	:= ssh
SCP	:= scp

GEN_OBJS	:= $(GEN_SRC:.cc=.o)
LOW_RUNTIME_OBJS:= $(LOW_RUNTIME_SRC:.cc=.o)
HIGH_RUNTIME_OBJS:=$(HIGH_RUNTIME_SRC:.cc=.o)
MAPPER_OBJS	:= $(MAPPER_SRC:.cc=.o)
ASM_OBJS	:= $(ASM_SRC:.S=.o)
# Only compile the gpu objects if we need to 
ifeq ($(strip $(SHARED_LOWLEVEL)),0)
GEN_GPU_OBJS	:= $(GEN_GPU_SRC:.cu=.o)
GPU_RUNTIME_OBJS:= $(GPU_RUNTIME_SRC:.cu=.o)
else
GEN_GPU_OBJS	:=
GPU_RUNTIME_OBJS:=
endif

ALL_OBJS	:= $(GEN_OBJS) $(GEN_GPU_OBJS) $(LOW_RUNTIME_OBJS) $(HIGH_RUNTIME_OBJS) $(GPU_RUNTIME_OBJS) $(MAPPER_OBJS) $(ASM_OBJS)

ifndef NO_BUILD_RULES
.PHONY: all
all: $(OUTFILE)

# If we're using the general low-level runtime we have to link with nvcc
$(OUTFILE) : $(ALL_OBJS)
	@echo "---> Linking objects into one binary: $(OUTFILE)"
	$(GCC) -o $(OUTFILE) $(ALL_OBJS) $(LD_FLAGS) $(GASNET_FLAGS)

$(GEN_OBJS) : %.o : %.cc
	$(GCC) -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)

$(ASM_OBJS) : %.o : %.S
	$(GCC) -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)

$(LOW_RUNTIME_OBJS) : %.o : %.cc
	$(GCC) -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)

$(HIGH_RUNTIME_OBJS) : %.o : %.cc
	$(GCC) -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)

$(MAPPER_OBJS) : %.o : %.cc
	$(GCC) -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)

$(GEN_GPU_OBJS) : %.o : %.cu
	$(NVCC) -o $@ -c $< $(INC_FLAGS) $(NVCC_FLAGS)

$(GPU_RUNTIME_OBJS): %.o : %.cu
	$(NVCC) -o $@ -c $< $(INC_FLAGS) $(NVCC_FLAGS)

clean:
	@$(RM) -rf $(ALL_OBJS) $(OUTFILE)
endif

