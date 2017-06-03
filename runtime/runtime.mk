# Copyright 2017 Stanford University, NVIDIA Corporation
# Copyright 2017 Los Alamos National Laboratory 
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

ifeq ($(shell uname -s),Darwin)
DARWIN = 1
CC_FLAGS += -DDARWIN
else
#use disk unless on DARWIN 
CC_FLAGS += -DUSE_DISK 
endif

# If using CUDA select a target GPU architecture
GPU_ARCH ?= fermi
#GPU_ARCH ?= kepler
#GPU_ARCH ?= k20
#GPU_ARCH ?= pascal

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
endif

# generate libraries for Legion and Realm
SLIB_LEGION     := liblegion.a
SLIB_REALM      := librealm.a
LEGION_LIBS     := -L. -llegion -lrealm

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
ifeq ($(findstring xs,$(shell uname -n)), xs)
GPU_ARCH=k80
GASNET=/home/stanford/aaiken/users/zhihao/tools/gasnet/release/
CONDUIT=ibv#not sure if this is true
CUDA=${CUDA_HOME}
endif
ifeq ($(findstring nics.utk.edu,$(shell uname -n)),nics.utk.edu)
GASNET=/nics/d/home/sequoia/gasnet-1.20.2-openmpi
MPI=/sw/kfs/openmpi/1.6.1/centos6.2_intel2011_sp1.11.339
CUDA=/sw/kfs/cuda/4.2/linux_binary
CONDUIT=ibv
GPU_ARCH=fermi
endif
ifeq ($(findstring titan,$(shell uname -n)),titan)
CXX=CC
F90=ftn
# without this, lapack stuff will link, but generate garbage output - thanks Cray!
LAPACK_LIBS=-L/opt/acml/5.3.1/gfortran64_fma4/lib -Wl,-rpath=/opt/acml/5.3.1/gfortran64_fma4/lib -lacml
MARCH=bdver1
CC_FLAGS += -DGASNETI_BUG1389_WORKAROUND=1
CUDA=${CUDATOOLKIT_HOME}
CONDUIT=gemini
GPU_ARCH=k20
LEGION_LD_FLAGS += ${CRAY_UGNI_POST_LINK_OPTS}
LEGION_LD_FLAGS += ${CRAY_PMI_POST_LINK_OPTS}
endif
ifeq ($(findstring daint,$(shell uname -n)),daint)
CXX=CC
F90=ftn
# Cray's magic wrappers automatically provide LAPACK goodness?
LAPACK_LIBS=
MARCH=corei7-avx
CC_FLAGS += -DGASNETI_BUG1389_WORKAROUND=1
CUDA=${CUDATOOLKIT_HOME}
CONDUIT=aries
GPU_ARCH=k20
LEGION_LD_FLAGS += ${CRAY_UGNI_POST_LINK_OPTS}
LEGION_LD_FLAGS += ${CRAY_PMI_POST_LINK_OPTS}
endif
ifeq ($(findstring excalibur,$(shell uname -n)),excalibur)
CXX=CC
F90=ftn
# Cray's magic wrappers automatically provide LAPACK goodness?
LAPACK_LIBS=
CC_FLAGS += -DGASNETI_BUG1389_WORKAROUND=1
CONDUIT=aries
LEGION_LD_FLAGS += ${CRAY_UGNI_POST_LINK_OPTS}
LEGION_LD_FLAGS += ${CRAY_PMI_POST_LINK_OPTS}
endif
ifeq ($(findstring cori,$(shell uname -n)),cori)
CXX=CC
F90=ftn
# Cray's magic wrappers automatically provide LAPACK goodness?
LAPACK_LIBS=
CC_FLAGS += -DGASNETI_BUG1389_WORKAROUND=1
CONDUIT=aries
LEGION_LD_FLAGS += ${CRAY_UGNI_POST_LINK_OPTS}
LEGION_LD_FLAGS += ${CRAY_PMI_POST_LINK_OPTS}
endif

ifneq (${MARCH},)
  CC_FLAGS += -march=${MARCH}
endif

INC_FLAGS	+= -I$(LG_RT_DIR) -I$(LG_RT_DIR)/realm -I$(LG_RT_DIR)/legion -I$(LG_RT_DIR)/mappers
ifneq ($(shell uname -s),Darwin)
LEGION_LD_FLAGS	+= -lrt -lpthread
else
LEGION_LD_FLAGS	+= -lpthread
endif

ifeq ($(strip $(USE_HWLOC)),1)
  ifndef HWLOC 
    $(error HWLOC variable is not defined, aborting build)
  endif
  CC_FLAGS        += -DREALM_USE_HWLOC
  INC_FLAGS   += -I$(HWLOC)/include
  LEGION_LD_FLAGS += -L$(HWLOC)/lib -lhwloc
endif

ifeq ($(strip $(USE_PAPI)),1)
  ifndef PAPI_ROOT
    ifdef PAPI
      PAPI_ROOT = $(PAPI)
    else
      $(error USE_PAPI set, but neither PAPI nor PAPI_ROOT is defined, aborting build)
    endif
  endif
  CC_FLAGS        += -DREALM_USE_PAPI
  INC_FLAGS   += -I$(PAPI_ROOT)/include
  LEGION_LD_FLAGS += -L$(PAPI_ROOT)/lib -lpapi
endif

USE_LIBDL ?= 1
ifeq ($(strip $(USE_LIBDL)),1)
ifneq ($(shell uname -s),Darwin)
#CC_FLAGS += -rdynamic
LEGION_LD_FLAGS += -ldl -rdynamic
else
LEGION_LD_FLAGS += -ldl -Wl,-export_dynamic
endif
endif

USE_LLVM ?= 0
ifeq ($(strip $(USE_LLVM)),1)
  # prefer 3.5 (actually, require it right now)
  LLVM_CONFIG ?= $(shell which llvm-config-3.5 llvm-config | head -1)
  ifeq ($(LLVM_CONFIG),)
    $(error cannot find llvm-config-* - set with LLVM_CONFIG if not in path)
  endif
  CC_FLAGS += -DREALM_USE_LLVM
  # NOTE: do not use these for all source files - just the ones that include llvm include files
  LLVM_CXXFLAGS ?= -std=c++11 -I$(shell $(LLVM_CONFIG) --includedir)
  LEGION_LD_FLAGS += $(shell $(LLVM_CONFIG) --ldflags --libs irreader jit mcjit x86)
  # llvm-config --system-libs gives you all the libraries you might need for anything,
  #  which includes things we don't need, and might not be installed
  # by default, filter out libedit
  LLVM_SYSTEM_LIBS ?= $(filter-out -ledit,$(shell $(LLVM_CONFIG) --system-libs))
  LEGION_LD_FLAGS += $(LLVM_SYSTEM_LIBS)
endif

USE_OPENMP ?= 0
ifeq ($(strip $(USE_OPENMP)),1)
  CC_FLAGS += -DREALM_USE_OPENMP
  REALM_OPENMP_GOMP_SUPPORT ?= 1
  ifeq ($(strip $(REALM_OPENMP_GOMP_SUPPORT)),1)
    CC_FLAGS += -DREALM_OPENMP_GOMP_SUPPORT
  endif
  REALM_OPENMP_KMP_SUPPORT ?= 1	
  ifeq ($(strip $(REALM_OPENMP_KMP_SUPPORT)),1)
    CC_FLAGS += -DREALM_OPENMP_KMP_SUPPORT
  endif
endif

# Flags for Realm

# Realm uses CUDA if requested
ifeq ($(strip $(CUDA)),)
  USE_CUDA ?= 0
  ifeq ($(strip $(USE_CUDA)),1)
    $(error CUDA variable is not defined, aborting build)
  endif
else
  USE_CUDA ?= 1
endif

# General CUDA variables
ifeq ($(strip $(USE_CUDA)),1)
CC_FLAGS        += -DUSE_CUDA
NVCC_FLAGS      += -DUSE_CUDA
INC_FLAGS	+= -I$(CUDA)/include 
ifeq ($(strip $(DEBUG)),1)
NVCC_FLAGS	+= -DDEBUG_REALM -DDEBUG_LEGION -g -O0
#NVCC_FLAGS	+= -G
else
NVCC_FLAGS	+= -O2
endif
ifeq ($(strip $(DARWIN)),1)
LEGION_LD_FLAGS	+= -L$(CUDA)/lib -lcuda
else
LEGION_LD_FLAGS	+= -L$(CUDA)/lib64 -lcuda -Xlinker -rpath=$(CUDA)/lib64
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
ifeq ($(strip $(GPU_ARCH)),pascal)
NVCC_FLAGS	+= -arch=compute_60 -code=sm_60
NVCC_FLAGS	+= -DPASCAL_ARCH
endif
NVCC_FLAGS	+= -Xptxas "-v" #-abi=no"
endif

# Realm uses GASNet if requested
ifeq ($(strip $(GASNET)),)
  USE_GASNET ?= 0
  ifeq ($(strip $(USE_GASNET)),1)
    $(error GASNET variable is not defined, aborting build)
  endif
else
  USE_GASNET ?= 1
endif

ifeq ($(strip $(USE_GASNET)),1)
  # General GASNET variables
  INC_FLAGS	+= -I$(GASNET)/include
  ifeq ($(strip $(DARWIN)),1)
    LEGION_LD_FLAGS	+= -L$(GASNET)/lib -lm
  else
    LEGION_LD_FLAGS	+= -L$(GASNET)/lib -lrt -lm
  endif
  CC_FLAGS	+= -DUSE_GASNET
  # newer versions of gasnet seem to need this
  CC_FLAGS	+= -DGASNETI_BUG1389_WORKAROUND=1

  # GASNET conduit variables
  ifeq ($(strip $(CONDUIT)),ibv)
    INC_FLAGS 	+= -I$(GASNET)/include/ibv-conduit
    CC_FLAGS	+= -DGASNET_CONDUIT_IBV
    LEGION_LD_FLAGS	+= -lgasnet-ibv-par -libverbs
    # GASNet needs MPI for interop support
    USE_MPI	= 1
  endif
  ifeq ($(strip $(CONDUIT)),gemini)
    INC_FLAGS	+= -I$(GASNET)/include/gemini-conduit
    CC_FLAGS	+= -DGASNET_CONDUIT_GEMINI
    LEGION_LD_FLAGS	+= -lgasnet-gemini-par -lugni -lpmi -lhugetlbfs
    # GASNet needs MPI for interop support
    USE_MPI	= 1
  endif
  ifeq ($(strip $(CONDUIT)),aries)
    INC_FLAGS   += -I$(GASNET)/include/aries-conduit
    CC_FLAGS    += -DGASNET_CONDUIT_ARIES
    LEGION_LD_FLAGS    += -lgasnet-aries-par -lugni -lpmi -lhugetlbfs
    # GASNet needs MPI for interop support
    USE_MPI	= 1
  endif
  ifeq ($(strip $(CONDUIT)),psm)
    INC_FLAGS 	+= -I$(GASNET)/include/psm-conduit
    CC_FLAGS	+= -DGASNET_CONDUIT_PSM
    LEGION_LD_FLAGS	+= -lgasnet-psm-par -lpsm2 -lpmi2 # PMI2 is required for OpenMPI
    # GASNet needs MPI for interop support
    USE_MPI	= 1
  endif
  ifeq ($(strip $(CONDUIT)),mpi)
    INC_FLAGS	+= -I$(GASNET)/include/mpi-conduit
    CC_FLAGS	+= -DGASNET_CONDUIT_MPI
    LEGION_LD_FLAGS	+= -lgasnet-mpi-par -lammpi -lmpi
    USE_MPI	= 1
  endif
  ifeq ($(strip $(CONDUIT)),udp)
    INC_FLAGS	+= -I$(GASNET)/include/udp-conduit
    CC_FLAGS	+= -DGASNET_CONDUIT_UDP
    LEGION_LD_FLAGS	+= -lgasnet-udp-par -lamudp
  endif

endif

# Realm doesn't use HDF by default
USE_HDF ?= 0
HDF_LIBNAME ?= hdf5
ifeq ($(strip $(USE_HDF)), 1)
  CC_FLAGS      += -DUSE_HDF -I/usr/include/hdf5/serial
  LEGION_LD_FLAGS      += -l$(HDF_LIBNAME)
endif

SKIP_MACHINES= titan% daint% excalibur% cori%
#Extra options for MPI support in GASNet
ifeq ($(strip $(USE_MPI)),1)
  # Skip any machines on this list list
  ifeq ($(filter-out $(SKIP_MACHINES),$(shell uname -n)),$(shell uname -n))
    CC		:= mpicc
    CXX		:= mpicxx
    F90         := mpif90
    LEGION_LD_FLAGS	+= -L$(MPI)/lib -lmpi
    LAPACK_LIBS ?= -lblas
  endif
endif


# libz
USE_ZLIB ?= 1
ZLIB_LIBNAME ?= z
ifeq ($(strip $(USE_ZLIB)),1)
  CC_FLAGS      += -DUSE_ZLIB
  LEGION_LD_FLAGS += -l$(ZLIB_LIBNAME)
endif


ifeq ($(strip $(DEBUG)),1)
CC_FLAGS	+= -DDEBUG_REALM -DDEBUG_LEGION -ggdb #-ggdb -Wall
else
CC_FLAGS	+= -O2 -fno-strict-aliasing #-ggdb
endif


# Manage the output setting
CC_FLAGS	+= -DCOMPILE_TIME_MIN_LEVEL=$(OUTPUT_LEVEL)

# demand warning-free compilation
CC_FLAGS        += -Wall -Wno-strict-overflow
ifeq ($(strip $(WARN_AS_ERROR)),1)
CC_FLAGS        += -Werror
endif

#CC_FLAGS += -DUSE_MASKED_COPIES

LOW_RUNTIME_SRC	?=
HIGH_RUNTIME_SRC?=
GPU_RUNTIME_SRC	?=
MAPPER_SRC	?=
ASM_SRC		?=

# Set the source files
LOW_RUNTIME_SRC += $(LG_RT_DIR)/realm/runtime_impl.cc \
	           $(LG_RT_DIR)/realm/transfer/channel.cc \
	           $(LG_RT_DIR)/realm/transfer/channel_disk.cc \
	           $(LG_RT_DIR)/realm/transfer/lowlevel_dma.cc \
	           $(LG_RT_DIR)/realm/module.cc \
	           $(LG_RT_DIR)/realm/threads.cc \
	           $(LG_RT_DIR)/realm/faults.cc \
		   $(LG_RT_DIR)/realm/operation.cc \
	           $(LG_RT_DIR)/realm/tasks.cc \
	           $(LG_RT_DIR)/realm/metadata.cc \
		   $(LG_RT_DIR)/realm/event_impl.cc \
		   $(LG_RT_DIR)/realm/rsrv_impl.cc \
		   $(LG_RT_DIR)/realm/proc_impl.cc \
		   $(LG_RT_DIR)/realm/mem_impl.cc \
		   $(LG_RT_DIR)/realm/inst_impl.cc \
		   $(LG_RT_DIR)/realm/idx_impl.cc \
		   $(LG_RT_DIR)/realm/machine_impl.cc \
		   $(LG_RT_DIR)/realm/sampling_impl.cc \
                   $(LG_RT_DIR)/lowlevel.cc \
                   $(LG_RT_DIR)/realm/transfer/lowlevel_disk.cc
LOW_RUNTIME_SRC += $(LG_RT_DIR)/realm/numa/numa_module.cc \
		   $(LG_RT_DIR)/realm/numa/numasysif.cc
ifeq ($(strip $(USE_OPENMP)),1)
LOW_RUNTIME_SRC += $(LG_RT_DIR)/realm/openmp/openmp_module.cc \
		   $(LG_RT_DIR)/realm/openmp/openmp_threadpool.cc \
		   $(LG_RT_DIR)/realm/openmp/openmp_api.cc
endif
LOW_RUNTIME_SRC += $(LG_RT_DIR)/realm/procset/procset_module.cc
ifeq ($(strip $(USE_CUDA)),1)
LOW_RUNTIME_SRC += $(LG_RT_DIR)/realm/cuda/cuda_module.cc \
		   $(LG_RT_DIR)/realm/cuda/cudart_hijack.cc
endif
ifeq ($(strip $(USE_LLVM)),1)
LOW_RUNTIME_SRC += $(LG_RT_DIR)/realm/llvmjit/llvmjit_module.cc \
                   $(LG_RT_DIR)/realm/llvmjit/llvmjit_internal.cc
endif
ifeq ($(strip $(USE_HDF)),1)
LOW_RUNTIME_SRC += $(LG_RT_DIR)/realm/hdf5/hdf5_module.cc \
		   $(LG_RT_DIR)/realm/hdf5/hdf5_internal.cc
endif
ifeq ($(strip $(USE_GASNET)),1)
LOW_RUNTIME_SRC += $(LG_RT_DIR)/activemsg.cc
endif
GPU_RUNTIME_SRC +=

LOW_RUNTIME_SRC += $(LG_RT_DIR)/realm/logging.cc \
	           $(LG_RT_DIR)/realm/cmdline.cc \
		   $(LG_RT_DIR)/realm/profiling.cc \
	           $(LG_RT_DIR)/realm/codedesc.cc \
		   $(LG_RT_DIR)/realm/timers.cc

MAPPER_SRC	+= $(LG_RT_DIR)/mappers/default_mapper.cc \
		   $(LG_RT_DIR)/mappers/mapping_utilities.cc \
		   $(LG_RT_DIR)/mappers/shim_mapper.cc \
		   $(LG_RT_DIR)/mappers/test_mapper.cc \
		   $(LG_RT_DIR)/mappers/replay_mapper.cc \
		   $(LG_RT_DIR)/mappers/debug_mapper.cc \
		   $(LG_RT_DIR)/mappers/wrapper_mapper.cc

ifeq ($(strip $(ALT_MAPPERS)),1)
MAPPER_SRC	+= $(LG_RT_DIR)/mappers/alt_mappers.cc
endif

HIGH_RUNTIME_SRC += $(LG_RT_DIR)/legion/legion.cc \
		    $(LG_RT_DIR)/legion/legion_c.cc \
		    $(LG_RT_DIR)/legion/legion_ops.cc \
		    $(LG_RT_DIR)/legion/legion_tasks.cc \
		    $(LG_RT_DIR)/legion/legion_context.cc \
		    $(LG_RT_DIR)/legion/legion_trace.cc \
		    $(LG_RT_DIR)/legion/legion_spy.cc \
		    $(LG_RT_DIR)/legion/legion_profiling.cc \
		    $(LG_RT_DIR)/legion/legion_profiling_serializer.cc \
		    $(LG_RT_DIR)/legion/legion_instances.cc \
		    $(LG_RT_DIR)/legion/legion_views.cc \
		    $(LG_RT_DIR)/legion/legion_analysis.cc \
		    $(LG_RT_DIR)/legion/legion_constraint.cc \
		    $(LG_RT_DIR)/legion/legion_mapping.cc \
		    $(LG_RT_DIR)/legion/region_tree.cc \
		    $(LG_RT_DIR)/legion/runtime.cc \
		    $(LG_RT_DIR)/legion/garbage_collection.cc \
		    $(LG_RT_DIR)/legion/mapper_manager.cc

# General shell commands
SHELL	:= /bin/sh
SH	:= sh
RM	:= rm
LS	:= ls
MKDIR	:= mkdir
MV	:= mv
CP	:= cp
SED	:= sed
ECHO	:= echo
TOUCH	:= touch
MAKE	:= make
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
ifeq ($(strip $(USE_CUDA)),1)
GEN_GPU_OBJS	:= $(GEN_GPU_SRC:.cu=.o)
GPU_RUNTIME_OBJS:= $(GPU_RUNTIME_SRC:.cu=.o)
else
GEN_GPU_OBJS	:=
GPU_RUNTIME_OBJS:=
endif

ifndef NO_BUILD_RULES
.PHONY: all
all: $(OUTFILE)

# If we're using CUDA we have to link with nvcc
$(OUTFILE) : $(GEN_OBJS) $(GEN_GPU_OBJS) $(SLIB_LEGION) $(SLIB_REALM)
	@echo "---> Linking objects into one binary: $(OUTFILE)"
	$(CXX) -o $(OUTFILE) $(GEN_OBJS) $(GEN_GPU_OBJS) $(LD_FLAGS) $(LEGION_LIBS) $(LEGION_LD_FLAGS) $(GASNET_FLAGS)

$(SLIB_LEGION) : $(HIGH_RUNTIME_OBJS) $(MAPPER_OBJS)
	rm -f $@
	$(AR) rc $@ $^

$(SLIB_REALM) : $(LOW_RUNTIME_OBJS)
	rm -f $@
	$(AR) rc $@ $^

$(GEN_OBJS) : %.o : %.cc
	$(CXX) -o $@ -c $< $(CC_FLAGS) $(INC_FLAGS)

$(ASM_OBJS) : %.o : %.S
	$(CXX) -o $@ -c $< $(CC_FLAGS) $(INC_FLAGS)

$(LOW_RUNTIME_OBJS) : %.o : %.cc
	$(CXX) -o $@ -c $< $(CC_FLAGS) $(INC_FLAGS)

$(HIGH_RUNTIME_OBJS) : %.o : %.cc
	$(CXX) -o $@ -c $< $(CC_FLAGS) $(INC_FLAGS)

$(MAPPER_OBJS) : %.o : %.cc
	$(CXX) -o $@ -c $< $(CC_FLAGS) $(INC_FLAGS)

$(GEN_GPU_OBJS) : %.o : %.cu
	$(NVCC) -o $@ -c $< $(NVCC_FLAGS) $(INC_FLAGS)

$(GPU_RUNTIME_OBJS): %.o : %.cu
	$(NVCC) -o $@ -c $< $(NVCC_FLAGS) $(INC_FLAGS)

clean::
	$(RM) -f $(OUTFILE) $(SLIB_LEGION) $(SLIB_REALM) $(GEN_OBJS) $(GEN_GPU_OBJS) $(LOW_RUNTIME_OBJS) $(HIGH_RUNTIME_OBJS) $(GPU_RUNTIME_OBJS) $(MAPPER_OBJS) $(ASM_OBJS)

endif

ifeq ($(strip $(USE_LLVM)),1)
llvmjit_internal.o : CC_FLAGS += $(LLVM_CXXFLAGS)
%/llvmjit_internal.o : CC_FLAGS += $(LLVM_CXXFLAGS)
endif
