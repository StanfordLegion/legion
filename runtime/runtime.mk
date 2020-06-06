# Copyright 2020 Stanford University, NVIDIA Corporation
# Copyright 2020 Los Alamos National Laboratory
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

# These flags are used to control compiler/linker settings, but should not
#  contain Legion/Realm-configuration-related flags - those should go in
#  {LEGION,REALM}_CC_FLAGS below
CC_FLAGS ?=
FC_FLAGS ?=
LD_FLAGS ?=
SO_FLAGS ?=
INC_FLAGS ?=
# Make sure that NVCC_FLAGS are simple expanded since later we use shell to append to it (performance).
NVCC_FLAGS := $(NVCC_FLAGS)
# These flags are NOT passed on the command line, but are used to
# generate the public-facing legion/realm_defines.h files.
# (Additional flags will be picked up from environment variables of
# the same names.)
LEGION_CC_FLAGS ?=
REALM_CC_FLAGS ?=

# Map some common GNU variable names into our variables
CXXFLAGS ?=
CPPFLAGS ?=
LDLIBS ?=
LDFLAGS ?=
CC_FLAGS += $(CXXFLAGS) $(CPPFLAGS)
FC_FLAGS += $(FFLAGS)
SO_FLAGS += $(LDLIBS)
LD_FLAGS += $(LDFLAGS)

USE_OPENMP ?= 0
ifeq ($(shell uname -s),Darwin)
DARWIN = 1
CC_FLAGS += -DDARWIN
FC_FLAGS += -DDARWIN
ifeq ($(strip $(USE_OPENMP)),1)
$(warning "Some versions of Clang on Mac OSX do not support OpenMP")
endif
endif

ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined, aborting build)
endif

# generate libraries for Legion and Realm
SHARED_OBJECTS ?= 0
ifeq ($(strip $(SHARED_OBJECTS)),0)
SLIB_LEGION     := liblegion.a
SLIB_REALM      := librealm.a
else
CC_FLAGS	+= -fPIC
FC_FLAGS	+= -fPIC
NVCC_FLAGS	+= -Xcompiler -fPIC
ifeq ($(shell uname -s),Darwin)
SLIB_LEGION     := liblegion.dylib
SLIB_REALM      := librealm.dylib
else
SLIB_LEGION     := liblegion.so
SLIB_REALM      := librealm.so
endif
# shared libraries can link against other shared libraries
SLIB_LEGION_DEPS = -L. -lrealm
SLIB_REALM_DEPS  =
ifeq ($(strip $(DARWIN)),1)
SO_FLAGS += -dynamiclib -single_module -undefined dynamic_lookup -fPIC
else
SO_FLAGS += -shared
endif
endif
LEGION_LIBS     := -L. -llegion -lrealm

# generate header files for public-facing defines
DEFINE_HEADERS_DIR ?= $(CURDIR)
LEGION_DEFINES_HEADER := $(DEFINE_HEADERS_DIR)/legion_defines.h
REALM_DEFINES_HEADER := $(DEFINE_HEADERS_DIR)/realm_defines.h

# Handle some of the common machines we frequent

ifeq ($(findstring xs,$(shell uname -n)), xs)
GPU_ARCH ?= k80
GASNET ?= /home/stanford/aaiken/users/zhihao/tools/gasnet/release/
CONDUIT ?= ibv #not sure if this is true
endif
ifeq ($(findstring nics.utk.edu,$(shell uname -n)),nics.utk.edu)
GASNET ?= /nics/d/home/sequoia/gasnet-1.20.2-openmpi
MPI=/sw/kfs/openmpi/1.6.1/centos6.2_intel2011_sp1.11.339
CUDA ?= /sw/kfs/cuda/4.2/linux_binary
CONDUIT ?= ibv
GPU_ARCH ?= fermi
endif
ifeq ($(findstring titan,$(shell uname -n)),titan)
# without this, lapack stuff will link, but generate garbage output - thanks Cray!
LAPACK_LIBS=-L/opt/acml/5.3.1/gfortran64_fma4/lib -Wl,-rpath=/opt/acml/5.3.1/gfortran64_fma4/lib -lacml
MARCH ?= bdver1
CONDUIT ?= gemini
GPU_ARCH ?= k20
endif
ifeq ($(findstring daint,$(shell uname -n)),daint)
CONDUIT ?= aries
GPU_ARCH ?= pascal
endif
ifeq ($(findstring excalibur,$(shell uname -n)),excalibur)
CONDUIT ?=aries
endif
ifeq ($(findstring cori,$(shell uname -n)),cori)
CONDUIT ?= aries
endif
ifeq ($(findstring sh,$(shell uname -n)), sh)
GPU_ARCH ?= pascal
CONDUIT ?= ibv #not sure if this is true
endif

# backwards-compatibility for GASNet builds - any of GASNET_ROOT, GASNET, or
#  USE_GASNET=1 will set REALM_NETWORKS=gasnet1
ifdef GASNET_ROOT
  GASNET ?= $(GASNET_ROOT)
endif
ifneq ($(strip $(GASNET)),)
  ifneq ($(strip $(USE_GASNET)),0)
    REALM_NETWORKS ?= gasnet1
  endif
endif
ifeq ($(strip $(USE_GASNET)),1)
  REALM_NETWORKS ?= gasnet1
endif

# defaults for CUDA
GPU_ARCH ?= auto

# if CUDA is not set, but CUDATOOLKIT_HOME or CUDA_HOME is, use that
ifdef CUDATOOLKIT_HOME
CUDA ?= $(CUDATOOLKIT_HOME)
endif
ifdef CUDA_HOME
CUDA ?= $(CUDA_HOME)
endif

# Customization specific to Cray programming environment
ifneq (${CRAYPE_VERSION},)
CXX=CC
FC=ftn
# Cray's magic wrappers automatically provide LAPACK goodness?
LAPACK_LIBS ?=
LEGION_LD_FLAGS += ${CRAY_UGNI_POST_LINK_OPTS}
LEGION_LD_FLAGS += ${CRAY_UDREG_POST_LINK_OPTS}
LEGION_LD_FLAGS += ${CRAY_PMI_POST_LINK_OPTS}
endif

# machine architecture (generally "native" unless cross-compiling)
MARCH ?= native

ifneq (${MARCH},)
  # Summit/Summitdev are strange and want to have this specified via -mcpu
  # instead of -march. Unclear if this is true in general for PPC.
  ifeq ($(findstring ppc64le,$(shell uname -p)),ppc64le)
    CC_FLAGS += -mcpu=${MARCH} -maltivec -mabi=altivec -mvsx
    FC_FLAGS += -mcpu=${MARCH} -maltivec -mabi=altivec -mvsx
  else
    CC_FLAGS += -march=${MARCH}
    FC_FLAGS += -march=${MARCH}
  endif
endif

INC_FLAGS	+= -I$(DEFINE_HEADERS_DIR) -I$(LG_RT_DIR) -I$(LG_RT_DIR)/mappers
# support libraries are OS specific unfortunately
ifeq ($(shell uname -s),Linux)
LEGION_LD_FLAGS	+= -lrt -lpthread
endif
ifeq ($(strip $(DARWIN)),1)
LEGION_LD_FLAGS	+= -lpthread
endif
ifeq ($(shell uname -s),FreeBSD)
LEGION_LD_FLAGS	+= -lexecinfo -lpthread
endif

USE_HALF ?= 0
ifeq ($(strip $(USE_HALF)),1)
  LEGION_CC_FLAGS += -DLEGION_REDOP_HALF
endif

USE_COMPLEX ?= 0
ifeq ($(strip $(USE_COMPLEX)),1)
  LEGION_CC_FLAGS += -DLEGION_REDOP_COMPLEX
endif

ifeq ($(strip $(USE_HWLOC)),1)
  ifndef HWLOC
    $(error HWLOC variable is not defined, aborting build)
  endif
  REALM_CC_FLAGS += -DREALM_USE_HWLOC
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
  REALM_CC_FLAGS += -DREALM_USE_PAPI
  INC_FLAGS   += -I$(PAPI_ROOT)/include
  LEGION_LD_FLAGS += -L$(PAPI_ROOT)/lib -lpapi
endif

USE_LIBDL ?= 1
ifeq ($(strip $(USE_LIBDL)),1)
REALM_CC_FLAGS += -DREALM_USE_LIBDL
ifneq ($(strip $(DARWIN)),1)
#CC_FLAGS += -rdynamic
# FreeBSD doesn't actually have a separate libdl
ifneq ($(shell uname -s),FreeBSD)
LEGION_LD_FLAGS += -ldl
endif
LEGION_LD_FLAGS += -rdynamic
else
LEGION_LD_FLAGS += -ldl -Wl,-export_dynamic
endif
endif

USE_LLVM ?= 0
ifeq ($(strip $(USE_LLVM)),1)
  # prefer known-working versions, if they can be named explicitly
  LLVM_CONFIG ?= $(shell which llvm-config-3.9 llvm-config-3.8 llvm-config-3.6 llvm-config-3.5 llvm-config-4.0 llvm-config-5.0 llvm-config | head -1)
  ifeq ($(LLVM_CONFIG),)
    $(error cannot find llvm-config-* - set with LLVM_CONFIG if not in path)
  endif
  LLVM_VERSION_NUMBER := $(shell $(LLVM_CONFIG) --version | cut -c1,3)
  REALM_CC_FLAGS += -DREALM_USE_LLVM -DREALM_LLVM_VERSION=$(LLVM_VERSION_NUMBER)

  # NOTE: do not use these for all source files - just the ones that include llvm include files
  LLVM_CXXFLAGS ?= -std=c++11 -I$(shell $(LLVM_CONFIG) --includedir)

  # realm can be configured to allow LLVM library linkage to be optional
  #  (i.e. a per-application choice)
  LLVM_LIBS_OPTIONAL ?= 0
  ifeq ($(strip $(LLVM_LIBS_OPTIONAL)),1)
    REALM_CC_FLAGS += -DREALM_ALLOW_MISSING_LLVM_LIBS
  else
    ifeq ($(LLVM_VERSION_NUMBER),35)
      LLVM_LIBS += $(shell $(LLVM_CONFIG) --ldflags --libs irreader jit mcjit x86)
    else
      LLVM_LIBS += $(shell $(LLVM_CONFIG) --ldflags --libs irreader mcjit x86)
    endif
    LEGION_LD_FLAGS += $(LLVM_LIBS)
  endif

  # llvm-config --system-libs gives you all the libraries you might need for anything,
  #  which includes things we don't need, and might not be installed
  # by default, filter out libedit
  LLVM_SYSTEM_LIBS ?= $(filter-out -ledit,$(shell $(LLVM_CONFIG) --system-libs))
  LEGION_LD_FLAGS += $(LLVM_SYSTEM_LIBS)
endif

OMP_FLAGS ?=
ifeq ($(strip $(USE_OPENMP)),1)
  REALM_CC_FLAGS += -DREALM_USE_OPENMP
  # Add the -fopenmp flag for Linux, but not for Mac as clang doesn't need it
  #ifneq ($(strip $(DARWIN)),1)
  OMP_FLAGS += -fopenmp
  #endif
  REALM_OPENMP_GOMP_SUPPORT ?= 1
  ifeq ($(strip $(REALM_OPENMP_GOMP_SUPPORT)),1)
    REALM_CC_FLAGS += -DREALM_OPENMP_GOMP_SUPPORT
  endif
  REALM_OPENMP_KMP_SUPPORT ?= 1
  ifeq ($(strip $(REALM_OPENMP_KMP_SUPPORT)),1)
    REALM_CC_FLAGS += -DREALM_OPENMP_KMP_SUPPORT
  endif
endif

USE_PYTHON ?= 0
ifeq ($(strip $(USE_PYTHON)),1)
  ifneq ($(strip $(USE_LIBDL)),1)
    $(error USE_PYTHON requires USE_LIBDL)
  endif

  # Attempt to auto-detect location of Python shared library based on
  # the location of Python executable on PATH. We do this because the
  # shared library may not be on LD_LIBRARY_PATH even when the
  # executable is on PATH.

  # Note: Set PYTHON_ROOT to an empty string to skip this logic and
  # defer to the normal search of LD_LIBRARY_PATH instead. Or set
  # PYTHON_LIB to specify the path to the shared library directly.
  ifndef PYTHON_LIB
    ifndef PYTHON_ROOT
      PYTHON_EXE := $(shell which python python3 | head -1)
      ifeq ($(PYTHON_EXE),)
        $(error cannot find python - set PYTHON_ROOT if not in PATH)
      endif
      PYTHON_VERSION_MAJOR := $(shell $(PYTHON_EXE) -c 'import sys; print(sys.version_info.major)')
      PYTHON_VERSION_MINOR := $(shell $(PYTHON_EXE) -c 'import sys; print(sys.version_info.minor)')
      PYTHON_ROOT := $(dir $(PYTHON_EXE))
    endif

    # Try searching for common locations of the Python shared library.
    ifneq ($(strip $(PYTHON_ROOT)),)
      ifeq ($(strip $(DARWIN)),1)
        PYTHON_EXT := dylib
      else
	PYTHON_EXT := so
      endif
      PYTHON_LIB := $(wildcard $(PYTHON_ROOT)/libpython$(PYTHON_VERSION_MAJOR).$(PYTHON_VERSION_MINOR)*.$(PYTHON_EXT))
      ifeq ($(strip $(PYTHON_LIB)),)
        PYTHON_LIB := $(wildcard $(abspath $(PYTHON_ROOT)/../lib/libpython$(PYTHON_VERSION_MAJOR).$(PYTHON_VERSION_MINOR)*.$(PYTHON_EXT)))
        ifeq ($(strip $(PYTHON_LIB)),)
          $(warning cannot find libpython$(PYTHON_VERSION_MAJOR).$(PYTHON_VERSION_MINOR)*.$(PYTHON_EXT) - falling back to using LD_LIBRARY_PATH)
          PYTHON_LIB :=
        endif
      endif
    endif
  endif

  ifneq ($(strip $(PYTHON_LIB)),)
    ifndef FORCE_PYTHON
      ifeq ($(wildcard $(PYTHON_LIB)),)
        $(error cannot find libpython$(PYTHON_VERSION_MAJOR).$(PYTHON_VERSION_MINOR).$(PYTHON_EXT) - PYTHON_LIB set but file does not exist)
      else
        REALM_CC_FLAGS += -DREALM_PYTHON_LIB="\"$(PYTHON_LIB)\""
      endif
    else
      REALM_CC_FLAGS += -DREALM_PYTHON_LIB="\"$(PYTHON_LIB)\""
    endif
  endif

  ifndef PYTHON_VERSION_MAJOR
    $(error cannot auto-detect Python version - please set PYTHON_VERSION_MAJOR)
  else
    REALM_CC_FLAGS += -DREALM_PYTHON_VERSION_MAJOR=$(PYTHON_VERSION_MAJOR)
  endif

  REALM_CC_FLAGS += -DREALM_USE_PYTHON
endif

USE_DLMOPEN ?= 0
ifeq ($(strip $(USE_DLMOPEN)),1)
  ifneq ($(strip $(USE_LIBDL)),1)
    $(error USE_DLMOPEN requires USE_LIBDL)
  endif

  REALM_CC_FLAGS += -DREALM_USE_DLMOPEN
endif

USE_SPY ?= 0
ifeq ($(strip $(USE_SPY)),1)
  LEGION_CC_FLAGS += -DLEGION_SPY
endif

# Flags for Realm

# Realm uses CUDA if requested
ifeq ($(strip $(CUDA)),)
  USE_CUDA ?= 0
  ifeq ($(strip $(USE_CUDA)),1)
    # try to auto-detect CUDA location
    CUDA := $(patsubst %/bin/nvcc,%,$(shell which nvcc | head -1))
    ifeq ($(strip $(CUDA)),)
      $(error CUDA variable is not defined, aborting build)
    else
      $(info auto-detected CUDA at: $(CUDA))
    endif
  endif
else
  USE_CUDA ?= 1
endif

# General CUDA variables
ifeq ($(strip $(USE_CUDA)),1)
NVCC	        ?= $(CUDA)/bin/nvcc
REALM_CC_FLAGS        += -DREALM_USE_CUDA
LEGION_CC_FLAGS       += -DLEGION_USE_CUDA
# provide this for backward-compatibility in applications
CC_FLAGS              += -DUSE_CUDA
FC_FLAGS	      += -DUSE_CUDA
REALM_USE_CUDART_HIJACK ?= 1
# Have this for backwards compatibility
ifdef USE_CUDART_HIJACK
REALM_USE_CUDART_HIJACK = $(USE_CUDART_HIJACK)
endif
ifeq ($(strip $(REALM_USE_CUDART_HIJACK)),1)
REALM_CC_FLAGS        += -DREALM_USE_CUDART_HIJACK
endif
INC_FLAGS	+= -I$(CUDA)/include -I$(LG_RT_DIR)/realm/transfer
ifeq ($(strip $(DEBUG)),1)
NVCC_FLAGS	+= -g -O0
#NVCC_FLAGS	+= -G
else
NVCC_FLAGS	+= -O2
endif
ifeq ($(strip $(DARWIN)),1)
ifeq ($(strip $(REALM_USE_CUDART_HIJACK)),1)
LEGION_LD_FLAGS	+= -L$(CUDA)/lib -lcuda
SLIB_LEGION_DEPS += -L$(CUDA)/lib -lcuda
SLIB_REALM_DEPS	+= -L$(CUDA)/lib -lcuda
else
LEGION_LD_FLAGS	+= -L$(CUDA)/lib -lcudart -lcuda
SLIB_LEGION_DEPS += -L$(CUDA)/lib -lcudart -lcuda
SLIB_REALM_DEPS	+= -L$(CUDA)/lib -lcuda
endif
else
ifeq ($(strip $(REALM_USE_CUDART_HIJACK)),1)
LEGION_LD_FLAGS	+= -L$(CUDA)/lib64 -L$(CUDA)/lib64/stubs -lcuda -Xlinker -rpath=$(CUDA)/lib64
SLIB_LEGION_DEPS += -L$(CUDA)/lib64 -L$(CUDA)/lib64/stubs -lcuda
SLIB_REALM_DEPS += -L$(CUDA)/lib64 -L$(CUDA)/lib64/stubs -lcuda
else
LEGION_LD_FLAGS	+= -L$(CUDA)/lib64 -L$(CUDA)/lib64/stubs -lcudart -lcuda -Xlinker -rpath=$(CUDA)/lib64
SLIB_LEGION_DEPS += -L$(CUDA)/lib64 -L$(CUDA)/lib64/stubs -lcudart -lcuda
SLIB_REALM_DEPS += -L$(CUDA)/lib64 -L$(CUDA)/lib64/stubs -lcuda
endif
endif
# Add support for Legion GPU reductions
LEGION_CC_FLAGS	+= -DLEGION_GPU_REDUCTIONS
# Convert CXXFLAGS and CPPFLAGS to NVCC_FLAGS
# Need to detect whether nvcc supports them directly or to use -Xcompiler
NVCC_FLAGS	+= ${shell                                                              \
		     for FLAG in $(CXXFLAGS); do                                        \
		       ( case "$$FLAG" in -I*) true;; *) false;; esac ||                \
		         $(NVCC) $$FLAG -x cu -c /dev/null -o /dev/null 2> /dev/null )  \
		       && echo "$$FLAG" || echo "-Xcompiler $$FLAG";                    \
		     done}
NVCC_FLAGS	+= ${shell                                                              \
		     for FLAG in $(CPPFLAGS); do                                        \
		       ( case "$$FLAG" in -I*) true;; *) false;; esac ||                \
		         $(NVCC) $$FLAG -x cu -c /dev/null -o /dev/null 2> /dev/null )  \
		       && echo "$$FLAG" || echo "-Xcompiler $$FLAG";                    \
		     done}
# CUDA arch variables

# translate legacy arch names into numbers
ifeq ($(strip $(GPU_ARCH)),fermi)
override GPU_ARCH = 20
NVCC_FLAGS	+= -DFERMI_ARCH
endif
ifeq ($(strip $(GPU_ARCH)),kepler)
override GPU_ARCH = 30
NVCC_FLAGS	+= -DKEPLER_ARCH
endif
ifeq ($(strip $(GPU_ARCH)),k20)
override GPU_ARCH = 35
NVCC_FLAGS	+= -DK20_ARCH
endif
ifeq ($(strip $(GPU_ARCH)),k80)
override GPU_ARCH = 37
NVCC_FLAGS	+= -DK80_ARCH
endif
ifeq ($(strip $(GPU_ARCH)),maxwell)
override GPU_ARCH = 52
NVCC_FLAGS	+= -DMAXWELL_ARCH
endif
ifeq ($(strip $(GPU_ARCH)),pascal)
override GPU_ARCH = 60
NVCC_FLAGS	+= -DPASCAL_ARCH
endif
ifeq ($(strip $(GPU_ARCH)),volta)
override GPU_ARCH = 70
NVCC_FLAGS	+= -DVOLTA_ARCH
endif
ifeq ($(strip $(GPU_ARCH)),turing)
override GPU_ARCH = 75
NVCC_FLAGS	+= -DTURING_ARCH
endif

ifeq ($(strip $(GPU_ARCH)),auto)
  # detect based on what nvcc supports
  ALL_ARCHES = 20 30 32 35 37 50 52 53 60 61 62 70 72 75
  override GPU_ARCH = $(shell for X in $(ALL_ARCHES) ; do \
    $(NVCC) -gencode arch=compute_$$X,code=sm_$$X -cuda -x c++ /dev/null -o /dev/null 2> /dev/null && echo $$X; \
  done)
endif

# finally, convert space-or-comma separated list of architectures (e.g. 35,50)
#  into nvcc -gencode arguments
COMMA=,
NVCC_FLAGS += $(foreach X,$(subst $(COMMA), ,$(GPU_ARCH)),-gencode arch=compute_$(X)$(COMMA)code=sm_$(X))

NVCC_FLAGS += -Xcudafe --diag_suppress=boolean_controlling_expr_is_constant
endif

# Realm uses GASNet if requested
ifneq ($(findstring gasnet1,$(REALM_NETWORKS)),)
  ifeq ($(GASNET),)
    $(error GASNET variable is not defined, aborting build)
  endif

  # Detect conduit, if requested
  CONDUIT ?= auto
  ifeq ($(strip $(CONDUIT)),auto)
    GASNET_PREFERRED_CONDUITS = ibv aries gemini pami mpi udp ofi psm mxm portals4 smp
    GASNET_LIBS_FOUND := $(wildcard $(GASNET_PREFERRED_CONDUITS:%=$(GASNET)/lib/libgasnet-%-par.*))
    ifeq ($(strip $(GASNET_LIBS_FOUND)),)
      $(error No multi-threaded GASNet conduits found in $(GASNET)/lib!)
    endif
    override CONDUIT=$(patsubst libgasnet-%-par,%,$(basename $(notdir $(firstword $(GASNET_LIBS_FOUND)))))
    # double-check that we got an actual conduit name
    ifeq ($(findstring $(CONDUIT),$(GASNET_PREFERRED_CONDUITS)),)
      $(error Problem parsing GASNet conduit name: got "$(CONDUIT)" instead of one of: $(GASNET_PREFERRED_CONDUITS))
    endif
  endif

  # General GASNET variables
  INC_FLAGS	+= -I$(GASNET)/include
  ifeq ($(strip $(DARWIN)),1)
    LEGION_LD_FLAGS	+= -L$(GASNET)/lib -lm
  else
    LEGION_LD_FLAGS	+= -L$(GASNET)/lib -lrt -lm
  endif
  REALM_CC_FLAGS	+= -DREALM_USE_GASNET1
  # newer versions of gasnet seem to need this
  CC_FLAGS	+= -DGASNETI_BUG1389_WORKAROUND=1

  # GASNET conduit variables
  ifeq ($(strip $(CONDUIT)),ibv)
    INC_FLAGS 	+= -I$(GASNET)/include/ibv-conduit
    REALM_CC_FLAGS	+= -DGASNET_CONDUIT_IBV
    LEGION_LD_FLAGS	+= -lgasnet-ibv-par -libverbs
    # GASNet needs MPI for interop support
    USE_MPI	= 1
  endif
  ifeq ($(strip $(CONDUIT)),gemini)
    INC_FLAGS	+= -I$(GASNET)/include/gemini-conduit
    REALM_CC_FLAGS	+= -DGASNET_CONDUIT_GEMINI
    LEGION_LD_FLAGS	+= -lgasnet-gemini-par -lugni -ludreg -lpmi -lhugetlbfs
    # GASNet needs MPI for interop support
    USE_MPI	= 1
  endif
  ifeq ($(strip $(CONDUIT)),aries)
    INC_FLAGS   += -I$(GASNET)/include/aries-conduit
    REALM_CC_FLAGS    += -DGASNET_CONDUIT_ARIES
    LEGION_LD_FLAGS    += -lgasnet-aries-par -lugni -ludreg -lpmi -lhugetlbfs
    # GASNet needs MPI for interop support
    USE_MPI	= 1
  endif
  ifeq ($(strip $(CONDUIT)),psm)
    INC_FLAGS 	+= -I$(GASNET)/include/psm-conduit
    REALM_CC_FLAGS	+= -DGASNET_CONDUIT_PSM
    LEGION_LD_FLAGS	+= -lgasnet-psm-par -lpsm2 -lpmi2 # PMI2 is required for OpenMPI
    # GASNet needs MPI for interop support
    USE_MPI	= 1
  endif
  ifeq ($(strip $(CONDUIT)),mpi)
    INC_FLAGS	+= -I$(GASNET)/include/mpi-conduit
    REALM_CC_FLAGS	+= -DGASNET_CONDUIT_MPI
    LEGION_LD_FLAGS	+= -lgasnet-mpi-par -lammpi -lmpi
    USE_MPI	= 1
  endif
  ifeq ($(strip $(CONDUIT)),udp)
    INC_FLAGS	+= -I$(GASNET)/include/udp-conduit
    REALM_CC_FLAGS	+= -DGASNET_CONDUIT_UDP
    LEGION_LD_FLAGS	+= -lgasnet-udp-par -lamudp
  endif
endif

# Realm uses MPI if requested
ifneq ($(findstring mpi,$(REALM_NETWORKS)),)
    REALM_CC_FLAGS        += -DREALM_USE_MPI
    USE_MPI = 1
endif

# Realm doesn't use HDF by default
USE_HDF ?= 0
HDF_LIBNAME ?= hdf5
ifeq ($(strip $(USE_HDF)), 1)
  REALM_CC_FLAGS      += -DREALM_USE_HDF5
  LEGION_CC_FLAGS     += -DLEGION_USE_HDF5
  # provide this for backward-compatibility in applications
  CC_FLAGS            += -DUSE_HDF
  FC_FLAGS	      += -DUSE_HDF
  LEGION_LD_FLAGS      += -l$(HDF_LIBNAME)
  ifdef HDF_ROOT
       CC_FLAGS    += -I$(HDF_ROOT)/include
       FC_FLAGS    += -I$(HDF_ROOT)/include
       LD_FLAGS    += -L$(HDF_ROOT)/lib
  else
    CC_FLAGS      += -I/usr/include/hdf5/serial
    FC_FLAGS	  += -I/usr/include/hdf5/serial
  endif
endif

SKIP_MACHINES= titan% daint% excalibur% cori%
# use mpi{cc,cxx,f90} compiler wrappers if USE_MPI=1
ifeq ($(strip $(USE_MPI)),1)
  # Skip any machines on this list list
  ifeq ($(filter-out $(SKIP_MACHINES),$(shell uname -n)),$(shell uname -n))
    CC		:= mpicc
    CXX		:= mpicxx
    FC		:= mpif90
    # Summit/Summitdev are strange and link this automatically (but still uses mpicxx).
    # FIXME: Unfortunately you can't match against the Summit hostname right now...
    ifneq ($(findstring ppc64le,$(shell uname -p)),ppc64le)
      LEGION_LD_FLAGS	+= -L$(MPI)/lib -lmpi
    endif
    LAPACK_LIBS ?= -lblas
  endif
endif


# libz
USE_ZLIB ?= 1
ZLIB_LIBNAME ?= z
ifeq ($(strip $(USE_ZLIB)),1)
  LEGION_CC_FLAGS += -DLEGION_USE_ZLIB
  LEGION_LD_FLAGS += -l$(ZLIB_LIBNAME)
endif


ifeq ($(strip $(DEBUG)),1)
CC_FLAGS	+= -O0 -ggdb #-ggdb -Wall
FC_FLAGS	+= -O0 -ggdb
REALM_CC_FLAGS	+= -DDEBUG_REALM
LEGION_CC_FLAGS	+= -DDEBUG_LEGION
else
CC_FLAGS	+= -O2 -fno-strict-aliasing #-ggdb
FC_FLAGS	+= -O2 -fno-strict-aliasing
endif

BOUNDS_CHECKS ?= 0
ifeq ($(strip $(BOUNDS_CHECKS)),1)
LEGION_CC_FLAGS	+= -DBOUNDS_CHECKS
endif

PRIVILEGE_CHECKS ?= 0
ifeq ($(strip $(PRIVILEGE_CHECKS)),1)
LEGION_CC_FLAGS	+= -DPRIVILEGE_CHECKS
endif

# DEBUG_TSAN=1 enables thread sanitizer (data race) checks
ifeq ($(strip $(DEBUG_TSAN)),1)
CC_FLAGS        += -fsanitize=thread -g -DTSAN_ENABLED
LD_FLAGS        += -fsanitize=thread
endif

# Set maximum number of dimensions
ifneq ($(strip ${MAX_DIM}),)
REALM_CC_FLAGS	+= -DREALM_MAX_DIM=$(MAX_DIM)
LEGION_CC_FLAGS	+= -DLEGION_MAX_DIM=$(MAX_DIM)
endif

# Set maximum number of fields
ifneq ($(strip ${MAX_FIELDS}),)
LEGION_CC_FLAGS	+= -DLEGION_MAX_FIELDS=$(MAX_FIELDS)
endif

# Optionally make all Legion warnings fatal
ifeq ($(strip ${LEGION_WARNINGS_FATAL}),1)
LEGION_CC_FLAGS += -DLEGION_WARNINGS_FATAL
endif

# Manage the output setting
REALM_CC_FLAGS	+= -DCOMPILE_TIME_MIN_LEVEL=$(OUTPUT_LEVEL)

# demand warning-free compilation
CC_FLAGS        += -Wall -Wno-strict-overflow
FC_FLAGS	+= -Wall -Wno-strict-overflow
ifeq ($(strip $(WARN_AS_ERROR)),1)
CC_FLAGS        += -Werror
FC_FLAGS	+= -Werror
endif

# if requested, add --defcheck flags to the compile line so that the
#  cxx_defcheck wrapper can verify that source files include the configuration
#  headers properly
USE_DEFCHECK ?= 0
ifeq ($(strip ${USE_DEFCHECK}),1)
  LEGION_DEFCHECK = --defcheck legion_defines.h
  REALM_DEFCHECK = --defcheck realm_defines.h
endif

REALM_SRC	?=
LEGION_SRC	?=
LEGION_GPU_SRC	?=
MAPPER_SRC	?=
ASM_SRC		?=

# Set the source files
REALM_SRC 	+= $(LG_RT_DIR)/realm/runtime_impl.cc \
	           $(LG_RT_DIR)/realm/transfer/transfer.cc \
	           $(LG_RT_DIR)/realm/transfer/channel.cc \
	           $(LG_RT_DIR)/realm/transfer/channel_disk.cc \
	           $(LG_RT_DIR)/realm/transfer/lowlevel_dma.cc \
	           $(LG_RT_DIR)/realm/mutex.cc \
	           $(LG_RT_DIR)/realm/module.cc \
	           $(LG_RT_DIR)/realm/threads.cc \
	           $(LG_RT_DIR)/realm/faults.cc \
		   $(LG_RT_DIR)/realm/operation.cc \
	           $(LG_RT_DIR)/realm/tasks.cc \
	           $(LG_RT_DIR)/realm/metadata.cc \
	           $(LG_RT_DIR)/realm/deppart/partitions.cc \
	           $(LG_RT_DIR)/realm/deppart/sparsity_impl.cc \
	           $(LG_RT_DIR)/realm/deppart/image.cc \
	           $(LG_RT_DIR)/realm/deppart/preimage.cc \
	           $(LG_RT_DIR)/realm/deppart/byfield.cc \
	           $(LG_RT_DIR)/realm/deppart/setops.cc \
		   $(LG_RT_DIR)/realm/event_impl.cc \
		   $(LG_RT_DIR)/realm/rsrv_impl.cc \
		   $(LG_RT_DIR)/realm/proc_impl.cc \
		   $(LG_RT_DIR)/realm/mem_impl.cc \
		   $(LG_RT_DIR)/realm/idx_impl.cc \
		   $(LG_RT_DIR)/realm/inst_impl.cc \
		   $(LG_RT_DIR)/realm/inst_layout.cc \
		   $(LG_RT_DIR)/realm/machine_impl.cc \
		   $(LG_RT_DIR)/realm/sampling_impl.cc \
		   $(LG_RT_DIR)/realm/subgraph_impl.cc \
                   $(LG_RT_DIR)/realm/transfer/lowlevel_disk.cc
# REALM_INST_SRC will be compiled {MAX_DIM}^2 times in parallel
REALM_INST_SRC  += $(LG_RT_DIR)/realm/deppart/image_tmpl.cc \
	           $(LG_RT_DIR)/realm/deppart/preimage_tmpl.cc \
	           $(LG_RT_DIR)/realm/deppart/byfield_tmpl.cc
REALM_SRC 	+= $(LG_RT_DIR)/realm/numa/numa_module.cc \
		   $(LG_RT_DIR)/realm/numa/numasysif.cc
ifneq ($(findstring gasnet1,$(REALM_NETWORKS)),)
REALM_SRC 	+= $(LG_RT_DIR)/realm/gasnet1/gasnet1_module.cc \
                   $(LG_RT_DIR)/realm/gasnet1/gasnetmsg.cc
endif
ifneq ($(findstring mpi,$(REALM_NETWORKS)),)
REALM_SRC 	+= $(LG_RT_DIR)/realm/mpi/mpi_module.cc \
                   $(LG_RT_DIR)/realm/mpi/am_mpi.cc
endif
ifeq ($(strip $(USE_OPENMP)),1)
REALM_SRC 	+= $(LG_RT_DIR)/realm/openmp/openmp_module.cc \
		   $(LG_RT_DIR)/realm/openmp/openmp_threadpool.cc \
		   $(LG_RT_DIR)/realm/openmp/openmp_api.cc
endif
REALM_SRC 	+= $(LG_RT_DIR)/realm/procset/procset_module.cc
ifeq ($(strip $(USE_PYTHON)),1)
REALM_SRC 	+= $(LG_RT_DIR)/realm/python/python_module.cc \
		   $(LG_RT_DIR)/realm/python/python_source.cc
endif
ifeq ($(strip $(USE_CUDA)),1)
REALM_SRC 	+= $(LG_RT_DIR)/realm/cuda/cuda_module.cc
ifeq ($(strip $(REALM_USE_CUDART_HIJACK)),1)
REALM_SRC       += $(LG_RT_DIR)/realm/cuda/cudart_hijack.cc
endif
endif
ifeq ($(strip $(USE_LLVM)),1)
REALM_SRC 	+= $(LG_RT_DIR)/realm/llvmjit/llvmjit_module.cc \
                   $(LG_RT_DIR)/realm/llvmjit/llvmjit_internal.cc
endif
ifeq ($(strip $(USE_HDF)),1)
REALM_SRC 	+= $(LG_RT_DIR)/realm/hdf5/hdf5_module.cc \
		   $(LG_RT_DIR)/realm/hdf5/hdf5_internal.cc \
		   $(LG_RT_DIR)/realm/hdf5/hdf5_access.cc
endif
REALM_SRC 	+= $(LG_RT_DIR)/realm/activemsg.cc \
                   $(LG_RT_DIR)/realm/nodeset.cc \
                   $(LG_RT_DIR)/realm/network.cc

REALM_SRC 	+= $(LG_RT_DIR)/realm/logging.cc \
	           $(LG_RT_DIR)/realm/cmdline.cc \
		   $(LG_RT_DIR)/realm/profiling.cc \
	           $(LG_RT_DIR)/realm/codedesc.cc \
		   $(LG_RT_DIR)/realm/timers.cc

MAPPER_SRC	+= $(LG_RT_DIR)/mappers/default_mapper.cc \
		   $(LG_RT_DIR)/mappers/mapping_utilities.cc \
		   $(LG_RT_DIR)/mappers/shim_mapper.cc \
		   $(LG_RT_DIR)/mappers/test_mapper.cc \
		   $(LG_RT_DIR)/mappers/null_mapper.cc \
		   $(LG_RT_DIR)/mappers/replay_mapper.cc \
		   $(LG_RT_DIR)/mappers/debug_mapper.cc \
		   $(LG_RT_DIR)/mappers/wrapper_mapper.cc \
		   $(LG_RT_DIR)/mappers/forwarding_mapper.cc \
		   $(LG_RT_DIR)/mappers/logging_wrapper.cc

LEGION_SRC 	+= $(LG_RT_DIR)/legion/legion.cc \
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
		    $(LG_RT_DIR)/legion/legion_redop.cc \
		    $(LG_RT_DIR)/legion/region_tree.cc \
		    $(LG_RT_DIR)/legion/runtime.cc \
		    $(LG_RT_DIR)/legion/garbage_collection.cc \
		    $(LG_RT_DIR)/legion/mapper_manager.cc
LEGION_GPU_SRC  += $(LG_RT_DIR)/legion/legion_redop.cu
# LEGION_INST_SRC will be compiled {MAX_DIM}^2 times in parallel
LEGION_INST_SRC  += $(LG_RT_DIR)/legion/region_tree_tmpl.cc

LEGION_FORTRAN_SRC += $(LG_RT_DIR)/legion/legion_f_types.f90 \
		    $(LG_RT_DIR)/legion/legion_f_c_interface.f90 \
		    $(LG_RT_DIR)/legion/legion_f.f90

# Header files for Legion installation
INSTALL_HEADERS += legion.h \
		   realm.h \
		   legion/bitmask.h \
		   legion/legion.inl \
		   legion/legion_agency.h \
		   legion/legion_agency.inl \
		   legion/accessor.h \
		   legion/arrays.h \
		   legion/legion_c.h \
		   legion/legion_config.h \
		   legion/legion_constraint.h \
		   legion/legion_domain.h \
		   legion/legion_domain.inl \
		   legion/legion_mapping.h \
		   legion/legion_mapping.inl \
		   legion/legion_redop.h \
		   legion/legion_redop.inl \
		   legion/legion_stl.h \
		   legion/legion_stl.inl \
		   legion/legion_template_help.h \
		   legion/legion_types.h \
		   mappers/debug_mapper.h \
		   mappers/default_mapper.h \
		   mappers/default_mapper.inl \
		   mappers/mapping_utilities.h \
		   mappers/null_mapper.h \
		   mappers/replay_mapper.h \
		   mappers/shim_mapper.h \
		   mappers/test_mapper.h \
		   mappers/wrapper_mapper.h \
		   mappers/forwarding_mapper.h \
		   mappers/logging_wrapper.h \
		   realm/realm_config.h \
		   realm/realm_c.h \
		   realm/profiling.h \
		   realm/profiling.inl \
		   realm/redop.h \
		   realm/event.h \
		   realm/event.inl \
		   realm/reservation.h \
		   realm/reservation.inl \
		   realm/processor.h \
		   realm/processor.inl \
		   realm/memory.h \
		   realm/instance.h \
		   realm/instance.inl \
		   realm/inst_layout.h \
		   realm/inst_layout.inl \
		   realm/logging.h \
		   realm/logging.inl \
		   realm/machine.h \
		   realm/machine.inl \
		   realm/runtime.h \
		   realm/indexspace.h \
		   realm/indexspace.inl \
		   realm/codedesc.h \
		   realm/codedesc.inl \
		   realm/compiler_support.h \
		   realm/bytearray.h \
		   realm/bytearray.inl \
		   realm/faults.h \
		   realm/faults.inl \
		   realm/atomics.h \
		   realm/atomics.inl \
		   realm/point.h \
		   realm/point.inl \
		   realm/custom_serdez.h \
		   realm/custom_serdez.inl \
		   realm/sparsity.h \
		   realm/sparsity.inl \
		   realm/subgraph.h \
		   realm/subgraph.inl \
		   realm/dynamic_templates.h \
		   realm/dynamic_templates.inl \
		   realm/serialize.h \
		   realm/serialize.inl \
		   realm/timers.h \
		   realm/timers.inl \
		   realm/utils.h \
		   realm/utils.inl

ifeq ($(strip $(USE_HALF)),1)
INSTALL_HEADERS += mathtypes/half.h
endif
ifeq ($(strip $(USE_COMPLEX)),1)
INSTALL_HEADERS += mathtypes/complex.h
endif
ifeq ($(strip $(USE_PYTHON)),1)
INSTALL_HEADERS += realm/python/python_source.h \
		   realm/python/python_source.inl
endif
ifeq ($(strip $(USE_LLVM)),1)
INSTALL_HEADERS += realm/llvm/llvmjit.h \
		   realm/llvm/llvmjit.inl
endif
ifeq ($(strip $(USE_HDF)),1)
INSTALL_HEADERS += realm/hdf5/hdf5_access.h \
		   realm/hdf5/hdf5_access.inl
endif

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
SSH	:= ssh
SCP	:= scp
PYTHON  := $(shell which python python3 | head -1)

ifneq ($(strip ${MAX_DIM}),)
  DIMS := $(shell bash -c "echo {1..$(strip $(MAX_DIM))}")
else
  DIMS := 1 2 3
endif
f_replace1 = $(1:_tmpl.cc=_$(2).cc.o)
f_replace2 = $(1:_tmpl.cc=_$(2)_$(3).cc.o)
f_expand1 = $(foreach N1,$(DIMS),$(call $(1),$(2),$(N1)))
f_expand2 = $(foreach N1,$(DIMS),$(foreach N2,$(DIMS),$(call $(1),$(2),$(N1),$(N2))))
REALM_INST_OBJS := $(call f_expand2,f_replace2,$(REALM_INST_SRC))
# Legion has both 1-dim and 2-dim versions
LEGION_INST_OBJS := $(call f_expand1,f_replace1,$(LEGION_INST_SRC)) \
                    $(call f_expand2,f_replace2,$(LEGION_INST_SRC))

GEN_OBJS	:= $(GEN_SRC:.cc=.cc.o)
REALM_OBJS	:= $(REALM_SRC:.cc=.cc.o)
LEGION_OBJS	:= $(LEGION_SRC:.cc=.cc.o)
MAPPER_OBJS	:= $(MAPPER_SRC:.cc=.cc.o)
ASM_OBJS	:= $(ASM_SRC:.S=.S.o)
# Only compile the gpu objects if we need to
ifeq ($(strip $(USE_CUDA)),1)
GEN_GPU_OBJS	:= $(GEN_GPU_SRC:.cu=.cu.o)
LEGION_GPU_OBJS := $(LEGION_GPU_SRC:.cu=.cu.o)
else
GEN_GPU_OBJS	:=
LEGION_GPU_OBJS :=
endif

LEGION_USE_FORTRAN ?= 0
ifeq ($(strip $(LEGION_USE_FORTRAN)),1)
LEGION_FORTRAN_OBJS := $(LEGION_FORTRAN_SRC:.f90=.f90.o)
GEN_FORTRAN_OBJS := $(GEN_FORTRAN_SRC:.f90=.f90.o)
FC_FLAGS += -cpp
LD_FLAGS += -lgfortran
else
LEGION_FORTRAN_OBJS :=
GEN_FORTRAN_OBJS :=
endif

# Provide build rules unless the user asks us not to
ifndef NO_BUILD_RULES
# Provide an all unless the user asks us not to
ifndef NO_BUILD_ALL
.PHONY: all
all: $(OUTFILE)
endif
# Provide support for installing legion with the make build system
.PHONY: install
ifdef PREFIX
INSTALL_BIN_FILES ?=
INSTALL_INC_FILES ?=
INSTALL_LIB_FILES ?=
install: $(OUTFILE)
	@echo "Installing into $(PREFIX)..."
	@mkdir -p $(PREFIX)/bin
	@mkdir -p $(PREFIX)/include/realm
	@mkdir -p $(PREFIX)/include/realm/hdf5
	@mkdir -p $(PREFIX)/include/realm/llvm
	@mkdir -p $(PREFIX)/include/realm/python
	@mkdir -p $(PREFIX)/include/legion
	@mkdir -p $(PREFIX)/include/mappers
	@mkdir -p $(PREFIX)/include/mathtypes
	@mkdir -p $(PREFIX)/lib
	@cp $(OUTFILE) $(PREFIX)/bin/$(OUTFILE)
	@$(foreach file,$(INSTALL_BIN_FILES),cp $(file) $(PREFIX)/bin/$(file);)
	@cp realm_defines.h $(PREFIX)/include/realm_defines.h
	@cp legion_defines.h $(PREFIX)/include/legion_defines.h
	@$(foreach file,$(INSTALL_HEADERS),cp $(LG_RT_DIR)/$(file) $(PREFIX)/include/$(file);)
	@$(foreach file,$(INSTALL_INC_FILES),cp $(file) $(PREFIX)/include/$(file);)
	@cp $(SLIB_REALM) $(PREFIX)/lib/$(SLIB_REALM)
	@cp $(SLIB_LEGION) $(PREFIX)/lib/$(SLIB_LEGION)
	@$(foreach file,$(INSTALL_LIB_FILES),cp $(file) $(PREFIX)/lib/$(file);)
	@echo "Installation complete"
else
install:
	$(error Must specify PREFIX for installation)
endif

# If we're using CUDA we have to link with nvcc
$(OUTFILE) : $(GEN_OBJS) $(GEN_GPU_OBJS) $(SLIB_LEGION) $(SLIB_REALM) $(GEN_FORTRAN_OBJS)
	@echo "---> Linking objects into one binary: $(OUTFILE)"
	$(CXX) -o $(OUTFILE) $(GEN_OBJS) $(GEN_GPU_OBJS) $(GEN_FORTRAN_OBJS) $(LD_FLAGS) $(LEGION_LIBS) $(LEGION_LD_FLAGS) $(GASNET_FLAGS)

ifeq ($(strip $(SHARED_OBJECTS)),0)
$(SLIB_LEGION) : $(LEGION_OBJS) $(LEGION_FORTRAN_OBJS) $(LEGION_INST_OBJS) $(MAPPER_OBJS) $(LEGION_GPU_OBJS)
	rm -f $@
	$(AR) rcs $@ $^

$(SLIB_REALM) : $(REALM_OBJS) $(REALM_INST_OBJS)
	rm -f $@
	$(AR) rcs $@ $^
else
$(SLIB_LEGION) : $(LEGION_OBJS) $(LEGION_FORTRAN_OBJS) $(LEGION_INST_OBJS) $(MAPPER_OBJS) $(LEGION_GPU_OBJS) $(SLIB_REALM)
	rm -f $@
	$(CXX) $(SO_FLAGS) -o $@ $(LEGION_OBJS) $(LEGION_FORTRAN_OBJS) $(LEGION_INST_OBJS) $(MAPPER_OBJS) $(LEGION_GPU_OBJS) $(SLIB_LEGION_DEPS)

$(SLIB_REALM) : $(REALM_OBJS) $(REALM_INST_OBJS)
	rm -f $@
	$(CXX) $(SO_FLAGS) -o $@ $^ $(SLIB_REALM_DEPS)
endif

$(GEN_OBJS) : %.cc.o : %.cc $(LEGION_DEFINES_HEADER) $(REALM_DEFINES_HEADER)
	$(CXX) -o $@ -c $< $(CC_FLAGS) $(INC_FLAGS) $(OMP_FLAGS)

$(ASM_OBJS) : %.S.o : %.S
	$(CXX) -o $@ -c $< $(CC_FLAGS) $(INC_FLAGS)

# special rules for per-dimension deppart source files
#  (hopefully making the path explicit doesn't break things too badly...)
$(LG_RT_DIR)/realm/deppart/image_%.cc.o : $(LG_RT_DIR)/realm/deppart/image_tmpl.cc $(LG_RT_DIR)/realm/deppart/image.cc $(LEGION_DEFINES_HEADER) $(REALM_DEFINES_HEADER)
	$(CXX) -o $@ -c $< $(CC_FLAGS) $(INC_FLAGS) -DINST_N1=$(word 1,$(subst _, ,$*)) -DINST_N2=$(word 2,$(subst _, ,$*)) $(REALM_DEFCHECK)

$(LG_RT_DIR)/realm/deppart/preimage_%.cc.o : $(LG_RT_DIR)/realm/deppart/preimage_tmpl.cc $(LG_RT_DIR)/realm/deppart/preimage.cc $(LEGION_DEFINES_HEADER) $(REALM_DEFINES_HEADER)
	$(CXX) -o $@ -c $< $(CC_FLAGS) $(INC_FLAGS) -DINST_N1=$(word 1,$(subst _, ,$*)) -DINST_N2=$(word 2,$(subst _, ,$*)) $(REALM_DEFCHECK)

$(LG_RT_DIR)/realm/deppart/byfield_%.cc.o : $(LG_RT_DIR)/realm/deppart/byfield_tmpl.cc $(LG_RT_DIR)/realm/deppart/byfield.cc $(LEGION_DEFINES_HEADER) $(REALM_DEFINES_HEADER)
	$(CXX) -o $@ -c $< $(CC_FLAGS) $(INC_FLAGS) -DINST_N1=$(word 1,$(subst _, ,$*)) -DINST_N2=$(word 2,$(subst _, ,$*)) $(REALM_DEFCHECK)

$(REALM_OBJS) : %.cc.o : %.cc $(LEGION_DEFINES_HEADER) $(REALM_DEFINES_HEADER)
	$(CXX) -o $@ -c $< $(CC_FLAGS) $(INC_FLAGS) $(REALM_DEFCHECK)

$(LG_RT_DIR)/legion/region_tree_%.cc.o : $(LG_RT_DIR)/legion/region_tree_tmpl.cc $(LEGION_DEFINES_HEADER) $(REALM_DEFINES_HEADER)
	$(CXX) -o $@ -c $< $(CC_FLAGS) $(INC_FLAGS) -DINST_N1=$(word 1,$(subst _, ,$*)) $(patsubst %,-DINST_N2=%,$(word 2,$(subst _, ,$*))) $(LEGION_DEFCHECK)

$(LEGION_OBJS) : %.cc.o : %.cc $(LEGION_DEFINES_HEADER) $(REALM_DEFINES_HEADER)
	$(CXX) -o $@ -c $< $(CC_FLAGS) $(INC_FLAGS) $(LEGION_DEFCHECK)

$(MAPPER_OBJS) : %.cc.o : %.cc $(LEGION_DEFINES_HEADER) $(REALM_DEFINES_HEADER)
	$(CXX) -o $@ -c $< $(CC_FLAGS) $(INC_FLAGS)

$(GEN_GPU_OBJS) : %.cu.o : %.cu $(LEGION_DEFINES_HEADER) $(REALM_DEFINES_HEADER)
	$(NVCC) -o $@ -c $< $(NVCC_FLAGS) $(INC_FLAGS)

$(LEGION_GPU_OBJS): %.cu.o : %.cu $(LEGION_DEFINES_HEADER) $(REALM_DEFINES_HEADER)
	$(NVCC) -o $@ -c $< $(NVCC_FLAGS) $(INC_FLAGS)

ifeq ($(strip $(LEGION_USE_FORTRAN)),1)
$(LG_RT_DIR)/legion/legion_f_types.f90.o : $(LG_RT_DIR)/legion/legion_f_types.f90 $(LEGION_DEFINES_HEADER) $(REALM_DEFINES_HEADER)
	$(FC) -J$(LG_RT_DIR) -o $@ -c $< $(FC_FLAGS) $(INC_FLAGS)

$(LG_RT_DIR)/legion/legion_f_c_interface.f90.o : $(LG_RT_DIR)/legion/legion_f_c_interface.f90 $(LG_RT_DIR)/legion/legion_f_types.f90.o $(LEGION_DEFINES_HEADER) $(REALM_DEFINES_HEADER)
	$(FC) -J$(LG_RT_DIR) -o $@ -c $< $(FC_FLAGS) $(INC_FLAGS)

$(LG_RT_DIR)/legion/legion_f.f90.o : $(LG_RT_DIR)/legion/legion_f.f90 $(LG_RT_DIR)/legion/legion_f_c_interface.f90.o $(LG_RT_DIR)/legion/legion_f_types.f90.o $(LEGION_DEFINES_HEADER) $(REALM_DEFINES_HEADER)
	$(FC) -J$(LG_RT_DIR) -o $@ -c $< $(FC_FLAGS) $(INC_FLAGS)

$(GEN_FORTRAN_OBJS) : %.f90.o : %.f90 $(LEGION_FORTRAN_OBJS) $(LEGION_DEFINES_HEADER) $(REALM_DEFINES_HEADER)
	$(FC) -o $@ -c $< $(FC_FLAGS) $(INC_FLAGS)
endif

# disable gmake's default rule for building % from %.o
% : %.o

clean::
	$(RM) -f $(OUTFILE) $(SLIB_LEGION) $(SLIB_REALM) $(GEN_OBJS) $(GEN_GPU_OBJS) $(REALM_OBJS) $(REALM_INST_OBJS) $(LEGION_OBJS) $(LEGION_INST_OBJS) $(LEGION_GPU_OBJS) $(MAPPER_OBJS) $(ASM_OBJS) $(LEGION_FORTRAN_OBJS) $(LG_RT_DIR)/*mod $(GEN_FORTRAN_OBJS) *.mod $(LEGION_DEFINES_HEADER) $(REALM_DEFINES_HEADER)

ifeq ($(strip $(USE_LLVM)),1)
llvmjit_internal.cc.o : CC_FLAGS += $(LLVM_CXXFLAGS)
%/llvmjit_internal.cc.o : CC_FLAGS += $(LLVM_CXXFLAGS)
endif

endif # NO_BUILD_RULES

# you get these build rules even with NO_BUILD_RULES=1

# by default, we'll always check to see if the defines headers need to be
#  overwritten due to changes in compile settings (from makefile or command line)
# set CHECK_DEFINES_HEADER_CONTENT=0 if you want to only rebuild when makefiles
#  change
ifneq ($(strip $(CHECK_DEFINES_HEADER_CONTENT)),0)
.PHONY: FORCE_DEFINES_HEADERS
DEFINES_HEADERS_DEPENDENCY = FORCE_DEFINES_HEADERS
GENERATE_DEFINES_FLAGS = -c
else
DEFINES_HEADERS_DEPENDENCY = $(MAKEFILE_LIST)
endif
$(LEGION_DEFINES_HEADER) : $(DEFINES_HEADERS_DEPENDENCY)
	$(PYTHON) $(LG_RT_DIR)/../tools/generate_defines.py $(LEGION_CC_FLAGS) $(GENERATE_DEFINES_FLAGS) -i $(LG_RT_DIR)/../cmake/legion_defines.h.in -o $@

$(REALM_DEFINES_HEADER) : $(DEFINES_HEADERS_DEPENDENCY)
	$(PYTHON) $(LG_RT_DIR)/../tools/generate_defines.py $(REALM_CC_FLAGS) $(GENERATE_DEFINES_FLAGS) -i $(LG_RT_DIR)/../cmake/realm_defines.h.in -o $@
