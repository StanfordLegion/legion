# Copyright 2022 Stanford University, NVIDIA Corporation
# Copyright 2022 Los Alamos National Laboratory
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
NVCC_FLAGS ?=
HIPCC_FLAGS ?=

# These flags are NOT passed on the command line, but are used to
# generate the public-facing legion/realm_defines.h files.
# (Additional flags will be picked up from environment variables of
# the same names.)
LEGION_CC_FLAGS ?=
REALM_CC_FLAGS ?=

# Map some common GNU variable names into our variables
CPPFLAGS ?=
CFLAGS ?=
CXXFLAGS ?=
LDLIBS ?=
LDFLAGS ?=
CC_FLAGS += $(CXXFLAGS) $(CPPFLAGS)
FC_FLAGS += $(FFLAGS)
SO_FLAGS += $(LDLIBS)
LD_FLAGS += $(LDFLAGS)

# the Legion/Realm version string is set by the first of these that works:
# 1) a defined value for the make REALM_VERSION variable
# 2) the output of `git describe`, if successful
# 3) the contents of 'VERSION' (at the root of the source tree), if available
# 4) "unknown", if all else fails
ifndef REALM_VERSION
  REALM_VERSION := $(shell git -C $(LG_RT_DIR) describe --dirty --match legion\* 2> /dev/null)
  ifneq ($(REALM_VERSION),)
    $(info Version string from git: ${REALM_VERSION})
  else
    REALM_VERSION := $(shell cat $(LG_RT_DIR)/../VERSION 2> /dev/null)
    ifneq ($(REALM_VERSION),)
      $(info Version string from VERSION file: ${REALM_VERSION})
    else
      REALM_VERSION := unknown
      $(warning Could not determine version string - using 'unknown')
    endif
  endif
endif
REALM_CC_FLAGS += -DREALM_VERSION='"${REALM_VERSION}"'
LEGION_CC_FLAGS += -DLEGION_VERSION='"${REALM_VERSION}"'

USE_OPENMP ?= 0
ifeq ($(shell uname -s),Darwin)
  DARWIN = 1
  CC_FLAGS += -DDARWIN
  FC_FLAGS += -DDARWIN
  # Detect if we're using Apple Clang or Normal Clang
  ifeq ($(findstring Apple,$(shell $(CXX) --version)),Apple)
    APPLECLANG = 1
    REALM_LIMIT_SYMBOL_VISIBILITY=0
    $(warning "Apple Clang is a weird compiler and untested by Legion CI. Tread lightly...")
  else
    APPLECLANG = 0
  endif
  ifeq ($(strip $(USE_OPENMP)),1)
    $(warning "Some versions of Clang on Mac OSX do not support OpenMP")
  endif
else
  APPLECLANG = 0
endif

ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined, aborting build)
endif

# generate libraries for Legion and Realm
SHARED_OBJECTS ?= 0
ifeq ($(strip $(SHARED_OBJECTS)),0)
SLIB_LEGION     := liblegion.a
SLIB_REALM      := librealm.a
OUTFILE		?= liblegion.a
ifeq ($(strip $(OUTFILE)),)
  OUTFILE	:= liblegion.a
endif
else
CC_FLAGS	+= -fPIC
FC_FLAGS	+= -fPIC
NVCC_FLAGS	+= -Xcompiler -fPIC
HIPCC_FLAGS     += -fPIC
ifeq ($(shell uname -s),Darwin)
SLIB_LEGION     := liblegion.dylib
SLIB_REALM      := librealm.dylib
OUTFILE		?= liblegion.dylib
ifeq ($(strip $(OUTFILE)),)
  OUTFILE	:= liblegion.dylib
endif
else
SLIB_LEGION     := liblegion.so
SLIB_REALM      := librealm.so
OUTFILE		?= liblegion.so
ifeq ($(strip $(OUTFILE)),)
  OUTFILE	:= liblegion.so
endif
endif
# shared libraries can link against other shared libraries
SLIB_LEGION_DEPS = -L. -lrealm
SLIB_REALM_DEPS  =
ifeq ($(strip $(DARWIN)),1)
SO_FLAGS += -dynamiclib -single_module -undefined dynamic_lookup -fPIC
LD_FLAGS += -Wl,-all_load
else
SO_FLAGS += -shared
endif
endif
LEGION_LIBS     := -L. -llegion -lrealm

# if requested, realm hides internal classes/methods from shared library exports
REALM_LIMIT_SYMBOL_VISIBILITY ?= 1
ifeq ($(strip $(REALM_LIMIT_SYMBOL_VISIBILITY)),1)
  REALM_SYMBOL_VISIBILITY = -fvisibility=hidden -fvisibility-inlines-hidden
  REALM_CC_FLAGS += -DREALM_LIMIT_SYMBOL_VISIBILITY
endif

# generate header files for public-facing defines
DEFINE_HEADERS_DIR ?= .
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

# Backwards-compatibility for GASNet builds
# GASNET_ROOT is a synonym for GASNET
ifdef GASNET_ROOT
  GASNET ?= $(GASNET_ROOT)
endif
# USE_GASNET=1 will set REALM_NETWORKS=gasnet1
USE_GASNET ?= 0
ifeq ($(strip $(USE_GASNET)),1)
  REALM_NETWORKS ?= gasnet1
endif
# Turn on network support if REALM_NETWORKS is not empty
REALM_NETWORKS ?=
ifndef USE_NETWORK
  ifeq ($(strip $(REALM_NETWORKS)),)
    USE_NETWORK := 0
  else
    USE_NETWORK := 1
  endif
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

USE_PGI ?= 0
# Check to see if this is the PGI compiler
# in whch case we need to use different flags in some cases
ifeq ($(strip $(USE_PGI)),0)
ifeq ($(findstring nvc++,$(shell $(CXX) --version)),nvc++)
  USE_PGI = 1
endif
endif

# machine architecture (generally "native" unless cross-compiling)
MARCH ?= native

ifneq (${MARCH},)
  # Summit/Summitdev are strange and want to have this specified via -mcpu
  # instead of -march. Unclear if this is true in general for PPC.
  ifeq ($(findstring ppc64le,$(shell uname -p)),ppc64le)
    ifeq ($(strip $(USE_PGI)),0)
      CC_FLAGS += -mcpu=${MARCH} -maltivec -mabi=altivec -mvsx
      FC_FLAGS += -mcpu=${MARCH} -maltivec -mabi=altivec -mvsx
    else
      $(error PGI compilers do not currently support the PowerPC architecture)
    endif
  else ifeq ($(strip $(USE_PGI)),1)
    CC_FLAGS += -tp=${MARCH}
    FC_FLAGS += -tp=${MARCH}
  else ifeq ($(strip $(APPLECLANG)),1)
    # For reasons passing understanding different versions of Apple clang support different arch flags
    ifeq ($(shell $(CXX) -x c++ -Werror -march=${MARCH} -c /dev/null -o /dev/null 2> /dev/null; echo $$?),0)
      CC_FLAGS += -march=${MARCH}
      FC_FLAGS += -march=${MARCH}
    else
      CC_FLAGS += -mcpu=${MARCH}
      FC_FLAGS += -mcpu=${MARCH}
    endif
  else
    CC_FLAGS += -march=${MARCH}
    FC_FLAGS += -march=${MARCH}
  endif
endif

INC_FLAGS	+= -I$(DEFINE_HEADERS_DIR) -I$(LG_RT_DIR) -I$(LG_RT_DIR)/mappers
# support libraries are OS specific unfortunately
ifeq ($(shell uname -s),Linux)
LEGION_LD_FLAGS	+= -lrt -lpthread -latomic
SLIB_REALM_DEPS += -lrt -lpthread -ldl
endif
ifeq ($(strip $(DARWIN)),1)
LEGION_LD_FLAGS	+= -lpthread
SLIB_REALM_DEPS += -lpthread
endif
ifeq ($(shell uname -s),FreeBSD)
LEGION_LD_FLAGS	+= -lexecinfo -lpthread -latomic
SLIB_REALM_DEPS += -lpthread
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
LEGION_CC_FLAGS += -DLEGION_USE_LIBDL
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
  REALM_CC_FLAGS += -DREALM_USE_LLVM

  # NOTE: do not use these for all source files - just the ones that include llvm include files
  LLVM_CXXFLAGS ?= -std=c++11 -I$(shell $(LLVM_CONFIG) --includedir)

  # realm can be configured to allow LLVM library linkage to be optional
  #  (i.e. a per-application choice)
  LLVM_LIBS_OPTIONAL ?= 0
  ifeq ($(strip $(LLVM_LIBS_OPTIONAL)),1)
    REALM_CC_FLAGS += -DREALM_ALLOW_MISSING_LLVM_LIBS
  else
    LLVM_LIBS += $(shell $(LLVM_CONFIG) --ldflags --libs irreader mcjit x86)
    LEGION_LD_FLAGS += $(LLVM_LIBS)
  endif

  # llvm-config --system-libs gives you all the libraries you might need for anything,
  #  which includes things we don't need, and might not be installed
  # by default, filter out libedit
  LLVM_SYSTEM_LIBS ?= $(filter-out -ledit,$(shell $(LLVM_CONFIG) --system-libs))
  LEGION_LD_FLAGS += $(LLVM_SYSTEM_LIBS)
endif

ifeq ($(strip $(USE_OPENMP)),1)
  REALM_CC_FLAGS += -DREALM_USE_OPENMP
  # Add the -fopenmp flag for Linux, but not for Mac as clang doesn't need it
  #ifneq ($(strip $(DARWIN)),1)
  CC_FLAGS += -fopenmp
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
      PYTHON_EXE := $(shell which python3 python | head -1)
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

# General HIP variables
ifeq ($(strip $(USE_HIP)),1)
  HIP_TARGET ?= ROCM
  USE_GPU_REDUCTIONS ?= 1
  ifndef HIP_PATH
    $(error HIP_PATH variable is not defined, aborting build)
  endif
  ifeq ($(strip $(HIP_TARGET)),ROCM)
    #HIP on AMD
    ifeq ($(strip $(USE_COMPLEX)),1)
      ifndef THRUST_PATH
        $(error THRUST_PATH variable is not defined, aborting build)
      endif
      # Please download the thrust from https://github.com/ROCmSoftwarePlatform/Thrust
      # We need to put thrust inc ahead of HIP_PATH because the thrust comes with hip is broken
      INC_FLAGS += -I$(THRUST_PATH)
    endif
    HIPCC	        ?= $(HIP_PATH)/bin/hipcc
    # Latter is preferred, former is for backwards compatability
    REALM_CC_FLAGS  += -DREALM_USE_HIP
    LEGION_CC_FLAGS += -DLEGION_USE_HIP
    CC_FLAGS        += -D__HIP_PLATFORM_AMD__
    HIPCC_FLAGS     += -fno-strict-aliasing
    INC_FLAGS       += -I$(HIP_PATH)/include -I$(HIP_PATH)/../include
    ifeq ($(strip $(DEBUG)),1)
      HIPCC_FLAGS	+= -g
    else
      HIPCC_FLAGS	+= -O2
    endif
    ifneq ($(strip $(HIP_ARCH)),)
      HIPCC_FLAGS	+= --amdgpu-target=$(HIP_ARCH)
    endif
    LEGION_LD_FLAGS	+= -lm -L$(HIP_PATH)/lib -lamdhip64
  else ifeq ($(strip $(HIP_TARGET)),CUDA)
    # HIP on CUDA
    ifndef CUDA_PATH
      $(error CUDA_PATH variable is not defined, aborting build)
    endif
    HIPCC ?= $(CUDA_PATH)/bin/nvcc
    # Latter is preferred, former is for backwards compatability
    REALM_CC_FLAGS  += -DREALM_USE_HIP
    LEGION_CC_FLAGS += -DLEGION_USE_HIP
    CC_FLAGS        += -D__HIP_PLATFORM_NVIDIA__
    HIPCC_FLAGS     += -D__HIP_PLATFORM_NVIDIA__
    INC_FLAGS       += -I$(CUDA_PATH)/include -I$(HIP_PATH)/include  -I$(HIP_PATH)/../include
    ifeq ($(strip $(DEBUG)),1)
      HIPCC_FLAGS	+= -g -O0
    else
      HIPCC_FLAGS	+= -O2
    endif
    LEGION_LD_FLAGS	+= -L$(CUDA_PATH)/lib64/stubs -lcuda -L$(CUDA_PATH)/lib64 -lcudart
  endif

  USE_HIP_HIJACK ?= 1
  ifeq ($(strip $(USE_HIP_HIJACK)),1)
    REALM_CC_FLAGS        += -DREALM_USE_HIP_HIJACK
  endif
endif

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
# If CUDA compiler is nvcc then set the host compiler
ifeq ($(findstring nvcc,$(NVCC)),nvcc)
CUDAHOSTCXX	?= $(CXX)
NVCC_FLAGS	+= -ccbin $(CUDAHOSTCXX)
endif
REALM_CC_FLAGS        += -DREALM_USE_CUDA
LEGION_CC_FLAGS       += -DLEGION_USE_CUDA
# provide this for backward-compatibility in applications
CC_FLAGS              += -DUSE_CUDA
FC_FLAGS	      += -DUSE_CUDA
REALM_USE_CUDART_HIJACK ?= 1
# We don't support the hijack for nvc++
ifeq ($(findstring nvc++,$(shell $(NVCC) --version)),nvc++)
REALM_USE_CUDART_HIJACK := 1
endif
# Have this for backwards compatibility
ifdef USE_CUDART_HIJACK
REALM_USE_CUDART_HIJACK = $(USE_CUDART_HIJACK)
endif
ifeq ($(strip $(REALM_USE_CUDART_HIJACK)),1)
REALM_CC_FLAGS        += -DREALM_USE_CUDART_HIJACK
endif
INC_FLAGS	+= -I$(CUDA)/include
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
SLIB_REALM_DEPS	+= -L$(CUDA)/lib -lcudart -lcuda
endif
else
ifeq ($(strip $(REALM_USE_CUDART_HIJACK)),1)
LEGION_LD_FLAGS	+= -L$(CUDA)/lib64 -L$(CUDA)/lib64/stubs -lcuda -Xlinker -rpath=$(CUDA)/lib64
SLIB_LEGION_DEPS += -L$(CUDA)/lib64 -L$(CUDA)/lib64/stubs -lcuda
SLIB_REALM_DEPS += -L$(CUDA)/lib64 -L$(CUDA)/lib64/stubs -lcuda
else
LEGION_LD_FLAGS	+= -L$(CUDA)/lib64 -L$(CUDA)/lib64/stubs -lcudart -lcuda -Xlinker -rpath=$(CUDA)/lib64
SLIB_LEGION_DEPS += -L$(CUDA)/lib64 -L$(CUDA)/lib64/stubs -lcudart -lcuda
SLIB_REALM_DEPS += -L$(CUDA)/lib64 -L$(CUDA)/lib64/stubs -lcudart -lcuda
endif
endif
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
ifeq ($(strip $(GPU_ARCH)),ampere)
override GPU_ARCH = 80
NVCC_FLAGS	+= -DAMPERE_ARCH
endif

ifeq ($(strip $(GPU_ARCH)),auto)
  # detect based on what nvcc supports
  ALL_ARCHES = 20 30 32 35 37 50 52 53 60 61 62 70 72 75 80
  override GPU_ARCH = $(shell for X in $(ALL_ARCHES) ; do \
    $(NVCC) -gencode arch=compute_$$X,code=sm_$$X -cuda -x c++ /dev/null -o /dev/null 2> /dev/null && echo $$X; \
  done)
endif

# finally, convert space-or-comma separated list of architectures (e.g. 35,50)
#  into nvcc -gencode arguments
ifeq ($(findstring nvc++,$(shell $(NVCC) --version)),nvc++)
NVCC_FLAGS += $(foreach X,$(subst $(COMMA), ,$(GPU_ARCH)),-gpu=cc$(X))
else
COMMA=,
NVCC_FLAGS += $(foreach X,$(subst $(COMMA), ,$(GPU_ARCH)),-gencode arch=compute_$(X)$(COMMA)code=sm_$(X))
endif

NVCC_FLAGS += -Xcudafe --diag_suppress=boolean_controlling_expr_is_constant
endif

# Realm uses GASNet if requested (detect both gasnet1 and gasnetex here)
ifeq ($(strip $(USE_NETWORK)),1)
ifeq ($(findstring gasnet,$(REALM_NETWORKS)),gasnet)
  ifeq ($(strip $(REALM_NETWORKS)),gasnetex)
    REALM_CC_FLAGS	+= -DREALM_USE_GASNETEX
  else
    ifeq ($(strip $(REALM_NETWORKS)),gasnet1)
      REALM_CC_FLAGS	+= -DREALM_USE_GASNET1
    else
      $(error Illegal value for REALM_NETWORKS: $(REALM_NETWORKS), needs to be either gasnet1, gasnetex, or mpi)
    endif
  endif
  ifeq ($(GASNET),)
    $(error GASNET variable is not defined, aborting build)
  endif
  # newer versions of gasnet seem to need this
  CC_FLAGS	+= -DGASNETI_BUG1389_WORKAROUND=1
  # Detect conduit, if requested
  CONDUIT ?= auto
  ifeq ($(strip $(CONDUIT)),auto)
    GASNET_PREFERRED_CONDUITS = ibv aries gemini pami mpi udp ofi psm mxm portals4 smp ucx
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
  # Suck in some GASNET variables that they define
  include $(GASNET)/include/$(strip $(CONDUIT))-conduit/$(strip $(CONDUIT))-par.mak
  INC_FLAGS += $(GASNET_INCLUDES)
  # I don't like some of the flags gasnet includes here like _GNU_SOURCE=1 in lot of cases which makes
  # this inherently non-portable code. We use many more compilers than just GNU
  #CC_FLAGS += $(GASNET_DEFINES)
  #LD_FLAGS += $(GASNET_LDFLAGS)
  REALM_CC_FLAGS += -DGASNET_CONDUIT_$(shell echo '$(CONDUIT)' | tr '[:lower:]' '[:upper:]')
  LEGION_LD_FLAGS += $(GASNET_LIBS)
  # Check if GASNet needs MPI for interop support
  ifeq ($(strip $(GASNET_LD_REQUIRES_MPI)),1)
    USE_MPI = 1
  endif 
else # Not GASNet network
# Realm uses MPI if requested
ifeq ($(strip $(REALM_NETWORKS)),mpi)
    REALM_CC_FLAGS        += -DREALM_USE_MPI
    USE_MPI = 1
else
  $(error Illegal value for REALM_NETWORKS: $(REALM_NETWORKS), needs to be either gasnet1, gasnetex, or mpi)
endif # Test for MPI
endif # Test for GASNet
endif # Only turn on networks if USE_NETWORK=1

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

# use mpi{cc,cxx,f90} compiler wrappers if USE_MPI=1 and we're not on a Cray system
ifeq ($(strip $(USE_MPI)),1)
  ifeq (${CRAYPE_VERSION},)
    # OpenMPI check
    ifneq ($(strip $(shell __INTEL_POST_CFLAGS+=' -we10006' $(CC) -showme:compile 2>&1 > /dev/null; echo $$?)),0)
      # MPICH check
      ifneq ($(strip $(shell __INTEL_POST_CFLAGS+=' -we10006' $(CC) -show 2>&1 > /dev/null; echo $$?)),0)
	export OMPI_CC  	:= $(CC)
	export MPICH_CC  	:= $(CC)
	CC			:= mpicc
      endif
    endif
    # OpenMPI check
    ifneq ($(strip $(shell __INTEL_POST_CFLAGS+=' -we10006' $(CXX) -showme:compile 2>&1 > /dev/null; echo $$?)),0)
      # MPICH check
      ifneq ($(strip $(shell __INTEL_POST_CFLAGS+=' -we10006' $(CXX) -show 2>&1 > /dev/null; echo $$?)),0)
	export OMPI_CXX 	:= $(CXX)
	export MPICH_CXX 	:= $(CXX)
	CXX			:= mpicxx
      endif
    endif
    # OpenMPI check
    ifneq ($(strip $(shell __INTEL_POST_CFLAGS+=' -we10006' $(FC) -showme:compile 2>&1 > /dev/null; echo $$?)),0) 
      # MPICH check
      ifneq ($(strip $(shell __INTEL_POST_CFLAGS+=' -we10006' $(FC) -show 2>&1 > /dev/null; echo $$?)),0)
	export OMPI_FC  	:= $(FC)
	export MPICH_FC  	:= $(FC)
	FC			:= mpif90
      endif
    endif
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
  SLIB_LEGION_DEPS += -l$(ZLIB_LIBNAME)
endif


ifeq ($(strip $(DEBUG)),1)
  ifeq ($(strip $(DARWIN)),1)
    CFLAGS	+= -O0 -glldb
    CC_FLAGS	+= -O0 -glldb
    FC_FLAGS	+= -O0 -glldb
  else ifeq ($(strip $(USE_PGI)),1)
    CFLAGS	+= -O0 -g
    CC_FLAGS	+= -O0 -g # --display_error_number
    FC_FLAGS	+= -O0 -g
  else
    CFLAGS	+= -0O -ggdb
    CC_FLAGS	+= -O0 -ggdb #-ggdb -Wall
    FC_FLAGS	+= -O0 -ggdb
  endif
  REALM_CC_FLAGS	+= -DDEBUG_REALM
  LEGION_CC_FLAGS	+= -DDEBUG_LEGION
else
  CFLAGS	+= -O2
  CC_FLAGS	+= -O2 #-ggdb
  FC_FLAGS	+= -O2
endif

BOUNDS_CHECKS ?= 0
LEGION_BOUNDS_CHECKS ?= 0
ifeq ($(strip $(BOUNDS_CHECKS)),1)
LEGION_CC_FLAGS	+= -DLEGION_BOUNDS_CHECKS
else ifeq ($(strip $(LEGION_BOUNDS_CHECKS)),1)
LEGION_CC_FLAGS	+= -DLEGION_BOUNDS_CHECKS
endif

PRIVILEGE_CHECKS ?= 0
LEGION_PRIVILEGE_CHECKS ?= 0
ifeq ($(strip $(PRIVILEGE_CHECKS)),1)
LEGION_CC_FLAGS	+= -DLEGION_PRIVILEGE_CHECKS
else ifeq ($(strip $(LEGION_PRIVILEGE_CHECKS)),1)
LEGION_CC_FLAGS	+= -DLEGION_PRIVILEGE_CHECKS
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
CC_FLAGS        += -Wall
FC_FLAGS	+= -Wall
ifeq ($(strip $(WARN_AS_ERROR)),1)
CC_FLAGS        += -Werror
FC_FLAGS	+= -Werror
endif

# Check for a minimum C++ version and if none is specified then set it to c++11
ifneq ($(findstring -std=c++,$(CC_FLAGS)),-std=c++)
ifeq ($(shell $(CXX) -x c++ -std=c++11 -c /dev/null -o /dev/null 2> /dev/null; echo $$?),0)
CC_FLAGS += -std=c++11
else ifeq ($(findstring nvc++,$(CXX)),nvc++)
# nvc++ is dumb and will give you an error if you try to overwrite the input
# file with the output file and so errors at our test above, we'll just assume
# that all versions of nvc++ will support c++11 for now
CC_FLAGS += -std=c++11
else
$(error Legion requires a C++ compiler that supports at least C++11)
endif
endif


# if requested, add --defcheck flags to the compile line so that the
#  cxx_defcheck wrapper can verify that source files include the configuration
#  headers properly
USE_DEFCHECK ?= 0
ifeq ($(strip ${USE_DEFCHECK}),1)
  LEGION_DEFCHECK = --defcheck legion_defines.h
  REALM_DEFCHECK = --defcheck realm_defines.h
endif

CC_SRC		?=
CXX_SRC		?=
# Backwards compatibility for older makefiles
GEN_SRC		?=
CXX_SRC		+= $(GEN_SRC)
FORT_SRC	?=
CUDA_SRC	?=
HIP_SRC         ?=
# Backwards compatibility for older makefiles
GEN_GPU_SRC	?= 
CUDA_SRC	+= $(GEN_GPU_SRC)
HIP_SRC         += $(GEN_GPU_SRC)
REALM_SRC	?=
LEGION_SRC	?=
LEGION_CUDA_SRC	?=
LEGION_HIP_SRC  ?=
MAPPER_SRC	?=

# Set the source files
REALM_SRC 	+= $(LG_RT_DIR)/realm/runtime_impl.cc \
		   $(LG_RT_DIR)/realm/bgwork.cc \
	           $(LG_RT_DIR)/realm/transfer/transfer.cc \
	           $(LG_RT_DIR)/realm/transfer/channel.cc \
	           $(LG_RT_DIR)/realm/transfer/channel_disk.cc \
	           $(LG_RT_DIR)/realm/transfer/lowlevel_dma.cc \
	           $(LG_RT_DIR)/realm/transfer/ib_memory.cc \
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
ifeq ($(strip $(USE_NETWORK)),1)
ifeq ($(findstring gasnet1,$(REALM_NETWORKS)),gasnet1)
REALM_SRC 	+= $(LG_RT_DIR)/realm/gasnet1/gasnet1_module.cc \
                   $(LG_RT_DIR)/realm/gasnet1/gasnetmsg.cc
endif
ifeq ($(findstring gasnetex,$(REALM_NETWORKS)),gasnetex)
REALM_SRC 	+= $(LG_RT_DIR)/realm/gasnetex/gasnetex_module.cc \
	           $(LG_RT_DIR)/realm/gasnetex/gasnetex_internal.cc \
	           $(LG_RT_DIR)/realm/gasnetex/gasnetex_handlers.cc
endif
ifeq ($(findstring mpi,$(REALM_NETWORKS)),mpi)
REALM_SRC 	+= $(LG_RT_DIR)/realm/mpi/mpi_module.cc \
                   $(LG_RT_DIR)/realm/mpi/am_mpi.cc
endif
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
REALM_SRC 	+= $(LG_RT_DIR)/realm/cuda/cuda_module.cc \
                   $(LG_RT_DIR)/realm/cuda/cuda_access.cc \
                   $(LG_RT_DIR)/realm/cuda/cuda_internal.cc
ifeq ($(strip $(REALM_USE_CUDART_HIJACK)),1)
REALM_SRC       += $(LG_RT_DIR)/realm/cuda/cudart_hijack.cc
endif
endif
ifeq ($(strip $(USE_HIP)),1)
REALM_SRC 	+= $(LG_RT_DIR)/realm/hip/hip_module.cc \
                   $(LG_RT_DIR)/realm/hip/hip_access.cc \
                   $(LG_RT_DIR)/realm/hip/hip_internal.cc
ifeq ($(strip $(USE_HIP_HIJACK)),1)
REALM_SRC       += $(LG_RT_DIR)/realm/hip/hip_hijack.cc
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
		   $(LG_RT_DIR)/realm/timers.cc \
		   $(LG_RT_DIR)/realm/utils.cc

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
LEGION_CUDA_SRC  += $(LG_RT_DIR)/legion/legion_redop.cu
LEGION_HIP_SRC   += $(LG_RT_DIR)/legion/legion_redop.cu
# LEGION_INST_SRC will be compiled {MAX_DIM}^2 times in parallel
LEGION_INST_SRC  += $(LG_RT_DIR)/legion/region_tree_tmpl.cc

LEGION_FORT_SRC  += $(LG_RT_DIR)/legion/legion_f_types.f90 \
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

ifeq ($(strip $(USE_CUDA)),1)
INSTALL_HEADERS += realm/cuda/cuda_redop.h \
                   realm/cuda/cuda_access.h
endif
ifeq ($(strip $(USE_HIP)),1)
INSTALL_HEADERS += hip_cuda_compat/hip_cuda.h \
                   realm/hip/hip_redop.h
endif
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
INSTALL_HEADERS += realm/llvmjit/llvmjit.h \
		   realm/llvmjit/llvmjit.inl
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

APP_OBJS	:= $(CC_SRC:.c=.c.o)
APP_OBJS	+= $(CXX_SRC:.cc=.cc.o)
APP_OBJS	+= $(ASM_SRC:.S=.S.o)
REALM_OBJS	:= $(REALM_SRC:.cc=.cc.o)
LEGION_OBJS	:= $(LEGION_SRC:.cc=.cc.o)
MAPPER_OBJS	:= $(MAPPER_SRC:.cc=.cc.o)
# Only compile the gpu objects if we need to
ifeq ($(strip $(USE_CUDA)),1)
APP_OBJS	+= $(CUDA_SRC:.cu=.cu.o)
LEGION_OBJS 	+= $(LEGION_CUDA_SRC:.cu=.cu.o)
endif

# Only compile the hip objects if we need to 
ifeq ($(strip $(USE_HIP)),1)
APP_OBJS	+= $(HIP_SRC:.cu=.cu.o)
LEGION_OBJS     += $(LEGION_HIP_SRC:.cu=.cu.o)
endif

USE_FORTRAN ?= 0
LEGION_USE_FORTRAN ?= 0
# For backwards compatibility
GEN_FORTRAN_SRC ?=
FORT_SRC	+= $(GEN_FORTRAN_SRC)
ifeq ($(strip $(USE_FORTRAN)),1)
LEGION_OBJS 	+= $(LEGION_FORT_SRC:.f90=.f90.o)
APP_OBJS 	+= $(FORT_SRC:.f90=.f90.o)
FC_FLAGS 	+= -cpp
LD_FLAGS 	+= -lgfortran
else ifeq ($(strip $(LEGION_USE_FORTRAN)),1)
USE_FORTRAN	:= 1
LEGION_OBJS 	+= $(LEGION_FORT_SRC:.f90=.f90.o)
APP_OBJS	+= $(FORT_SRC:.f90=.f90.o)
FC_FLAGS 	+= -cpp
LD_FLAGS 	+= -lgfortran
endif

# Provide build rules unless the user asks us not to
ifndef NO_BUILD_RULES
# Provide an all unless the user asks us not to
ifndef NO_BUILD_ALL
.PHONY: all
all: $(OUTFILE)
endif
# Provide support for installing legion with the make build system
.PHONY: install COPY_FILES_AFTER_BUILD
ifdef PREFIX
INSTALL_BIN_FILES += $(OUTFILE)
INSTALL_INC_FILES += legion_defines.h realm_defines.h
INSTALL_LIB_FILES += $(SLIB_REALM) $(SLIB_LEGION)
TARGET_HEADERS := $(addprefix $(strip $(PREFIX))/include/,$(INSTALL_HEADERS))
TARGET_BIN_FILES := $(addprefix $(strip $(PREFIX))/bin/,$(INSTALL_BIN_FILES))
TARGET_INC_FILES := $(addprefix $(strip $(PREFIX))/include/,$(INSTALL_INC_FILES))
TARGET_LIB_FILES := $(addprefix $(strip $(PREFIX))/lib/,$(INSTALL_LIB_FILES))
install: $(OUTFILE)
	$(MAKE) COPY_FILES_AFTER_BUILD
COPY_FILES_AFTER_BUILD: $(TARGET_HEADERS) $(TARGET_BIN_FILES) $(TARGET_INC_FILES) $(TARGET_LIB_FILES)
$(TARGET_HEADERS) : $(strip $(PREFIX))/include/% : $(LG_RT_DIR)/%
	mkdir -p $(dir $@)
	cp $< $@
$(TARGET_BIN_FILES) : $(strip $(PREFIX))/bin/% : %
	mkdir -p $(dir $@)
	cp $< $@
$(TARGET_INC_FILES) : $(strip $(PREFIX))/include/% : %
	mkdir -p $(dir $@)
	cp $< $@
$(TARGET_LIB_FILES) : $(strip $(PREFIX))/lib/% : %
	mkdir -p $(dir $@)
	cp $< $@
else
install:
	$(error Must specify PREFIX for installation)
endif

# Include generated dependency files
DEP_FILES += $(APP_OBJS:.o=.d)
DEP_FILES += $(REALM_OBJS:.o=.d)
DEP_FILES += $(REALM_INST_OBJS:.o=.d)
DEP_FILES += $(LEGION_OBJS:.o=.d)
DEP_FILES += $(LEGION_INST_OBJS:.o=.d)
DEP_FILES += $(MAPPER_OBJS:.o=.d)
-include $(DEP_FILES)

$(OUTFILE) : $(APP_OBJS) $(SLIB_LEGION) $(SLIB_REALM)
	@echo "---> Linking objects into one binary: $(OUTFILE)"
	$(CXX) -o $(OUTFILE) $(APP_OBJS) $(LD_FLAGS) $(LEGION_LIBS) $(LEGION_LD_FLAGS)

ifeq ($(strip $(SHARED_OBJECTS)),0)
$(SLIB_LEGION) : $(LEGION_OBJS) $(LEGION_INST_OBJS) $(MAPPER_OBJS)
	rm -f $@
	$(AR) rcs $@ $^

$(SLIB_REALM) : $(REALM_OBJS) $(REALM_INST_OBJS)
	rm -f $@
	$(AR) rcs $@ $^
else
$(SLIB_LEGION) : $(LEGION_OBJS) $(LEGION_INST_OBJS) $(MAPPER_OBJS) $(SLIB_REALM)
	rm -f $@
	$(CXX) $(SO_FLAGS) -o $@ $(LEGION_OBJS) $(LEGION_INST_OBJS) $(MAPPER_OBJS) $(LD_FLAGS) $(SLIB_LEGION_DEPS)

$(SLIB_REALM) : $(REALM_OBJS) $(REALM_INST_OBJS)
	rm -f $@
	$(CXX) $(SO_FLAGS) -o $@ $^ $(LD_FLAGS) $(SLIB_REALM_DEPS)
endif

$(filter %.c.o,$(APP_OBJS)) : %.c.o : %.c $(LEGION_DEFINES_HEADER) $(REALM_DEFINES_HEADER)
	$(CC) -MMD -o $@ -c $< $(CPPFLAGS) $(CFLAGS) $(INC_FLAGS)

$(filter %.cc.o,$(APP_OBJS)) : %.cc.o : %.cc $(LEGION_DEFINES_HEADER) $(REALM_DEFINES_HEADER)
	$(CXX) -MMD -o $@ -c $< $(CC_FLAGS) $(INC_FLAGS)

$(filter %.S.o,$(APP_OBJS)) : %.S.o : %.S
	$(AS) -o $@ -c $< $(CC_FLAGS) $(INC_FLAGS)

# special rules for per-dimension deppart source files
#  (hopefully making the path explicit doesn't break things too badly...)
ifneq ($(USE_PGI),1)
$(LG_RT_DIR)/realm/deppart/image_%.cc.o : $(LG_RT_DIR)/realm/deppart/image_tmpl.cc $(LG_RT_DIR)/realm/deppart/image.cc $(LEGION_DEFINES_HEADER) $(REALM_DEFINES_HEADER)
	$(CXX) -MMD -o $@ -c $< $(CC_FLAGS) $(REALM_SYMBOL_VISIBILITY) $(INC_FLAGS) -DINST_N1=$(word 1,$(subst _, ,$*)) -DINST_N2=$(word 2,$(subst _, ,$*)) $(REALM_DEFCHECK)

$(LG_RT_DIR)/realm/deppart/preimage_%.cc.o : $(LG_RT_DIR)/realm/deppart/preimage_tmpl.cc $(LG_RT_DIR)/realm/deppart/preimage.cc $(LEGION_DEFINES_HEADER) $(REALM_DEFINES_HEADER)
	$(CXX) -MMD -o $@ -c $< $(CC_FLAGS) $(REALM_SYMBOL_VISIBILITY) $(INC_FLAGS) -DINST_N1=$(word 1,$(subst _, ,$*)) -DINST_N2=$(word 2,$(subst _, ,$*)) $(REALM_DEFCHECK)

$(LG_RT_DIR)/realm/deppart/byfield_%.cc.o : $(LG_RT_DIR)/realm/deppart/byfield_tmpl.cc $(LG_RT_DIR)/realm/deppart/byfield.cc $(LEGION_DEFINES_HEADER) $(REALM_DEFINES_HEADER)
	$(CXX) -MMD -o $@ -c $< $(CC_FLAGS) $(REALM_SYMBOL_VISIBILITY) $(INC_FLAGS) -DINST_N1=$(word 1,$(subst _, ,$*)) -DINST_N2=$(word 2,$(subst _, ,$*)) $(REALM_DEFCHECK)
else
# nvc++ names some symbols based on the source filename, so the trick above
#  of compiling multiple things from the same template with different defines
#  causes linker errors - work around by generating a different source file for
#  each case, but don't leave them lying around
$(LG_RT_DIR)/realm/deppart/image_%.cc :
	echo '#define' INST_N1 $(word 1,$(subst _, ,$*)) > $@
	echo '#define' INST_N2 $(word 2,$(subst _, ,$*)) >> $@
	echo '#include' '"image_tmpl.cc"' >> $@

$(LG_RT_DIR)/realm/deppart/preimage_%.cc :
	echo '#define' INST_N1 $(word 1,$(subst _, ,$*)) > $@
	echo '#define' INST_N2 $(word 2,$(subst _, ,$*)) >> $@
	echo '#include' '"preimage_tmpl.cc"' >> $@

$(LG_RT_DIR)/realm/deppart/byfield_%.cc :
	echo '#define' INST_N1 $(word 1,$(subst _, ,$*)) > $@
	echo '#define' INST_N2 $(word 2,$(subst _, ,$*)) >> $@
	echo '#include' '"byfield_tmpl.cc"' >> $@

.INTERMEDIATE: $(REALM_INST_OBJS:.o=)

REALM_OBJS += $(REALM_INST_OBJS)
endif

$(REALM_OBJS) : %.cc.o : %.cc $(LEGION_DEFINES_HEADER) $(REALM_DEFINES_HEADER)
	$(CXX) -MMD -o $@ -c $< $(CC_FLAGS) $(REALM_SYMBOL_VISIBILITY) $(INC_FLAGS) $(REALM_DEFCHECK)

ifneq ($(USE_PGI),1)
$(LG_RT_DIR)/legion/region_tree_%.cc.o : $(LG_RT_DIR)/legion/region_tree_tmpl.cc $(LEGION_DEFINES_HEADER) $(REALM_DEFINES_HEADER)
	$(CXX) -MMD -o $@ -c $< $(CC_FLAGS) $(INC_FLAGS) -DINST_N1=$(word 1,$(subst _, ,$*)) $(patsubst %,-DINST_N2=%,$(word 2,$(subst _, ,$*))) $(LEGION_DEFCHECK)
else
# nvc++ names some symbols based on the source filename, so the trick above
#  of compiling multiple things from the same template with different defines
#  causes linker errors - work around by generating a different source file for
#  each case, but don't leave them lying around
$(LG_RT_DIR)/legion/region_tree_%.cc :
	echo '#define' INST_N1 $(word 1,$(subst _, ,$*)) > $@
	[ -z "$(word 2,$(subst _, ,$*))" ] || echo '#define' INST_N2 $(word 2,$(subst _, ,$*)) >> $@
	echo '#include' '"region_tree_tmpl.cc"' >> $@

.INTERMEDIATE: $(LEGION_INST_OBJS:.o=)

LEGION_OBJS += $(LEGION_INST_OBJS)
endif

$(filter %.cc.o,$(LEGION_OBJS)) : %.cc.o : %.cc $(LEGION_DEFINES_HEADER) $(REALM_DEFINES_HEADER)
	$(CXX) -MMD -o $@ -c $< $(CC_FLAGS) $(INC_FLAGS) $(LEGION_DEFCHECK)

$(MAPPER_OBJS) : %.cc.o : %.cc $(LEGION_DEFINES_HEADER) $(REALM_DEFINES_HEADER)
	$(CXX) -MMD -o $@ -c $< $(CC_FLAGS) $(INC_FLAGS)

# GPU compilation rules; We can't use -MMD for dependency generation because
# it's not supported by old versions of nvcc.

ifeq ($(strip $(USE_HIP)),1)
$(filter %.cu.o,$(APP_OBJS)) : %.cu.o : %.cu $(LEGION_DEFINES_HEADER) $(REALM_DEFINES_HEADER)
	$(HIPCC) -o $<.d -M -MT $@ $< $(HIPCC_FLAGS) $(INC_FLAGS)
	$(HIPCC) -o $@ -c $< $(HIPCC_FLAGS) $(INC_FLAGS)
endif

ifeq ($(strip $(USE_CUDA)),1)
$(filter %.cu.o,$(APP_OBJS)) : %.cu.o : %.cu $(LEGION_DEFINES_HEADER) $(REALM_DEFINES_HEADER)
	$(NVCC) -o $<.d -M -MT $@ $< $(NVCC_FLAGS) $(INC_FLAGS)
	$(NVCC) -o $@ -c $< $(NVCC_FLAGS) $(INC_FLAGS)
endif

ifeq ($(strip $(USE_HIP)),1)
$(filter %.cu.o,$(LEGION_OBJS)): %.cu.o : %.cu $(LEGION_DEFINES_HEADER) $(REALM_DEFINES_HEADER)
	$(HIPCC) -o $<.d -M -MT $@ $< $(HIPCC_FLAGS) $(INC_FLAGS)
	$(HIPCC) -o $@ -c $< $(HIPCC_FLAGS) $(INC_FLAGS)
endif

ifeq ($(strip $(USE_CUDA)),1)
$(filter %.cu.o,$(LEGION_OBJS)): %.cu.o : %.cu $(LEGION_DEFINES_HEADER) $(REALM_DEFINES_HEADER)
	$(NVCC) -o $<.d -M -MT $@ $< $(NVCC_FLAGS) $(INC_FLAGS)
	$(NVCC) -o $@ -c $< $(NVCC_FLAGS) $(INC_FLAGS)
endif

# Special rules for building the legion fortran files because the fortran compiler is dumb
ifeq ($(strip $(USE_FORTRAN)),1)
$(LG_RT_DIR)/legion/legion_f_types.f90.o : $(LG_RT_DIR)/legion/legion_f_types.f90 $(LEGION_DEFINES_HEADER) $(REALM_DEFINES_HEADER)
	$(FC) -J$(LG_RT_DIR) -o $@ -c $< $(FC_FLAGS) $(INC_FLAGS)

$(LG_RT_DIR)/legion/legion_f_c_interface.f90.o : $(LG_RT_DIR)/legion/legion_f_c_interface.f90 $(LG_RT_DIR)/legion/legion_f_types.f90.o $(LEGION_DEFINES_HEADER) $(REALM_DEFINES_HEADER)
	$(FC) -J$(LG_RT_DIR) -o $@ -c $< $(FC_FLAGS) $(INC_FLAGS)

$(LG_RT_DIR)/legion/legion_f.f90.o : $(LG_RT_DIR)/legion/legion_f.f90 $(LG_RT_DIR)/legion/legion_f_c_interface.f90.o $(LG_RT_DIR)/legion/legion_f_types.f90.o $(LEGION_DEFINES_HEADER) $(REALM_DEFINES_HEADER)
	$(FC) -J$(LG_RT_DIR) -o $@ -c $< $(FC_FLAGS) $(INC_FLAGS)

$(filter %.f90.o,$(APP_OBJS)) : %.f90.o : %.f90 $(LEGION_DEFINES_HEADER) $(REALM_DEFINES_HEADER) $(filter %.f90.o,$(LEGION_OBJS))
	$(FC) -o $@ -c $< $(FC_FLAGS) $(INC_FLAGS)
endif

# disable gmake's default rule for building % from %.o
% : %.o

clean::
	$(RM) -f $(OUTFILE) $(SLIB_LEGION) $(SLIB_REALM) $(APP_OBJS) $(REALM_OBJS) $(REALM_INST_OBJS) $(LEGION_OBJS) $(LEGION_INST_OBJS) $(MAPPER_OBJS) $(LG_RT_DIR)/*mod *.mod $(LEGION_DEFINES_HEADER) $(REALM_DEFINES_HEADER) $(DEP_FILES)

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
