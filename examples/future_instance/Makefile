# Copyright 2022 Stanford University
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
DEBUG           ?= 0		# Include debugging symbols
MAX_DIM         ?= 3		# Maximum number of dimensions
MAX_FIELDS	?= 256		# Maximum number of fields in a field space
OUTPUT_LEVEL    ?= LEVEL_PRINT 	# Compile time logging level
USE_FORTRAN	?= 0		# Include Fortran support		
USE_CUDA        ?= 1		# Include CUDA support (requires CUDA)
USE_OPENMP	?= 0		# Include OpenMP processor support
USE_GASNET      ?= 0		# Include GASNet support (requires GASNet)
USE_ZLIB	?= 1		# Use ZLib for compression of log files
USE_LIBDL	?= 1		# Use LibDL for finding function pointer names
USE_LLVM	?= 0		# Include support for LLVM task variants
USE_HDF         ?= 0		# Include HDF5 support (requires HDF5)
USE_SPY		?= 0		# Enable support for detailed Legion Spy logging
USE_HALF	?= 0		# Include support for half-precision reductions
USE_COMPLEX	?= 0		# Include support for complex type reductions
SHARED_OBJECTS	?= 1		# Generate shared objects for Legion and Realm
BOUNDS_CHECKS	?= 0		# Enable runtime bounds checks
PRIVILEGE_CHECKS ?= 0		# Enable runtime privilege checks
MARCH		?= native	# Set the name of the target CPU archiecture
GPU_ARCH	?= turing	# Set the name of the target GPU architecture
CONDUIT		?= ibv		# Set the name of the GASNet conduit to use
#REALM_NETWORKS	?= gasnetex	# Set the kind of networking layer to use
GASNET		?=		# Location of GASNet installation
CUDA		?=		# Location of CUDA installation
HDF_ROOT	?=		# Location of HDF5 installation
PREFIX		?= /usr		# Location of where to install Legion

# Put the binary file name here
OUTFILE		?= future_instance
# List all the application source files here
CC_SRC		?=		# .c files
CXX_SRC		?= 		# .cc files
CUDA_SRC	?= future_instance.cu		# .cu files
FORT_SRC	?=		# .f90 files
ASM_SRC		?=		# .S files

# You can modify these variables, some will be appended to by the runtime makefile
INC_FLAGS	?=		# Include flags for all compilers
CC_FLAGS	?=		# Flags for all C++ compilers
FC_FLAGS	?=		# Flags for all Fortran compilers
NVCC_FLAGS	?=		# Flags for all NVCC files
SO_FLAGS	?=		# Flags for building shared objects
LD_FLAGS	?=		# Flags for linking binaries
# Canonical GNU flags you can modify as well
CPPFLAGS 	?=
CFLAGS		?=
CXXFLAGS 	?=
FFLAGS 		?=
LDLIBS 		?=
LDFLAGS 	?=

###########################################################################
#
#   Don't change anything below here
#
###########################################################################

include $(LG_RT_DIR)/runtime.mk

