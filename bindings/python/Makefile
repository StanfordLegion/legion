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
MAX_DIM         ?= 3		# Maximum number of dimensions
OUTPUT_LEVEL    ?= LEVEL_INFO	# Compile time logging level
USE_CUDA        ?= 0		# Include CUDA support (requires CUDA)
USE_GASNET      ?= 0		# Include GASNet support (requires GASNet)
USE_HDF         ?= 0		# Include HDF5 support (requires HDF5)
ALT_MAPPERS     ?= 0		# Include alternative mappers (not recommended)

USE_PYTHON := 1
SHARED_OBJECTS := 1 # Make liblegion.so and librealm.so

# Put the binary file name here
OUTFILE		?= legion_python
# List all the application source files here
GEN_SRC		?= main.cc		# .cc files
GEN_GPU_SRC	?=		# .cu files

# You can modify these variables, some will be appended to by the runtime makefile
INC_FLAGS	?=
CC_FLAGS	?= -DBINDINGS_AUGMENT_PYTHONPATH
NVCC_FLAGS	?=
GASNET_FLAGS	?=
LD_FLAGS	?=

# Use reduction operators for complex types by default
USE_COMPLEX = 1

ifeq ($(shell uname -s),Darwin)
LEGION_CANONICAL_PYTHON_LIB := liblegion_canonical_python.dylib
else
LEGION_CANONICAL_PYTHON_LIB := liblegion_canonical_python.so
endif

NO_BUILD_ALL=1
.PHONY: all
all: $(OUTFILE) legion_info.py legion_builtin_cffi.py legion_canonical_cffi.py $(LEGION_CANONICAL_PYTHON_LIB)

###########################################################################
#
#   Don't change anything below here
#
###########################################################################

PYTHON_BINDINGS_DIR:=$(dir $(abspath $(lastword $(MAKEFILE_LIST))))

include $(LG_RT_DIR)/runtime.mk

legion_builtin_cffi.py: legion_cffi.py.in legion_cffi_build.py $(LEGION_DEFINES_HEADER) $(REALM_DEFINES_HEADER)
	$(PYTHON_EXE) $(PYTHON_BINDINGS_DIR)/legion_cffi_build.py --defines-dir $(DEFINE_HEADERS_DIR)

legion_canonical_cffi.py: legion_cffi.py.in legion_cffi_build.py $(LEGION_DEFINES_HEADER) $(REALM_DEFINES_HEADER)
	$(PYTHON_EXE) $(PYTHON_BINDINGS_DIR)/legion_cffi_build.py --canonical --defines-dir $(DEFINE_HEADERS_DIR) --libname $(LEGION_CANONICAL_PYTHON_LIB)

legion_info.py: legion_info_build.py
	$(PYTHON_EXE) $(PYTHON_BINDINGS_DIR)/legion_info_build.py

# build the shared lib for canonical python
$(LEGION_CANONICAL_PYTHON_LIB): canonical_python.cc $(SLIB_LEGION) $(SLIB_REALM)
	$(CXX) $(CC_FLAGS) -fPIC $(INC_FLAGS) -shared $< $(LD_FLAGS) $(LEGION_LIBS) $(LEGION_LD_FLAGS) -o $@

clean::
	$(RM) -f $(OUTFILE) $(SLIB_LEGION) $(SLIB_REALM) $(GEN_OBJS) $(GEN_GPU_OBJS) $(REALM_OBJS) $(LEGION_OBJS) $(GPU_RUNTIME_OBJS) $(MAPPER_OBJS) $(ASM_OBJS) $(LEGION_CANONICAL_PYTHON_LIB) legion_defines.h realm_defines.h legion_builtin_cffi.py legion_canonical_cffi.py legion_info.py *.pyc
	$(RM) -rf build dist *.egg-info
