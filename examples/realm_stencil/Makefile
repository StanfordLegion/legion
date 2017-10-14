# Copyright 2017 Stanford University
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

# Put the binary file name here
OUTFILE		?= realm_stencil
# List all the application source files here
GEN_SRC		?= realm_stencil.cc		# .cc files
GEN_GPU_SRC	?=		# .cu files

# You can modify these variables, some will be appended to by the runtime makefile
INC_FLAGS	?=
CC_FLAGS	?=
NVCC_FLAGS	?=
GASNET_FLAGS	?=
LD_FLAGS	?=

# Stencil parameters
DTYPE ?= double
RESTRICT ?= __restrict__
RADIUS ?= 2
CC_FLAGS += -DDTYPE=$(DTYPE) -DRESTRICT=$(RESTRICT) -DRADIUS=$(RADIUS)

# These files have to be compiled separately to be absolutely sure they use -O3
KERNEL_SRC := cpu_kernels.cc

###########################################################################
#
#   Don't change anything below here
#
###########################################################################

include $(LG_RT_DIR)/runtime.mk

# Redefine targets to include kernels
KERNEL_OBJS := $(KERNEL_SRC:.cc=.o)
$(KERNEL_OBJS) : %.o : %.cc
	$(CXX) -o $@ -c $< $(CC_FLAGS) -O3 $(INC_FLAGS)

$(OUTFILE) : $(GEN_OBJS) $(KERNEL_OBJS) $(GEN_GPU_OBJS) $(SLIB_LEGION) $(SLIB_REALM)
	@echo "---> Linking objects into one binary: $(OUTFILE)"
	$(CXX) -o $(OUTFILE) $(GEN_OBJS) $(KERNEL_OBJS) $(GEN_GPU_OBJS) $(LD_FLAGS) $(LEGION_LIBS) $(LEGION_LD_FLAGS) $(GASNET_FLAGS)

clean::
	$(RM) -f $(OUTFILE) $(SLIB_LEGION) $(SLIB_REALM) $(GEN_OBJS) $(KERNEL_OBJS) $(GEN_GPU_OBJS) $(LOW_RUNTIME_OBJS) $(HIGH_RUNTIME_OBJS) $(GPU_RUNTIME_OBJS) $(MAPPER_OBJS) $(ASM_OBJS)
