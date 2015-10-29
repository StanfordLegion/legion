# Copyright 2015 Stanford University
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

ifndef TERRA_DIR
$(error TERRA_DIR variable is not defined, aborting build)
endif

#Flags for directing the runtime makefile what to include
DEBUG?=1                   # Include debugging symbols
OUTPUT_LEVEL?=LEVEL_DEBUG  # Compile time print level
SHARED_LOWLEVEL?=1	  # Use the shared low level
#ALT_MAPPERS?=1		  # Include the alternative mappers

# Put the binary file name here
OUTFILE		:= liblegion_terra.so
# List all the application source files here
GEN_SRC		:= \
  legion_terra.cc lua_mapper_wrapper.cc \
  legion_terra_partitions.cc \
  liszt_gpu_mapper.cc \
  # .cc files
GEN_GPU_SRC	:=		# .cu files

CC_FLAGS ?= #-DPROF_BINDING #-DPOINTER_CHECKS #-DLEGION_SPY
NVCC_FLAGS ?=
GASNET_FLAGS ?=
INC_FLAGS += -I$(TERRA_DIR)/release/include/terra -I$(TERRA_DIR)/include/terra
LD_FLAGS +=

CC_FLAGS += -fPIC
ifeq ($(shell uname), Darwin)
	LD_FLAGS += -dynamiclib -single_module -undefined dynamic_lookup -fPIC
else
	LD_FLAGS += -shared
endif

###########################################################################
#
#   Don't change anything below here
#
###########################################################################

include $(LG_RT_DIR)/runtime.mk

