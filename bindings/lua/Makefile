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

ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined, aborting build)
endif

ifndef LUAJIT_DIR
ifndef LUA_DIR
	$(error neither LUAJIT_DIR nor LUA_DIR variable is defined, aborting build)
endif
endif

ifndef LUABIND_DIR
$(error LUABIND_DIR variable is not defined, aborting build)
endif

#Flags for directing the runtime makefile what to include
DEBUG?=1                   # Include debugging symbols
OUTPUT_LEVEL=LEVEL_DEBUG  # Compile time print level
SHARED_LOWLEVEL=1	  # Use the shared low level
#ALT_MAPPERS=1		  # Include the alternative mappers
WITH_TERRA?=0
LEGION_LOGGING?=0

ifeq ($(WITH_TERRA),1)
ifndef TERRA_DIR
$(error TERRA_DIR variable is not defined, aborting build)
endif
endif

# Put the binary file name here
OUTFILE		:= libbinding.so
# List all the application source files here
GEN_SRC		:= binding.cc	# .cc files
GEN_GPU_SRC	:=		# .cu files

# You can modify these variables, some will be appended to by the runtime makefile
INC_FLAGS	:= -I$(LUABIND_DIR)/include/
ifdef LUAJIT_DIR
INC_FLAGS += -I$(LUAJIT_DIR)/include/luajit-2.0/ -I$(LUAJIT_DIR)/include/ 
else 
ifdef LUA_DIR
INC_FLAGS += -I$(LUA_DIR)/include
endif
endif

CC_FLAGS	:= -fPIC #-DPOINTER_CHECKS #-DLEGION_SPY 
NVCC_FLAGS	:=
GASNET_FLAGS	:=
LD_FLAGS	:= -L$(LUABIND_DIR)/lib/ -lluabind
ifdef LUAJIT_DIR
LD_FLAGS += -L$(LUAJIT_DIR)/lib/ -lluajit-5.1
else 
ifdef LUA_DIR
LD_FLAGS += -L$(LUA_DIR)/lib/ -llua
endif
endif

ifeq ($(WITH_TERRA),1)
	INC_FLAGS += -I$(TERRA_DIR)/src
	LD_FLAGS += -L$(TERRA_DIR)/build -lterra
	CC_FLAGS += -DWITH_TERRA
endif

ifeq ($(LEGION_LOGGING), 1)
	CC_FLAGS += -DLEGION_LOGGING
endif

ifeq ($(PROF_BINDING), 1)
	CC_FLAGS += -DPROF_BINDING
endif


###########################################################################
#
#   Don't change anything below here
#   
###########################################################################

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

common_all : all

.PHONY	: common_all

# All these variables will be filled in by the runtime makefile
LOW_RUNTIME_SRC	:=
HIGH_RUNTIME_SRC:=
GPU_RUNTIME_SRC	:=
MAPPER_SRC	:=

include $(LG_RT_DIR)/runtime.mk

GEN_OBJS	:= $(GEN_SRC:.cc=.o)
LOW_RUNTIME_OBJS:= $(LOW_RUNTIME_SRC:.cc=.o)
HIGH_RUNTIME_OBJS:=$(HIGH_RUNTIME_SRC:.cc=.o)
MAPPER_OBJS	:= $(MAPPER_SRC:.cc=.o)
# Only compile the gpu objects if we need to 
ifndef SHARED_LOWLEVEL
GEN_GPU_OBJS	:= $(GEN_GPU_SRC:.cu=.o)
GPU_RUNTIME_OBJS:= $(GPU_RUNTIME_SRC:.cu=.o)
else
GEN_GPU_OBJS	:=
GPU_RUNTIME_OBJS:=
endif

ALL_OBJS	:= $(GEN_OBJS) $(GEN_GPU_OBJS) $(LOW_RUNTIME_OBJS) $(HIGH_RUNTIME_OBJS) $(GPU_RUNTIME_OBJS) $(MAPPER_OBJS)

all:
	$(MAKE) $(OUTFILE)

# If we're using the general low-level runtime we have to link with nvcc
$(OUTFILE) : $(ALL_OBJS)
	@echo "---> Linking objects into one binary: $(OUTFILE)"
ifdef SHARED_LOWLEVEL
	$(GCC) -o $(OUTFILE) $(ALL_OBJS) $(LD_FLAGS) -shared $(GASNET_FLAGS)
else
	$(NVCC) -o $(OUTFILE) $(ALL_OBJS) $(LD_FLAGS) -shared $(GASNET_FLAGS)
endif

$(GEN_OBJS) : %.o : %.cc
	$(GCC) -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)

$(LOW_RUNTIME_OBJS) : %.o : %.cc
	$(GCC) -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)

$(HIGH_RUNTIME_OBJS) : %.o : %.cc
	$(GCC) -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)

$(MAPPER_OBJS)	: %.o : %.cc
	$(GCC) -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)

$(GEN_GPU_OBJS) : %.o : %.cu
	$(NVCC) -o $@ -c $< $(INC_FLAGS) $(NVCC_FLAGS)

$(GPU_RUNTIME_OBJS): %.o : %.cu
	$(NVCC) -o $@ -c $< $(INC_FLAGS) $(NVCC_FLAGS)

clean:
	@$(RM) -rf $(ALL_OBJS) $(OUTFILE)
