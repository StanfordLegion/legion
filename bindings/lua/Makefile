# Copyright 2018 Stanford University
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
#ALT_MAPPERS=1		  # Include the alternative mappers
WITH_TERRA?=0
WITH_TERRA_STATIC_LINKING?=0
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
ifeq ($(WITH_TERRA_STATIC_LINKING),0)
LD_FLAGS += -L$(LUAJIT_DIR)/lib/ -lluajit-5.1
endif
else 
ifdef LUA_DIR
LD_FLAGS += -L$(LUA_DIR)/lib/ -llua
endif
endif

ifeq ($(WITH_TERRA),1)
	INC_FLAGS += -I$(TERRA_DIR)/src
ifeq ($(WITH_TERRA_STATIC_LINKING),0)
	LD_FLAGS += -L$(TERRA_DIR)/build -lterra
endif
	CC_FLAGS += -DWITH_TERRA
endif

ifeq ($(WITH_TERRA_STATIC_LINKING),1)
ifeq ($(shell uname), Darwin)
	LD_FLAGS += -dynamiclib -single_module -undefined dynamic_lookup -fPIC
else
	LD_FLAGS += -shared
endif
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

include $(LG_RT_DIR)/runtime.mk

