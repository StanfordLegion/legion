# Copyright 2021 Stanford University
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
# vpath stuff below doesn't like a trailing slash in LG_RT_DIR
override LG_RT_DIR := $(patsubst %/,%,$(LG_RT_DIR))

#Flags for directing the runtime makefile what to include
DEBUG=1                   # Include debugging symbols
OUTPUT_LEVEL=LEVEL_DEBUG  # Compile time print level
#ALT_MAPPERS=1		  # Compile the alternative mappers

# Put the binary file name here
OUTFILE		:= 
# List all the application source files here
GEN_SRC		:= 				# .cc files
GEN_GPU_SRC	:=				# .cu files

# You can modify these variables, some will be appended to by the runtime makefile
INC_FLAGS	?=
CC_FLAGS	?=
NVCC_FLAGS	?=
GASNET_FLAGS	?=
LD_FLAGS	?=

CXX ?= g++

# we're going to include runtime.mk to get variable settings, but then
#  do our own build steps
NO_BUILD_RULES=1
include $(LG_RT_DIR)/runtime.mk

TESTS := serializing test_profiling ctxswitch barrier_reduce taskreg memspeed idcheck inst_reuse transpose
TESTS += proc_group
TESTS += deppart
TESTS += scatter
TESTS += compqueue
TESTS += event_subscribe
TESTS += deferred_allocs
TESTS += test_nodeset
TESTS += subgraphs
TESTS += large_tls
TESTS += coverings
TESTS += alltoall

# can set arguments to be passed to a test when running
TESTARGS_ctxswitch := -ll:io 1 -t 20 -i 10000
TESTARGS_proc_group := -ll:cpu 4
TESTARGS_compqueue := -ll:cpu 4
TESTARGS_event_subscribe := -ll:cpu 4
TESTARGS_deferred_allocs := -ll:gsize 0 -all
TESTARGS_scatter := -p1 2 -p2 2

REALM_OBJS := $(patsubst %.cc,%.o,$(notdir $(REALM_SRC))) \
              $(patsubst %.cc.o,%.o,$(notdir $(REALM_INST_OBJS))) \
              $(patsubst %.S,%.o,$(notdir $(ASM_SRC)))
EMPTY :=
SPACE := $(EMPTY) $(EMPTY)
RUNTIME_VPATH := $(subst $(SPACE),:,$(sort $(dir $(REALM_SRC))))
vpath %.cc .:$(RUNTIME_VPATH)
vpath %.S .:$(RUNTIME_VPATH)
#VPATH = .:$(RUNTIME_VPATH)

REALM_LIB := librealm.a

TEST_OBJS := $(TESTS:%=%.o)

run_all : $(TESTS:%=run_%)

run_% : %
	@# this echos exactly once, even if -s was specified
	@echo $(LAUNCHER) ./$* $(TESTARGS_$*)
	@$(LAUNCHER) ./$* $(TESTARGS_$*)

build : $(TESTS)

clean :
	rm -f $(REALM_LIB) $(REALM_OBJS) $(TESTS) $(TEST_OBJS)

# scatter uses C++11 lambdas
scatter.o : CC_FLAGS += -std=c++11

$(TESTS) : % : %.o librealm.a
	$(CXX) -o $@ $< $(EXTRAOBJS_$*) -L. -lrealm $(LEGION_LD_FLAGS) $(LD_FLAGS)

$(REALM_LIB) : $(REALM_OBJS)
	rm -f $(REALM_LIB)
	ar rc $(REALM_LIB) $(REALM_OBJS)

ifeq ($(strip $(USE_CUDA)),1)
EXTRAOBJS_memspeed := memspeed_gpu.o
TEST_OBJS += $(EXTRAOBJS_memspeed)
memspeed : memspeed_gpu.o

%.o : %.cu $(REALM_DEFINES_HEADER) $(LEGION_DEFINES_HEADER)
	$(NVCC) -o $@ -c $< $(INC_FLAGS) $(NVCC_FLAGS)
endif

%.o : %.cc $(REALM_DEFINES_HEADER) $(LEGION_DEFINES_HEADER)
	$(CXX) -fPIC -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)

$(REALM_OBJS) : CC_FLAGS+=$(REALM_SYMBOL_VISIBILITY)

%.o : %.S
	$(CXX) -fPIC -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)

# deppart-related stuff
image_%.o : image_tmpl.cc $(REALM_DEFINES_HEADER) $(LEGION_DEFINES_HEADER)
	$(CXX) -fPIC -o $@ -c $< $(INC_FLAGS) $(REALM_SYMBOL_VISIBILITY) $(CC_FLAGS) -DINST_N1=$(word 1,$(subst _, ,$*)) -DINST_N2=$(word 2,$(subst _, ,$*))

preimage_%.o : preimage_tmpl.cc $(REALM_DEFINES_HEADER) $(LEGION_DEFINES_HEADER)
	$(CXX) -fPIC -o $@ -c $< $(INC_FLAGS) $(REALM_SYMBOL_VISIBILITY) $(CC_FLAGS) -DINST_N1=$(word 1,$(subst _, ,$*)) -DINST_N2=$(word 2,$(subst _, ,$*))

byfield_%.o : byfield_tmpl.cc $(REALM_DEFINES_HEADER) $(LEGION_DEFINES_HEADER)
	$(CXX) -fPIC -o $@ -c $< $(INC_FLAGS) $(REALM_SYMBOL_VISIBILITY) $(CC_FLAGS) -DINST_N1=$(word 1,$(subst _, ,$*)) -DINST_N2=$(word 2,$(subst _, ,$*))

