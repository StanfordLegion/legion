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
HIPCC_FLAGS	?=
GASNET_FLAGS	?=
LD_FLAGS	?=

REALM_CUDA_OBJS ?=

CXX ?= g++

# we're going to include runtime.mk to get variable settings, but then
#  do our own build steps
NO_BUILD_RULES=1
include $(LG_RT_DIR)/runtime.mk

TESTS := serializing test_profiling ctxswitch barrier_reduce taskreg memspeed idcheck inst_reuse transpose
TESTS += version_check
TESTS += proc_group
TESTS += deppart
TESTS += scatter
TESTS += compqueue
TESTS += event_subscribe
#TESTS += deferred_allocs
TESTS += test_nodeset
TESTS += subgraphs
TESTS += large_tls
TESTS += coverings
TESTS += alltoall
TESTS += simple_reduce
TESTS += realm_reinit
TESTS += sparse_construct
TESTS += memmodel
TESTS += extres_alias
TESTS += reservations
TESTS += multiaffine
TESTS += machine_config
TESTS += rect_transform_test
TESTS += rsrv_acquire_poisoned

ifeq ($(strip $(USE_CUDA)),1)
TESTS += task_stream
TESTS += test_cuhook
TESTS += cuda_memcpy_test
TESTS += cuda_transpose_test
endif

ifeq ($(strip $(USE_HIP)),1)
TESTS += task_stream
endif

# can set arguments to be passed to a test when running
TESTARGS_ctxswitch := -ll:io 1 -t 30 -i 10000
TESTARGS_proc_group := -ll:cpu 4
TESTARGS_compqueue := -ll:cpu 4
TESTARGS_event_subscribe := -ll:cpu 4
#TESTARGS_deferred_allocs := -ll:gsize 0 -all
TESTARGS_scatter := -p1 2 -p2 2
TESTARGS_alltoall := -ll:csize 1024
TESTARGS_sparse_construct := -verbose
TESTARGS_task_stream := -ll:gpu 1
TESTARGS_machine_config_args := -test_args 1 -ll:cpu 4 -ll:util 2 -ll:io 1 -ll:csize 16 -ll:stacksize 4 -ll:pin_util 1 -ll:ext_sysmem 0 -ll:rsize 2
ifeq ($(strip $(USE_CUDA)),1)
TESTARGS_machine_config_args += -ll:gpu 1 -ll:fsize 1024 -ll:zsize 8 -ll:ib_fsize 16 -ll:ib_zsize 32 -ll:msize 64 -cuda:dynfb 1 -cuda:dynfb_max 128 -ll:streams 2 -ll:d2d_streams 2
endif
ifeq ($(strip $(USE_HIP)),1)
TESTARGS_machine_config_args += -ll:gpu 1 -ll:fsize 1024 -ll:zsize 8 -ll:ib_fsize 16 -ll:ib_zsize 32 -hip:dynfb 1 -hip:dynfb_max 128 -ll:streams 2 -ll:d2d_streams 2
endif
ifeq ($(strip $(USE_OPENMP)),1)
TESTARGS_machine_config_args += -ll:ocpu 1 -ll:othr 2 -ll:onuma 0 -ll:ostack 4
endif
ifeq ($(strip $(USE_PYTHON)),1)
TESTARGS_machine_config_args += -ll:py 1 -ll:pystack 4
endif

REALM_OBJS := $(patsubst %.cc,%.o,$(notdir $(REALM_SRC))) \
              $(patsubst %.cc.o,%.o,$(notdir $(REALM_INST_OBJS))) \
              $(patsubst %.S,%.o,$(notdir $(ASM_SRC)))

REALM_CUDA_OBJS ?=
ifeq ($(strip $(USE_CUDA)),1)
REALM_CUDA_OBJS := $(patsubst %.cu,$(REALM_CUDA_DIR)/%.o,$(notdir $(REALM_CUDA_SRC)))
endif

EMPTY :=
SPACE := $(EMPTY) $(EMPTY)
RUNTIME_VPATH := $(subst $(SPACE),:,$(sort $(dir $(REALM_SRC))))
vpath %.cc .:$(RUNTIME_VPATH)
vpath %.S .:$(RUNTIME_VPATH)
#VPATH = .:$(RUNTIME_VPATH)

REALM_LIB := librealm.a

TEST_OBJS := $(TESTS:%=%.o)

run_all : $(TESTS:%=run_%) run_machine_config_args

run_% : %
	@# this echos exactly once, even if -s was specified
	@echo $(LAUNCHER) ./$* $(TESTARGS_$*)
	@$(LAUNCHER) ./$* $(TESTARGS_$*)

run_machine_config_args : machine_config
	@# this echos exactly once, even if -s was specified
	@echo $(LAUNCHER) ./machine_config $(TESTARGS_machine_config_args)
	@$(LAUNCHER) ./machine_config $(TESTARGS_machine_config_args)

build : $(TESTS)

clean :
	rm -f $(REALM_LIB) $(REALM_OBJS) $(TESTS) $(TEST_OBJS) $(REALM_FATBIN_SRC) $(REALM_FATBIN) $(SLIB_REALM_CUHOOK) $(REALM_CUHOOK_OBJS)

machine_config.o : CC_FLAGS += $(REALM_SYMBOL_VISIBILITY)

$(TESTS) : % : %.o librealm.a $(SLIB_REALM_CUHOOK)
	$(CXX) -o $@ $< $(EXTRAOBJS_$*) -L. -lrealm $(LEGION_LD_FLAGS) $(LD_FLAGS)

$(REALM_LIB) : $(REALM_OBJS) $(REALM_CUDA_OBJS)
	rm -f $(REALM_LIB)
	ar rc $(REALM_LIB) $(REALM_OBJS) $(REALM_CUDA_OBJS)

ifeq ($(strip $(USE_CUDA)),1)
EXTRAOBJS_memspeed := memspeed_gpu.o
EXTRAOBJS_simple_reduce := simple_reduce_gpu.o
EXTRAOBJS_multiaffine := multiaffine_gpu.o
EXTRAOBJS_task_stream := task_stream_gpu.o
EXTRAOBJS_test_cuhook := test_cuhook_gpu.o
TEST_OBJS += $(EXTRAOBJS_memspeed) $(EXTRAOBJS_simple_reduce) $(EXTRAOBJS_multiaffine) $(EXTRAOBJS_task_stream) $(EXTRAOBJS_test_cuhook)
memspeed : memspeed_gpu.o
simple_reduce : simple_reduce_gpu.o
multiaffine : multiaffine_gpu.o
task_stream : task_stream_gpu.o
test_cuhook : test_cuhook_gpu.o

%.o : %.cu $(REALM_DEFINES_HEADER) $(LEGION_DEFINES_HEADER)
	$(NVCC) -o $@ -c $< $(INC_FLAGS) $(NVCC_FLAGS)

$(SLIB_REALM_CUHOOK) : $(REALM_CUHOOK_OBJS)
	rm -f $@
	$(CXX) --shared $(SO_FLAGS) -o $@ $^ -L$(CUDA)/lib64/stubs -lcuda -Xlinker -rpath=$(CUDA)/lib64

$(REALM_CUHOOK_OBJS) : %.cc.o : %.cc $(LEGION_DEFINES_HEADER) $(REALM_DEFINES_HEADER)
	# $(NVCC) --compiler-options '-fPIC' -o $@ -c $< $(INC_FLAGS) $(NVCC_FLAGS)
	$(CXX) -MMD -fPIC -o $@ -c $< $(CC_FLAGS) $(REALM_SYMBOL_VISIBILITY) $(INC_FLAGS) $(REALM_DEFCHECK)
endif

ifeq ($(strip $(USE_HIP)),1)
EXTRAOBJS_memspeed := memspeed_gpu.o
EXTRAOBJS_simple_reduce := simple_reduce_gpu.o
# FIXME: https://github.com/StanfordLegion/legion/issues/1476
# EXTRAOBJS_multiaffine := multiaffine_gpu.o
EXTRAOBJS_task_stream := task_stream_gpu.o
TEST_OBJS += $(EXTRAOBJS_memspeed) $(EXTRAOBJS_simple_reduce) $(EXTRAOBJS_multiaffine) $(EXTRAOBJS_task_stream)
memspeed : memspeed_gpu.o
simple_reduce : simple_reduce_gpu.o
# multiaffine : multiaffine_gpu.o
task_stream : task_stream_gpu.o

%.o : %.cu $(REALM_DEFINES_HEADER) $(LEGION_DEFINES_HEADER)
	$(HIPCC) -o $@ -c $< $(INC_FLAGS) $(HIPCC_FLAGS)
endif

%.o : %.cc $(REALM_DEFINES_HEADER) $(LEGION_DEFINES_HEADER)
	$(CXX) -fPIC -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)

$(REALM_OBJS) : CC_FLAGS+=$(REALM_SYMBOL_VISIBILITY)

%.o : %.S
	$(CXX) -fPIC -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)

# deppart-related stuff
ifneq ($(USE_PGI),1)
image_%.o : image_tmpl.cc $(REALM_DEFINES_HEADER) $(LEGION_DEFINES_HEADER)
	$(CXX) -fPIC -o $@ -c $< $(INC_FLAGS) $(REALM_SYMBOL_VISIBILITY) $(CC_FLAGS) -DINST_N1=$(word 1,$(subst _, ,$*)) -DINST_N2=$(word 2,$(subst _, ,$*))

preimage_%.o : preimage_tmpl.cc $(REALM_DEFINES_HEADER) $(LEGION_DEFINES_HEADER)
	$(CXX) -fPIC -o $@ -c $< $(INC_FLAGS) $(REALM_SYMBOL_VISIBILITY) $(CC_FLAGS) -DINST_N1=$(word 1,$(subst _, ,$*)) -DINST_N2=$(word 2,$(subst _, ,$*))

byfield_%.o : byfield_tmpl.cc $(REALM_DEFINES_HEADER) $(LEGION_DEFINES_HEADER)
	$(CXX) -fPIC -o $@ -c $< $(INC_FLAGS) $(REALM_SYMBOL_VISIBILITY) $(CC_FLAGS) -DINST_N1=$(word 1,$(subst _, ,$*)) -DINST_N2=$(word 2,$(subst _, ,$*))
else
# nvc++ names some symbols based on the source filename, so the trick above
#  of compiling multiple things from the same template with different defines
#  causes linker errors - work around by generating a different source file for
#  each case, but don't leave them lying around
image_%.cc :
	echo '#define' INST_N1 $(word 1,$(subst _, ,$(notdir $*))) > $@
	echo '#define' INST_N2 $(word 2,$(subst _, ,$(notdir $*))) >> $@
	echo '#include' '"realm/deppart/image_tmpl.cc"' >> $@

preimage_%.cc :
	echo '#define' INST_N1 $(word 1,$(subst _, ,$(notdir $*))) > $@
	echo '#define' INST_N2 $(word 2,$(subst _, ,$(notdir $*))) >> $@
	echo '#include' '"realm/deppart/preimage_tmpl.cc"' >> $@

byfield_%.cc :
	echo '#define' INST_N1 $(word 1,$(subst _, ,$(notdir $*))) > $@
	echo '#define' INST_N2 $(word 2,$(subst _, ,$(notdir $*))) >> $@
	echo '#include' '"realm/deppart/byfield_tmpl.cc"' >> $@

.INTERMEDIATE: $(patsubst %.cc.o,%.cc,$(notdir $(REALM_INST_OBJS)))
endif

