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

# Flags for directing the runtime makefile what to include
DEBUG           ?= 1		# Include debugging symbols
MAX_DIM         ?= 3		# Maximum number of dimensions
OUTPUT_LEVEL    ?= LEVEL_DEBUG	# Compile time logging level
USE_CUDA        ?= 0		# Include CUDA support (requires CUDA)
USE_GASNET      ?= 0		# Include GASNet support (requires GASNet)
USE_HDF         ?= 0		# Include HDF5 support (requires HDF5)
ALT_MAPPERS     ?= 0		# Include alternative mappers (not recommended)

# Include FPGA support (requires Xilinx FPGA)
USE_FPGA        ?= 1
# Path to the Xilinx platform definition file
PLATFORM        ?= /home/centos/src/project_data/aws-fpga/Vitis/aws_platform/xilinx_aws-vu9p-f1_shell-v04261818_201920_2/xilinx_aws-vu9p-f1_shell-v04261818_201920_2.xpfm
# FPGA kenrel compilation target: Software emulation (sw_emu), hardware emulation (hw_emu), hardware (hw)
TARGET          ?= sw_emu
# v++ flags
VPPFLAGS = -g -I.


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

TESTS := fpga_vadd

# can set arguments to be passed to a test when running
TESTARGS_fpga_vadd :=  -ll:fpga 1 -ll:fpga_size 4 -ll:fpga_xclbin vadd.xclbin -level fpga=1 -level app=1

REALM_OBJS := $(patsubst %.cc,%.o,$(notdir $(REALM_SRC))) \
              $(patsubst %.cc.o,%.o,$(notdir $(REALM_INST_OBJS))) \
              $(patsubst %.S,%.o,$(notdir $(ASM_SRC)))
EMPTY :=
SPACE := $(EMPTY) $(EMPTY)
RUNTIME_VPATH := $(subst $(SPACE),:,$(sort $(dir $(REALM_SRC))))
vpath %.cc .:$(RUNTIME_VPATH)
vpath %.S .:$(RUNTIME_VPATH)

REALM_LIB := librealm.a

TEST_OBJS := $(TESTS:%=%.o)

.PHONY: run_all
run_all : $(TESTS:%=run_%)

.PHONY: run_%
run_% : % vadd.xclbin emconfig
	@# this echos exactly once, even if -s was specified
	@echo XCL_EMULATION_MODE=$(TARGET) $(LAUNCHER) ./$* $(TESTARGS_$*)
	@XCL_EMULATION_MODE=$(TARGET) $(LAUNCHER) ./$* $(TESTARGS_$*)

.PHONY: build
build : $(TESTS) vadd.xclbin emconfig

.PHONY: cleankernel
cleankernel:
	rm -rf _x *.xo *.xclbin *.log *.xclbin.* *.xo.* emconfig.json .Xil .run

.PHONY: clean
clean : cleankernel
	rm -f $(REALM_LIB) $(REALM_OBJS) $(TESTS) $(TEST_OBJS)
	rm -f legion_defines.h realm_defines.h

$(TESTS) : % : %.o librealm.a
	$(CXX) -o $@ $< -L. -lrealm $(LEGION_LD_FLAGS) $(LD_FLAGS)

$(REALM_LIB) : $(REALM_OBJS)
	rm -f $(REALM_LIB)
	ar rc $(REALM_LIB) $(REALM_OBJS)

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

# Create FPGA kernel (vadd with 4 CUs)
vadd.xo: vadd.cpp
	v++ $(VPPFLAGS) -t $(TARGET) --platform $(PLATFORM) -c -k vadd -o vadd.xo vadd.cpp

vadd.xclbin: vadd.xo
	v++ $(VPPFLAGS) -t $(TARGET) --platform $(PLATFORM) -l --connectivity.nk vadd:4 --connectivity.sp vadd_1.m_axi_gmem:DDR[0] --connectivity.sp vadd_2.m_axi_gmem:DDR[0] --connectivity.sp vadd_3.m_axi_gmem:DDR[0] --connectivity.sp vadd_4.m_axi_gmem:DDR[0] -o vadd.xclbin vadd.xo

# Create emulation config file
emconfig: emconfig.json
emconfig.json:
	emconfigutil --platform $(PLATFORM)
