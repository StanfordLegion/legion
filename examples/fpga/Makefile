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

# Flags for directing the runtime makefile what to include
DEBUG           ?= 1		# Include debugging symbols
MAX_DIM         ?= 3		# Maximum number of dimensions
OUTPUT_LEVEL    ?= LEVEL_DEBUG	# Compile time logging level
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
OUTFILE		?= fpga
# List all the application source files here
GEN_SRC		?= fpga.cc	# .cc files

# You can modify these variables, some will be appended to by the runtime makefile
INC_FLAGS	?=
CC_FLAGS	?=
NVCC_FLAGS	?=
HIPCC_FLAGS     ?=
GASNET_FLAGS	?=
LD_FLAGS	?=

.PHONY: run
run: kernel all
	XCL_EMULATION_MODE=$(TARGET) ./fpga -ll:fpga 1 -ll:fpga_size 4 -ll:fpga_xclbin vadd.xclbin

kernel: vadd.xclbin emconfig

vadd.xo: vadd.cpp
	v++ $(VPPFLAGS) -t $(TARGET) --platform $(PLATFORM) -c -k vadd -o vadd.xo vadd.cpp

# create 4 CUs
vadd.xclbin: vadd.xo
	v++ $(VPPFLAGS) -t $(TARGET) --platform $(PLATFORM) -l --connectivity.nk vadd:4 --connectivity.sp vadd_1.m_axi_gmem:DDR[0] --connectivity.sp vadd_2.m_axi_gmem:DDR[0] --connectivity.sp vadd_3.m_axi_gmem:DDR[0] --connectivity.sp vadd_4.m_axi_gmem:DDR[0] -o vadd.xclbin vadd.xo

emconfig: emconfig.json
emconfig.json:
	emconfigutil --platform $(PLATFORM)

.PHONY: cleankernel
cleankernel:
	rm -rf _x *.xo *.xclbin *.log *.xclbin.* *.xo.* emconfig.json .Xil .run

.PHONY: cleanall
cleanall: clean cleankernel

###########################################################################
#
#   Don't change anything below here
#
###########################################################################

include $(LG_RT_DIR)/runtime.mk

