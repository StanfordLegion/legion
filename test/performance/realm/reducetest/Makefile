
ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined, aborting build)
endif

#Flags for directing the runtime makefile what to include
DEBUG ?= 0                   # Include debugging symbols
OUTPUT_LEVEL ?= LEVEL_INFO  # Compile time print level

# GASNet and CUDA off by default for now
USE_GASNET ?= 0
USE_CUDA ?= 0

# Put the binary file name here
OUTFILE		:= reducetest
# List all the application source files here
GEN_SRC		:= reducetest.cc # .cc files
GEN_GPU_SRC	:=		    # .cu files

# You can modify these variables, some will be appended to by the runtime makefile
INC_FLAGS	:=
NVCC_FLAGS	:= -Xptxas -v
GASNET_FLAGS	:=
LD_FLAGS	:=

include $(LG_RT_DIR)/runtime.mk

ifndef NVCC
NVCC	= $(CUDA)/bin/nvcc
endif

TESTARGS.default =
TESTARGS.short = -batches 1
TESTARGS.long = -batches 1024
RUNMODE ?= default

run : $(OUTFILE)
	@echo $(dir $(OUTFILE))$(notdir $(OUTFILE)) $(TESTARGS.$(RUNMODE))
	@$(dir $(OUTFILE))$(notdir $(OUTFILE)) $(TESTARGS.$(RUNMODE))
