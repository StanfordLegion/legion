
ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined, aborting build)
endif

#Flags for directing the runtime makefile what to include
DEBUG ?= 1                   # Include debugging symbols
OUTPUT_LEVEL ?= LEVEL_INFO  # Compile time print level

# GASNet and CUDA off by default for now
USE_GASNET ?= 0
USE_CUDA ?= 0

# Put the binary file name here
OUTFILE		:= memcpy 
# List all the application source files here
GEN_SRC		:= memcpy.cc   # .cc files
GEN_GPU_SRC	:=		    # .cu files

# You can modify these variables, some will be appended to by the runtime makefile
INC_FLAGS	:=
NVCC_FLAGS	:=
GASNET_FLAGS	:=
LD_FLAGS	:=

include $(LG_RT_DIR)/runtime.mk

TESTARGS.default =
RUNMODE ?= default

run : $(OUTFILE)
	@echo $(dir $(OUTFILE))$(notdir $(OUTFILE)) $(TESTARGS.$(RUNMODE))
	@$(dir $(OUTFILE))$(notdir $(OUTFILE)) $(TESTARGS.$(RUNMODE))
