# Required environment variables
ifndef CONDUIT
$(error CONDUIT variable is not defined)
endif
ifndef CUDA
$(error CUDA variable is not defined)
endif
ifndef GPU_ARCH
$(error GPU_ARCH variable is not defined)
endif
ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined)
endif
ifndef GASNET
$(error GASNET variable is not defined)
endif

# Flags for directing the runtime makefile what to include
DEBUG           ?= 1		# Include debugging symbols
MAX_DIM         ?= 3		# Maximum number of dimensions
OUTPUT_LEVEL    ?= LEVEL_DEBUG	# Compile time logging level
USE_CUDA        ?= 1		# Include CUDA support (requires CUDA)
USE_GASNET      ?= 1		# Include GASNet support (requires GASNet)
USE_HDF         ?= 0		# Include HDF5 support (requires HDF5)
ALT_MAPPERS     ?= 0		# Include alternative mappers (not recommended)

# Put the binary file name here
OUTFILE		?= nccl_legion_demo
# List all the application source files here
GEN_SRC		?=                              # .cc files
GEN_GPU_SRC	?= nccl_legion_demo.cu          # .cu files

# You can modify these variables, some will be appended to by the runtime makefile
INC_FLAGS	?= -I$(CUDA)/include
CC_FLAGS	?=
NVCC_FLAGS	?=
GASNET_FLAGS	?=
LD_FLAGS	?= -lnccl

###########################################################################
#
#   Don't change anything below here
#
###########################################################################

include $(LG_RT_DIR)/runtime.mk
