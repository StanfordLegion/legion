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

LG_RT_DIR=../../runtime/

#Flags for directing the runtime makefile what to include
DEBUG=1                   # Include debugging symbols
OUTPUT_LEVEL=LEVEL_DEBUG  # Compile time print level
SHARED_LOWLEVEL=0	  # Use the shared low level
USE_GASNET=1

#ALT_MAPPERS=1		  # Compile the alternative mappers
USE_CUDA	=0 # no cuda thank you

GASNET		:= ${HOME}/local
CONDUIT		:= ibv
# Put the binary file name here
OUTFILE		:= tester_io
# List all the application source files here
GEN_SRC		:= legion_io.cc tester_io.cc	# .cc files
GEN_GPU_SRC	:=				# .cu files
USE_HDF 	= 1

# You can modify these variables, some will be appended to by the runtime makefile
INC_FLAGS	:= -I/opt/local/include/ 
CC_FLAGS	:= -g -Wno-array-bounds $(DEBUG_FLAGS)
NVCC_FLAGS	:=
GASNET_FLAGS	:=
LD_FLAGS	:= -L/opt/local/lib -lhdf5

###########################################################################
#
#   Don't change anything below here
#   
###########################################################################


include $(LG_RT_DIR)/runtime.mk

