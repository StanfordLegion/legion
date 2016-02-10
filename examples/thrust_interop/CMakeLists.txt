#------------------------------------------------------------------------------#
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
#------------------------------------------------------------------------------#

if (ENABLE_CUDA)
CUDA_INCLUDE_DIRECTORIES( ".") 
CUDA_COMPILE( CUDA_OBJECT thrust_shim.cu)
endif (ENABLE_CUDA)

add_executable(thrust_interop thrust_interop.cc ${CUDA_OBJECT})

target_link_libraries(thrust_interop ${EXTRA_LIBS} legion realm legion_mappers)
target_include_directories(thrust_interop
    PUBLIC "${CMAKE_SOURCE_DIR}/runtime"
   "${CMAKE_SOURCE_DIR}/runtime/legion"
    "${CMAKE_SOURCE_DIR}/runtime/mappers" )
