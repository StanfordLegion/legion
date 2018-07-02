#------------------------------------------------------------------------------#
# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#------------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.1)
project(LegionBindings_regent)

# Only search if were building stand-alone and not as part of Legion
if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

# Regent bindings
list(APPEND REGENT_SRC
  legion_terra.h                legion_terra.cc
  legion_terra_partitions.h     legion_terra_partitions.cc
  legion_terra_partitions_cxx.h
  legion_terra_util.h
)

if(Legion_USE_CUDA)
  list(APPEND REGENT_SRC
    legion_terra_cudart_hijack.h legion_terra_cudart_hijack.cc
  )
endif()

# Bishop bindings
list(APPEND REGENT_SRC
  bishop_mapper.h bishop_mapper.cc
  bishop_c.h      bishop_c.cc
)

# Murmur3 hash
list(APPEND REGENT_SRC
  murmur_hash3.h  murmur_hash3.cc
)

add_library(Regent ${REGENT_SRC})
target_link_libraries(Regent PUBLIC Legion::Legion)
set_target_properties(Regent PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(Regent PROPERTIES OUTPUT_NAME regent)
set_target_properties(Regent PROPERTIES SOVERSION ${SOVERSION})

install(TARGETS Regent EXPORT LegionTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
