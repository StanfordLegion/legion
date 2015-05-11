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

#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#

list(APPEND lg_headers
    ${CMAKE_CURRENT_SOURCE_DIR}/accessor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/activemsg.h
    ${CMAKE_CURRENT_SOURCE_DIR}/alt_mappers.h
    ${CMAKE_CURRENT_SOURCE_DIR}/arrays.h
    ${CMAKE_CURRENT_SOURCE_DIR}/atomics.h
    ${CMAKE_CURRENT_SOURCE_DIR}/common.h
    ${CMAKE_CURRENT_SOURCE_DIR}/default_mapper.h
    ${CMAKE_CURRENT_SOURCE_DIR}/field_tree.h
    ${CMAKE_CURRENT_SOURCE_DIR}/garbage_collection.h
    ${CMAKE_CURRENT_SOURCE_DIR}/interval_tree.h
    ${CMAKE_CURRENT_SOURCE_DIR}/legion_allocation.h
    ${CMAKE_CURRENT_SOURCE_DIR}/legion_c.h
    ${CMAKE_CURRENT_SOURCE_DIR}/legion_config.h
    ${CMAKE_CURRENT_SOURCE_DIR}/legion_c_util.h
    ${CMAKE_CURRENT_SOURCE_DIR}/legion.h
    ${CMAKE_CURRENT_SOURCE_DIR}/legion_logging.h
    ${CMAKE_CURRENT_SOURCE_DIR}/legion_ops.h
    ${CMAKE_CURRENT_SOURCE_DIR}/legion_profiling.h
    ${CMAKE_CURRENT_SOURCE_DIR}/legion_spy.h
    ${CMAKE_CURRENT_SOURCE_DIR}/legion_tasks.h
    ${CMAKE_CURRENT_SOURCE_DIR}/legion_trace.h
    ${CMAKE_CURRENT_SOURCE_DIR}/legion_types.h
    ${CMAKE_CURRENT_SOURCE_DIR}/legion_utilities.h
    ${CMAKE_CURRENT_SOURCE_DIR}/lowlevel_config.h
    ${CMAKE_CURRENT_SOURCE_DIR}/lowlevel_dma.h
    ${CMAKE_CURRENT_SOURCE_DIR}/lowlevel_gpu.h
    ${CMAKE_CURRENT_SOURCE_DIR}/lowlevel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/lowlevel_impl.h
    ${CMAKE_CURRENT_SOURCE_DIR}/mapping_utilities.h
    ${CMAKE_CURRENT_SOURCE_DIR}/rectangle_set.h
    ${CMAKE_CURRENT_SOURCE_DIR}/region_tree.h
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime.h
    ${CMAKE_CURRENT_SOURCE_DIR}/shim_mapper.h
    ${CMAKE_CURRENT_SOURCE_DIR}/utilities.h
)

#------------------------------------------------------------------------------#
# Set the source files
#------------------------------------------------------------------------------# 
# initialize to nothing
set(low_runtime_src)

if(ENABLE_SHARED_LOWLEVEL)

  # Shared low-level
  list(APPEND low_runtime_src
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_lowlevel.cc
  )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSHARED_LOWLEVEL")

else()

  list(APPEND low_runtime_src
    ${CMAKE_CURRENT_SOURCE_DIR}/lowlevel.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/lowlevel_disk.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/lowlevel_dma.cc
  )

  # CUDA
  if(ENABLE_CUDA)
    list(APPEND low_runtime_src
      ${CMAKE_CURRENT_SOURCE_DIR}/lowlevel_gpu.cc
    )
  endif(ENABLE_CUDA)

  # GASNet
  if(ENABLE_GASNET)
    list(APPEND low_runtime_src
      ${CMAKE_CURRENT_SOURCE_DIR}/activemsg.cc
    )
  endif(ENABLE_GASNET)

endif(ENABLE_SHARED_LOWLEVEL)

#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#

if(ENABLE_SHARED_MAPPER)
  list(APPEND mapper_src
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_mapper.cc
  )
else()
  list(APPEND mapper_src
    ${CMAKE_CURRENT_SOURCE_DIR}/default_mapper.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/shim_mapper.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/mapping_utilities.cc
  )
endif(ENABLE_SHARED_MAPPER)

#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#

list(APPEND high_runtime_src
    ${CMAKE_CURRENT_SOURCE_DIR}/legion.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/legion_ops.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/legion_tasks.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/legion_trace.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/legion_spy.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/region_tree.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/garbage_collection.cc
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(lglow ${low_runtime_src})
add_library(lgmap ${mapper_src})
add_library(lg ${high_runtime_src})

install(TARGETS lg DESTINATION lib)
install(TARGETS lgmap DESTINATION lib)
install(TARGETS lglow DESTINATION lib)
install(FILES ${lg_headers} DESTINATION include)

#------------------------------------------------------------------------------#
# vim: set tabstop=2 shiftwidth=2 expandtab :
#------------------------------------------------------------------------------#
