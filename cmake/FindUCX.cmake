#
# Copyright (c) 2021, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

find_path(UCX_INCLUDE_DIR ucp/api/ucp.h)

find_library(UCP_LIBRARY ucp)
find_library(UCT_LIBRARY uct)
find_library(UCM_LIBRARY ucm)
find_library(UCS_LIBRARY ucs)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(UCX DEFAULT_MSG UCX_INCLUDE_DIR
                                  UCP_LIBRARY UCT_LIBRARY UCM_LIBRARY UCS_LIBRARY)

if (UCX_FOUND)
  set(UCX_LIBRARIES ${UCP_LIBRARY} ${UCT_LIBRARY} ${UCM_LIBRARY} ${UCS_LIBRARY})
  message(STATUS "  UCX libraries: ${UCX_LIBRARIES})")
  mark_as_advanced(UCX_INCLUDE_DIR UCP_LIBRARY UCT_LIBRARY UCM_LIBRARY UCS_LIBRARY)

  foreach(UCX_MODULE UCP UCT UCM UCS)
    if (NOT TARGET UCX::${UCX_MODULE})
      add_library(UCX::${UCX_MODULE} UNKNOWN IMPORTED)
      set_target_properties(UCX::${UCX_MODULE} PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "C"
        IMPORTED_LOCATION "${${UCX_MODULE}_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${UCX_INCLUDE_DIR}")
    endif()
  endforeach()
endif ()
