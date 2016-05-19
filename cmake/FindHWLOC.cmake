#=============================================================================
# Copyright 2016 Kitware, Inc.
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
#=============================================================================

# This module produces the "HWLOC" link target which carries with it all the
# necessary interface properties.  If the HWLOC_ROOT CMake or HWLOC wnvironment
# variable are present then they are used to guide the search
#
if(NOT HWLOC_FOUND AND NOT TARGET HWLOC)
  if(NOT HWLOC_ROOT AND ENV{HWLOC})
    set(HWLOC_ROOT $ENV{HWLOC})
  endif()
  if(HWLOC_ROOT)
    # Save the existing prefix options
    set(_CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH})
    set(_CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH})
    set(_HWLOC_FIND_OPTS
      NO_CMAKE_ENVIRONMENT_PATH
      NO_SYSTEM_ENVIRONMENT_PATH
      NO_CMAKE_SYSTEM_PATH
      NO_CMAKE_FIND_ROOT_PATH
    )
  endif()

  find_path(HWLOC_INCLUDE_DIR hwloc.h ${_HWLOC_FIND_OPTS})
  find_library(HWLOC_LIBRARY hwloc ${_HWLOC_FIND_OPTS})

  if(HWLOC_ROOT)
    # Restore the existing prefix options
    set(CMAKE_PREFIX_PATH ${_CMAKE_PREFIX_PATH})
    set(CMAKE_LIBRARY_PATH ${_CMAKE_LIBRARY_PATH})
  endif()

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(HWLOC
    FOUND_VAR HWLOC_FOUND
    REQUIRED_VARS HWLOC_INCLUDE_DIR HWLOC_LIBRARY
  )
endif()

if(HWLOC_FOUND AND NOT TARGET HWLOC)
  if(HWLOC_LIBRARY MATCHES ".*${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(HWLOC_LIBTYPE SHARED)
  else()
    set(HWLOC_LIBTYPE STATIC)
  endif()
  add_library(HWLOC ${HWLOC_LIBTYPE} IMPORTED)
  set_target_properties(HWLOC PROPERTIES
    IMPORTED_LOCATION ${HWLOC_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${HWLOC_INCLUDE_DIR}
  )
endif()
