#=============================================================================
# Copyright 2014 Kitware, Inc.
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

# The following options determine which libraries and components are searched
# for:
# GASNet_THREADING       - Threading mode to use, valid values are:
#                          seq (DEFAULT)
#                          par
#                          parsync
# GASNet_CONDUIT       - Communication conduit to use, valid values are:
#                          gemini
#                          ibv (DEFAULT)
#                          mpi
#                          mxm
#                          pami
#                          portals4
#                          shmem
#                          smp
#                          udp
#
# The following variables are produced by this module:
# GASNet_INCLUDE_DIRS        - The location of gasnet.h
# GASNet_LIBRARIES           - Necessary GASNet libraries to link to
# GASNet_COMPILE_DEFINITIONS - Additional compile definitions needed to use
#                              GASNet

if(NOT GASNet_FOUND)
  # Check the value of passed in options
  if(NOT GASNet_THREADING)
    set(GASNet_THREADING "seq" CACHE STRING "Parallel mode to use")
  elseif(NOT GASNet_THREADING MATCHES "^(seq|par|parmode)$")
    message(FATAL_ERROR "Invalid value for GASNet_THREADING, \"${GASNet_THREADING}\". Valid values are seq, par, or parsync")
  endif()
  if(NOT GASNet_CONDUIT)
    set(GASNet_CONDUIT "ibv" CACHE STRING "Communication conduit to use")
  elseif(NOT GASNet_THREADING MATCHES "^(gemini|ibv|mpi|mxm|pami|portals4|shmem|smp|udp)$")
    message(FATAL_ERROR "Invalid value for GASNet_THREADING, \"${GASNet_THREADING}\". Valid values are gemini, ibv, mpi, mxm, pami, portals4, shmem, smp, and udp")
  endif()

  find_path(GASNet_INCLUDE_DIR gasnet.h)

  # Make sure that all GASNet componets are found in the same install
  if(GASNet_INCLUDE_DIR)
    # Save the existing prefix options
    set(_CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH})
    set(_CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH})

    # Set new restrictive search paths
    get_filename_component(CMAKE_PREFIX_PATH "${GASNet_INCLUDE_DIR}" DIRECTORY)
    unset(CMAKE_LIBRARY_PATH)

    # Limit the search to the discovered prefix path
    set(_GASNet_FIND_OPTS
      NO_CMAKE_ENVIRONMENT_PATH
      NO_SYSTEM_ENVIRONMENT_PATH
      NO_CMAKE_SYSTEM_PATH
      NO_CMAKE_FIND_ROOT_PATH
    )

    # Look for the conduit specific header
    find_path(GASNet_CONDUIT_INCLUDE_DIR
      ${GASNet_CONDUIT}-conduit/gasnet_core.h
      ${GASNet_FIND_OPTS}
    )

    # Look for the conduit specific library
    find_library(GASNet_LIBRARY
      gasnet-${GASNet_CONDUIT}-${GASNet_THREADING}
      ${_GASNet_FIND_OPTS}
    )
  endif()

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(GASNet
    FOUND_VAR GASNet_FOUND
    REQUIRED_VARS GASNet_INCLUDE_DIR GASNet_CONDUIT_INCLUDE_DIR GASNet_LIBRARY
  )
  if(GASNet_FOUND)
    set(GASNet_INCLUDE_DIRS
      ${GASNet_INCLUDE_DIR}
      ${GASNet_CONDUIT_INCLUDE_DIR}
    )
    set(GASNet_COMPILE_DEFINITIONS GASNET_${GASNet_THREADING})
  endif()
endif()
