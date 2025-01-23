# Copyright 2025 Stanford University, NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

if(UNIX)
  find_package(PkgConfig)
  if(NOT PkgConfig_FOUND)
    return()
  endif()
  if(UCX_ROOT)
    set(_pkg_config $ENV{PKG_CONFIG_PATH} ${UCX_ROOT}/lib/pkgconfig)
    list(JOIN _pkg_config ':' _pkg_config_var)
    set(ENV{PKG_CONFIG_PATH} ${_pkg_config_var})
  endif()
  pkg_check_modules(ucx ucx QUIET)
endif()

if(NOT ucx_FOUND)
  find_path(
    ucx_INCLUDEDIR
    NAMES ucx
    HINTS ${UCX_ROOT}/include
  )
  find_library(
    ucx_LIBRARY
    NAMES ucx
    HINTS ${UCX_ROOT}/lib
  )
  if(ucx_LIBRARY)
    get_filename_component(ucx_LIBDIR ${ucx_LIBRARY} DIRECTORY)
    set(ucx_FOUND TRUE)
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ucx DEFAULT_MSG ucx_LIBDIR ucx_INCLUDEDIR)

mark_as_advanced(ucx_INCLUDEDIR ucx_LIBRARY)

if(ucx_FOUND)
  add_library(ucx::ucx SHARED IMPORTED)
  set_target_properties(
    ucx::ucx
    PROPERTIES IMPORTED_LOCATION
               ${ucx_LIBRARY}/${CMAKE_SHARED_LIBRARY_PREFIX}ucx${CMAKE_SHARED_LIBRARY_SUFFIX}
  )
  target_include_directories(ucx::ucx INTERFACE ${ucx_INCLUDEDIR})

  add_library(ucx::ucx_static STATIC IMPORTED)
  set_target_properties(
    ucx::ucx_static
    PROPERTIES IMPORTED_LOCATION
               ${ucx_LIBRARY_DIRS}/${CMAKE_STATIC_LIBRARY_PREFIX}ucx${CMAKE_STATIC_LIBRARY_SUFFIX}
  )
  target_include_directories(ucx::ucx_static INTERFACE ${ucx_INCLUDEDIR})
endif()
