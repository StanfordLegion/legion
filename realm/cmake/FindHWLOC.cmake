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

if(HWLOC_FOUND)
  return()
endif()

if(UNIX)
  find_package(PkgConfig)
  if(NOT PkgConfig_FOUND)
    return()
  endif()
  if(HWLOC_ROOT)
    set(_pkg_config $ENV{PKG_CONFIG_PATH} ${HWLOC_ROOT}/lib/pkgconfig)
    list(JOIN _pkg_config ':' _pkg_config_var)
    set(ENV{PKG_CONFIG_PATH} ${_pkg_config_var})
  endif()
  pkg_check_modules(HWLOC hwloc IMPORTED_TARGET)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HWLOC DEFAULT_MSG HWLOC_LIBDIR HWLOC_INCLUDEDIR)

mark_as_advanced(HWLOC_INCLUDEDIR HWLOC_LIBRARY)
add_library(HWLOC::HWLOC ALIAS PkgConfig::HWLOC)
