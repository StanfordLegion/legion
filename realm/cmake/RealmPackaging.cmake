# Copyright 2025 Stanford University, NVIDIA Corporation, Los Alamos National Laboratory
# SPDX-License-Identifier: Apache-2.0
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

if(BUILD_SHARED_LIBS)
  set(lib_type "shared")
else()
  set(lib_type "static")
endif()

install(
  TARGETS Realm
  EXPORT Realm_targets
  RUNTIME COMPONENT Realm_runtime
  LIBRARY COMPONENT Realm_runtime
  ARCHIVE COMPONENT Realm_devel
  PUBLIC_HEADER COMPONENT Realm_devel DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/realm"
  INCLUDES
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/realm"
)

# Install the gdb auto-load script next to the library so it can be
# `source`'d manually if auto-load safe-path declines the embedded copy.
# Only meaningful when debug info is present, so restrict to Debug and
# RelWithDebInfo configurations.  ELF only -- the script is a no-op on
# macOS/Windows anyway.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "FreeBSD")
  install(
    FILES "${REALM_SOURCE_DIR}/realm-gdb.py"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    COMPONENT Realm_runtime
    CONFIGURATIONS Debug RelWithDebInfo
  )
endif()

# Install the realm_kokkos as well when needed
if(REALM_USE_KOKKOS)
  install(
    TARGETS realm_kokkos
    EXPORT Realm_targets
    RUNTIME COMPONENT Realm_runtime
    LIBRARY COMPONENT Realm_runtime
    ARCHIVE COMPONENT Realm_devel
    PUBLIC_HEADER COMPONENT Realm_devel DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/realm"
    INCLUDES
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/realm"
  )
endif()

# Install the realm_gex_wrapper as well if we have to link directly to it
if(REALM_INSTALL_GASNETEX_WRAPPER)
  install(
    TARGETS realm_gex_wrapper
    EXPORT Realm_targets
    RUNTIME COMPONENT Realm_runtime
    LIBRARY COMPONENT Realm_runtime
    ARCHIVE COMPONENT Realm_devel
  )
endif()

if(REALM_INSTALL_UCX_BOOTSTRAPS)
  install(
    TARGETS ${UCX_BACKENDS}
    EXPORT Realm_targets
    RUNTIME COMPONENT Realm_runtime DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY COMPONENT Realm_runtime DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  )
endif()

# TODO(cperry): Separate out public headers from internal ones
# Unfortunately public and internal headers are all mixed up, so we need to glob together
# all the header files in the source directory and install them.  Ideally we would just
# add the public headers to a cmake FILE_SET
install(
  FILES "${REALM_SOURCE_DIR}/../realm.h"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/"
  COMPONENT Realm_devel
)
install(
  DIRECTORY "${REALM_SOURCE_DIR}/../hip_cuda_compat"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/"
  COMPONENT Realm_devel
)
install(
  DIRECTORY "${REALM_SOURCE_DIR}/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/realm"
  COMPONENT Realm_devel
  FILES_MATCHING
  PATTERN "*.h"
)
install(
  DIRECTORY "${REALM_SOURCE_DIR}/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/realm"
  COMPONENT Realm_devel
  FILES_MATCHING
  PATTERN "*.inl"
)
install(
  DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  COMPONENT Realm_devel
)

install(
  DIRECTORY examples/
  DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/realm/examples"
  COMPONENT Realm_samples
  PATTERN "examples/CMakeLists.txt" EXCLUDE
)

install(
  DIRECTORY tutorials/
  DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/realm/tutorials"
  COMPONENT Realm_samples
  PATTERN "tutorials/CMakeLists.txt" EXCLUDE
)

#region pkgconfig and supporting cmake files
write_basic_package_version_file(
  RealmConfigVersion.cmake
  VERSION ${Realm_SHORT_VERSION}
  COMPATIBILITY SameMinorVersion
)

# Get a list of pkgconf dependencies
if(NOT BUILD_SHARED_LIBS)
  list(
    TRANSFORM REALM_STATIC_DEPENDS
    TOLOWER
    OUTPUT_VARIABLE REALM_PKGCONF_REQUIRES
  )
  string(REPLACE ";" " " REALM_PKGCONF_REQUIRES "${REALM_PKGCONF_REQUIRES}")
endif()

# Setup pkgconfig module
function(get_pkgconfig_path path out)
  if(IS_ABSOLUTE "${path}")
    # Normalize first because CMake normalizes the install prefix.
    cmake_path(SET path NORMALIZE "${path}")
    # Check if path is contained in the install prefix.
    string(FIND "${path}" "${CMAKE_INSTALL_PREFIX}" path_in_prefix)
    if(${path_in_prefix} EQUAL 0)
      # Absolute path inside prefix, compute a relative path for relocation.
      file(RELATIVE_PATH relative_path "${CMAKE_INSTALL_PREFIX}" "${path}")
      set(${out} "\${prefix}/${relative_path}" PARENT_SCOPE)
    else()
      # Path is absolute but outside prefix, cannot relocate it.
      set(${out} "${path}" PARENT_SCOPE)
    endif()
  else()
    set(${out} "\${prefix}/${path}" PARENT_SCOPE)
  endif()
endfunction()

get_pkgconfig_path(${CMAKE_INSTALL_LIBDIR} PKGCONFIG_CMAKE_INSTALL_LIBDIR)
get_pkgconfig_path(${CMAKE_INSTALL_INCLUDEDIR} PKGCONFIG_CMAKE_INSTALL_INCLUDEDIR)
get_pkgconfig_path(${CMAKE_INSTALL_DATAROOTDIR} PKGCONFIG_CMAKE_INSTALL_DATAROOTDIR)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/realm.pc.in realm.pc
  INSTALL_DESTINATION "${CMAKE_INSTALL_ROOTDATADIR}/pkgconfig"
)

# Set up RealmConfig file.
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/RealmConfig.cmake.in" "RealmConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/realm"
)

install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/realm.pc"
  DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig"
  COMPONENT Realm_devel
)
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/RealmConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/RealmConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/realm"
  COMPONENT Realm_devel
)

# Make sure to install all the find modules as a last resort for RealmConfig to find them
install(
  FILES "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindGASNet.cmake"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindHWLOC.cmake"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindLLVM.cmake"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindPapi.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/realm"
  COMPONENT Realm_devel
)

install(
  EXPORT Realm_targets
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/realm"
  NAMESPACE Realm::
  FILE Realm-${lib_type}-targets.cmake
  COMPONENT Realm_devel
)

export(PACKAGE Realm)
export(EXPORT Realm_targets
       FILE "${CMAKE_CURRENT_BINARY_DIR}/Realm-${lib_type}-targets.cmake"
       NAMESPACE Realm::)
#endregion

#region Documentation
if(REALM_BUILD_DOCS)
  install(
    DIRECTORY "${CMAKE_BINARY_DIR}/docs/html/"
    DESTINATION "${CMAKE_INSTALL_DOCDIR}/realm-${REALM_VERSION}"
    COMPONENT Realm_doc
  )
  install(
    FILES "${CMAKE_CURRENT_SOURCE_DIR}/doxygen/selectversion.js"
          "${CMAKE_CURRENT_SOURCE_DIR}/doxygen/dropdown.css"
    DESTINATION "${CMAKE_INSTALL_DOCDIR}/"
    COMPONENT Realm_doc
  )
endif()
#endregion

#region Packaging
set(CPACK_DEBIAN_PACKAGE_DEBUG ON)
set(CPACK_PACKAGE_NAME "realm")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "${PROJECT_DESCRIPTION}")
set(CPACK_STRIP_FILES YES)
set(CPACK_INSTALL_DEFAULT_DIRECTORY_PERMISSIONS
    OWNER_READ
    OWNER_WRITE
    OWNER_EXECUTE
    GROUP_READ
    GROUP_EXECUTE
    WORLD_READ
    WORLD_EXECUTE
)
set(CPACK_PACKAGE_CONTACT "mike@lightsighter.org")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")
set(CPACK_SOURCE_IGNORE_FILES
    "/\\\\.git/"
    "/\\\\.github/"
    "/\\\\.vscode/"
    "/\\\\.swp$"
    "/\\\\.gitignore$"
    "/\\\\.#"
    "/build/"
    "/install/"
)
set(CPACK_COMPONENT_UNSPECIFIED_HIDDEN YES)
set(CPACK_COMPONENT_UNSPECIFIED_REQUIRED YES)
set(CPACK_COMPONENTS_ALL Realm_runtime Realm_devel Realm_samples)

set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
set(CPACK_DEBIAN_PACKAGE_DEPENDS "")
set(CPACK_DEBIAN_ENABLE_COMPONENT_DEPENDS YES)
set(CPACK_DEBIAN_PACKAGE_GENERATE_SHLIBS YES)
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS YES)
set(CPACK_DEB_COMPONENT_INSTALL YES)
set(CPACK_DEBIAN_ENABLE_COMPONENT_DEPENDS YES)
set(CPACK_DEBIAN_PACKAGE_GENERATE_SHLIBS_POLICY ">=")
if(REALM_BUILD_DOCS)
  list(APPEND CPACK_COMPONENTS_ALL Realm_doc)
endif()

# Snap the version for the source package and add it to the source package via the custom CPack script
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/version/VERSION" "${REALM_VERSION}")
# Also add this to the installation package for systems that don't support either cmake nor pkg-config (like osx)
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/version/VERSION"
  DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/realm"
  COMPONENT Realm_devel
)

set(CPACK_INSTALL_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/CPack.cmake")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CPack.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/CPack.cmake"
  @ONLY
)

include(CPack)
include(InstallRequiredSystemLibraries)

cpack_add_component(
  Realm_runtime
  DISPLAY_NAME "Runtime"
  DESCRIPTION "Runtime dependencies and libraries components"
)
cpack_add_component(
  Realm_devel
  DISPLAY_NAME "Development"
  DESCRIPTION "Header files and configuration scripts"
  DEPENDS Realm_runtime
)
cpack_add_component(
  Realm_samples
  DISPLAY_NAME "Samples"
  DESCRIPTION "Tutorials and example application sources"
  DEPENDS Realm_devel
)
if(REALM_BUILD_DOCS AND DOXYGEN_FOUND)
  cpack_add_component(
    Realm_doc
    DISPLAY_NAME "Documentation"
    DESCRIPTION "Doxygen documentation"
  )
endif()
#endregion
