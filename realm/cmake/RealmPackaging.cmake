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

if(BUILD_SHARED_LIBS)
  set(lib_type "shared")
else()
  set(lib_type "static")
endif()

get_property(REALM_COMPILE_DEFINITIONS TARGET realm_obj PROPERTY COMPILE_DEFINITIONS)

# Adding realm_obj doesn't really do anything there, it won't install object
# files or anything, it just makes it so it's exported since it's referenced
# by the realm target
install(
  TARGETS realm realm_obj
  EXPORT Realm_targets
  RUNTIME COMPONENT Realm_runtime
  LIBRARY COMPONENT Realm_runtime
  ARCHIVE COMPONENT Realm_devel
  PUBLIC_HEADER COMPONENT Realm_devel DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/realm"
  INCLUDES
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/realm"
)

# TODO(cperry): Separate out public headers from internal ones
# Unfortunately public and internal headers are all mixed up, so we need to glob together
# all the header files in the source directory and install them.  Ideally we would just
# add the public headers to a cmake FILE_SET
install(
  DIRECTORY "${REALM_SOURCE_DIR}/" "${CMAKE_CURRENT_BINARY_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/realm"
  COMPONENT Realm_devel
  FILES_MATCHING
  PATTERN "*.h"
)
install(
  DIRECTORY "${REALM_SOURCE_DIR}/" "${CMAKE_CURRENT_BINARY_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/realm"
  COMPONENT Realm_devel
  FILES_MATCHING
  PATTERN "*.inl"
)

install(
  DIRECTORY examples/
  DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/realm/examples"
  COMPONENT Realm_samples
)

install(
  DIRECTORY tutorials/
  DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/realm/tutorials"
  COMPONENT Realm_samples
)

#region pkgconfig and supporting cmake files
write_basic_package_version_file(
  RealmConfigVersion.cmake
  VERSION ${REALM_SHORT_VERSION}
  COMPATIBILITY SameMinorVersion
)

# Get a list of pkgconf dependencies
list(
  TRANSFORM REALM_STATIC_DEPENDS
  TOLOWER
  OUTPUT_VARIABLE REALM_PKGCONF_REQUIRES
)
string(REPLACE ";" " " REALM_PKGCONF_REQUIRES "${REALM_PKGCONF_REQUIRES}")

# Setup pkgconfig module
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/realm.pc.in realm.pc
  INSTALL_DESTINATION "${CMAKE_INSTALL_ROOTDATADIR}/pkgconfig"
  PATH_VARS CMAKE_INSTALL_PREFIX CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_INCLUDEDIR
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
  FILES "${CMAKE_CURRENT_SOURCE_DIR}/FindGASNet.cmake" "${CMAKE_CURRENT_BINARY_DIR}/FindHWLOC.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/FindLLVM.cmake" "${CMAKE_CURRENT_BINARY_DIR}/FindPapi.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/Finducx.cmake"
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
#endregion

#region Documentation
if(REALM_BUILD_DOCS)
  install(
    DIRECTORY "${CMAKE_BINARY_DIR}/docs/html/"
    DESTINATION "${CMAKE_INSTALL_DOCDIR}/realm/realm-${REALM_VERSION}"
    COMPONENT Realm_doc
  )
  install(
    FILES "${CMAKE_CURRENT_SOURCE_DIR}/doxygen/selectversion.js"
          "${CMAKE_CURRENT_SOURCE_DIR}/doxygen/dropdown.css"
    DESTINATION "${CMAKE_INSTALL_DOCDIR}/realm/"
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
