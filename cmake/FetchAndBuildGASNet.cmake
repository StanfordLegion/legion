#------------------------------------------------------------------------------#
# Copyright 2024 Stanford, NVIDIA Corp..
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

# This cmake module retrieves gasnetex source and builds it based on the
# following configuration variables:
#
# GASNet_CONDUIT:
# GASNet_System:
# GASNet_GITREPO:
# GASNet_GITREF:
# GASNet_CONFIGURE_ARGS:


if (NOT GASNet_CONDUIT)
  message(FATAL_ERROR "GASNet_CONDUIT not specified, please set a select a conduit to build GASNet with")
endif()
if (NOT GASNet_GITREPO)
  set(GASNet_GITREPO "https://github.com/StanfordLegion/gasnet.git"
    CACHE STRING "URL for cloing StanfordLegion/gasnet repository")
endif()
if (NOT GASNet_GITREF)
  set(GASNet_GITREF
    "0fb5a0556e76d1988ea5b59df2789a25d4e1ad99" # master as of 2024-03-11
    CACHE STRING "Branch/tag/commit to use from StanfordLegion/gasnet repository")
endif()
if (NOT GASNet_VERSION)
  set(GASNet_VERSION "" CACHE STRING "Override GASNet version to build")
endif()
if (NOT GASNet_CONFIGURE_ARGS)
  set(GASNet_CONFIGURE_ARGS "" CACHE STRING "Extra configuration arguments for GASNet")
endif()

file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/embed-gasnet)
set(GASNet_SOURCE_DIR ${PROJECT_BINARY_DIR}/embed-gasnet/source)
set(GASNet_BUILD_DIR ${PROJECT_BINARY_DIR}/embed-gasnet/build)
set(GASNet_INSTALL_DIR ${PROJECT_BINARY_DIR}/embed-gasnet/install)
set(GASNet_BUILD_OUTPUT ${PROJECT_BINARY_DIR}/embed-gasnet/build.log)
set(GASNet_CONFIG_FILE ${PROJECT_BINARY_DIR}/embed-gasnet/config.txt)

if (GASNet_BUILD_SHARED)
  set(GASNET_CFLAGS -fPIC)
  set(GASNET_CXXFLAGS -fPIC)
else()
  set(GASNET_CFLAGS "")
  set(GASNET_CXXFLAGS "")
endif()

list(APPEND GASNet_CONFIG_SETTINGS "LEGION_GASNET_CONDUIT=${GASNet_CONDUIT}" "LEGION_GASNET_SYSTEM=${GASNet_SYSTEM}" "GASNET_EXTRA_CONFIGURE_ARGS=${GASNet_CONFIGURE_ARGS}" "GASNET_CFLAGS=${GASNET_CFLAGS}" "GASNET_CXXFLAGS=${GASNET_CXXFLAGS}")
if(GASNet_VERSION)
  # make the source directory version-specific
  set(GASNet_SOURCE_DIR ${PROJECT_BINARY_DIR}/embed-gasnet/${GASNet_VERSION})
  list(APPEND GASNet_CONFIG_SETTINGS "GASNET_VERSION=${GASNet_VERSION}")
endif()

if(EXISTS ${GASNet_CONFIG_FILE})
  file(READ ${GASNet_CONFIG_FILE} PREV_CONFIG_SETTINGS)
  if("${GASNet_CONFIG_SETTINGS}" STREQUAL "${PREV_CONFIG_SETTINGS}")
    # configs match - no build needed
    set(GASNET_BUILD_NEEDED OFF)
  else()
    message(STATUS "Embedded GASNet configuration has changed - rebuilding...")
    file(REMOVE_RECURSE ${GASNet_BUILD_DIR} ${GASNet_INSTALL_DIR} ${GASNet_CONFIG_FILE})
    # clear any cached paths/conduits/etc.
    unset(GASNet_INCLUDE_DIR CACHE)
    unset(GASNet_CONDUITS CACHE)
    unset(GASNet_THREADING_OPTS CACHE)
    set(GASNET_BUILD_NEEDED ON)
  endif()
else()
  message(STATUS "Configuring and building embedded GASNet...")
  set(GASNET_BUILD_NEEDED ON)
endif()

if(GASNET_BUILD_NEEDED)
  if(GASNet_LOCALSRC)
  # relative paths are relative to the _build_ dir - this does the
  #  intuitive thing when you do:
  #     cmake ... -DGASNet_LOCALSRC=relative/to/cwd (source_dir)
  #  but is a little nonobvious when you type:
  #     cmake -S (source_dir) -B (build_dir) ... -DGASNet_LOCALSRC=still/relative/to/build_dir
  get_filename_component(
    EMBEDDED_GASNET_SRC
    "${GASNet_LOCALSRC}"
    ABSOLUTE BASE_DIR ${PROJECT_BINARY_DIR})
  else()
    include(FetchContent)
    FetchContent_Declare(embed-gasnet
      GIT_REPOSITORY ${GASNet_GITREPO}
      GIT_TAG        ${GASNet_GITREF}
      )
    message(STATUS "Downloading StanfordLegion/gasnet repo from: ${GASNet_GITREPO}")
    FetchContent_Populate(embed-gasnet)
    set(EMBEDDED_GASNET_SRC "${embed-gasnet_SOURCE_DIR}")
  endif()
  execute_process(
    COMMAND make -C ${EMBEDDED_GASNET_SRC} GASNET_SOURCE_DIR=${GASNet_SOURCE_DIR} GASNET_BUILD_DIR=${GASNet_BUILD_DIR} GASNET_INSTALL_DIR=${GASNet_INSTALL_DIR} ${GASNet_CONFIG_SETTINGS}
    RESULT_VARIABLE GASNET_BUILD_STATUS
    OUTPUT_FILE ${GASNet_BUILD_OUTPUT}
    ERROR_FILE ${GASNet_BUILD_OUTPUT}
  )
  if(GASNET_BUILD_STATUS)
    message(FATAL_ERROR "GASNet build result = ${GASNET_BUILD_STATUS} - see ${GASNet_BUILD_OUTPUT} for more details")
  endif()
  set(GASNet_ROOT_DIR ${GASNet_INSTALL_DIR} CACHE STRING "Root directory for GASNet" FORCE)
endif()
