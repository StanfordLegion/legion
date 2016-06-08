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

# This module produces the "GASNet" link target which carries with it all the
# necessary interface properties.  The following options determine which
# GASNet backend configuration get's used:
#
# GASNet_CONDUIT   - Communication conduit to use
# GASNet_THREADING - Threading mode to use
#
# Valid options for these are dependenent on the specific GASNet installation
#
# GASNet_ROOT_DIR  - Prefix to use when searching for GASNet.  If specified
#                    then this search path will be used exclusively and all
#                    others ignored.
# ENV{GASNET_ROOT} - Environment variable used to initialize the value of
#                    GASNet_ROOT_DIR if not already specified
#

macro(_GASNet_parse_conduit_and_threading_names
  MAKEFILE CONDUIT_LIST_VAR THREADING_LIST_VAR)
  get_filename_component(_BASE ${MAKEFILE} NAME_WE)
  string(REGEX MATCH "^([^\\-]*)-([^\\-]*)$" _BASE "${_BASE}")
  list(FIND ${CONDUIT_LIST_VAR} "${CMAKE_MATCH_1}" _I)
  if(_I EQUAL -1)
    list(APPEND ${CONDUIT_LIST_VAR} "${CMAKE_MATCH_1}")
  endif()
  list(FIND ${THREADING_LIST_VAR} "${CMAKE_MATCH_2}" _I)
  if(_I EQUAL -1)
    list(APPEND ${THREADING_LIST_VAR} "${CMAKE_MATCH_2}")
  endif()
endmacro()

macro(_GASNet_parse_conduit_makefile _GASNet_MAKEFILE _GASNet_THREADING)
  set(_TEMP_MAKEFILE ${CMAKE_CURRENT_BINARY_DIR}/FindGASNetParseConduitOpts.mak)
  if("${_GASNet_THREADING}" STREQUAL "parsync")
    file(WRITE ${_TEMP_MAKEFILE} "include ${_GASNet_MAKEFILE}")
  else()
    get_filename_component(MFDIR "${_GASNet_MAKEFILE}" DIRECTORY)
    file(WRITE ${_TEMP_MAKEFILE} "include ${_GASNet_MAKEFILE}
include ${MFDIR}/../gasnet_tools-${_GASNet_THREADING}.mak"
    )
  endif()
  file(APPEND ${_TEMP_MAKEFILE} "
gasnet-cc:
	@echo $(GASNET_CC)
gasnet-cflags:
	@echo $(GASNET_CPPFLAGS) $(GASNET_CFLAGS) $(GASNETTOOLS_CPPFLAGS) $(GASNETTOOLS_CFLAGS)
gasnet-cxx:
	@echo $(GASNET_CXX)
gasnet-cxxflags:
	@echo $(GASNET_CXXCPPFLAGS) $(GASNET_CXXFLAGS) $(GASNETTOOLS_CPPFLAGS) $(GASNETTOOLS_CXXFLAGS)
gasnet-ldflags:
	@echo $(GASNET_LDFLAGS) $(GASNETTOOLS_LDFLAGS)
gasnet-libs:
	@echo $(GASNET_LIBS) $(GASNETTOOLS_LIBS)"
  )
  find_program(GASNet_MAKE_PROGRAM NAMES gmake make smake)
  if(NOT GASNet_MAKE_PROGRAM)
    message(WARNING "Unable to locate compatible make for parsing GASNet makefile options")
  else()
    execute_process(
      COMMAND ${GASNet_MAKE_PROGRAM} -s -f ${_TEMP_MAKEFILE} gasnet-cflags
      OUTPUT_VARIABLE _GASNet_CFLAGS
      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
    )
    execute_process(
      COMMAND ${GASNet_MAKE_PROGRAM} -s -f ${_TEMP_MAKEFILE} gasnet-cxxflags
      OUTPUT_VARIABLE _GASNet_CXXFLAGS
      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
    )
    execute_process(
      COMMAND ${GASNet_MAKE_PROGRAM} -s -f ${_TEMP_MAKEFILE} gasnet-ldflags
      OUTPUT_VARIABLE _GASNet_LDFLAGS
      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
    )
    execute_process(
      COMMAND ${GASNet_MAKE_PROGRAM} -s -f ${_TEMP_MAKEFILE} gasnet-libs
      OUTPUT_VARIABLE _GASNet_LIBS
      ERROR_VARIABLE _GASNet_LIBS_ERROR
      OUTPUT_STRIP_TRAILING_WHITESPACE 
    )
    file(REMOVE ${_TEMP_MAKEFILE})
  endif()
endmacro()

macro(_GASNet_parse_flags INVAR FLAG OUTVAR)
  # Ignore the optimization flags
  string(REGEX REPLACE "-O[0-9]" "" INVAR2 "${${INVAR}}")
  string(REGEX MATCHALL "(^| +)${FLAG}([^ ]*)" OUTTMP "${INVAR2}")
  foreach(OPT IN LISTS OUTTMP)
    string(REGEX REPLACE "(^| +)${FLAG}([^ ]*)" "\\2" OPT "${OPT}")
    if(OPT STREQUAL "NDEBUG") # NDEBUG should not get propogated
      continue()
    endif()
    list(FIND ${OUTVAR} "${OPT}" _I)
    if(_I EQUAL -1)
      list(APPEND ${OUTVAR} "${OPT}")
    endif()
  endforeach()
endmacro()

function(_GASNet_create_component_target _GASNet_MAKEFILE COMPONENT_NAME
  COMPONENT_LIB)
  string(REGEX MATCH "^([^\\-]*)-([^\\-]*)$" COMPONENT_NAME "${COMPONENT_NAME}")
  _GASNet_parse_conduit_makefile(${_GASNet_MAKEFILE} ${CMAKE_MATCH_2})
  get_filename_component(LDIRS "${COMPONENT_LIB}" DIRECTORY)
  foreach(V _GASNet_CFLAGS _GASNet_CXXFLAGS _GASNet_LDFLAGS _GASNet_LIBS)
    _GASNet_parse_flags(${V} "-I" IDIRS)
    _GASNet_parse_flags(${V} "-D" DEFS)
    _GASNet_parse_flags(${V} "-L" LDIRS)
    _GASNet_parse_flags(${V} "-l" LIBS)
  endforeach()
  if(NOT LIBS)
    message(WARNING "Unable to find link libraries for gasnet-${COMPONENT_NAME}")
    return()
  endif()
  list(REMOVE_ITEM LIBS gasnet-${COMPONENT_NAME})

  foreach(L IN LISTS LIBS)
    find_library(GASNet_${L}_LIBRARY ${L} PATHS ${LDIRS})
    if(GASNet_${L}_LIBRARY)
      list(APPEND COMPONENT_DEPS "${GASNet_${L}_LIBRARY}")
    else()
      message(WARNING
        "Unable to locate GASNet ${COMPONENT_NAME} dependency ${L}"
      )
    endif()
    mark_as_advanced(GASNet_${L}_LIBRARY)
  endforeach()
  add_library(GASNet::${COMPONENT_NAME} UNKNOWN IMPORTED)
  set_target_properties(GASNet::${COMPONENT_NAME} PROPERTIES
    IMPORTED_LOCATION "${COMPONENT_LIB}"
  )
  if(DEFS)
    set_target_properties(GASNet::${COMPONENT_NAME} PROPERTIES
      INTERFACE_COMPILE_DEFINITIONS "${DEFS}"
    )
  endif()
  if(IDIRS)
    set_target_properties(GASNet::${COMPONENT_NAME} PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${IDIRS}"
    )
  endif()
  if(COMPONENT_DEPS)
    set_target_properties(GASNet::${COMPONENT_NAME} PROPERTIES
      INTERFACE_LINK_LIBRARIES "${COMPONENT_DEPS}"
    )
  endif()
endfunction()

if(NOT GASNet_FOUND AND NOT TARGET GASNet::GASNet)
  set(GASNet_ROOT_DIR "$ENV{GASNET_ROOT}" CACHE STRING "Root directory for GASNet")
  if(GASNet_ROOT_DIR)
    set(_GASNet_FIND_INCLUDE_OPTS PATHS ${GASNet_ROOT_DIR}/include NO_DEFAULT_PATH)
  endif()
  find_path(GASNet_INCLUDE_DIR gasnet.h ${_GASNet_FIND_INCLUDE_OPTS})

  # Make sure that all GASNet componets are found in the same install
  if(GASNet_INCLUDE_DIR)
    # Save the existing prefix options
    set(_CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH})
    set(_CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH})

    # Set new restrictive search paths
    get_filename_component(CMAKE_PREFIX_PATH "${GASNet_INCLUDE_DIR}" DIRECTORY)
    unset(CMAKE_LIBRARY_PATH)
    set(GASNet_ROOT_DIR ${CMAKE_PREFIX_PATH} CACHE STRING "Root directory for GASNet")

    # Limit the search to the discovered prefix path
    set(_GASNet_LIBRARY_FIND_OPTS
      NO_CMAKE_ENVIRONMENT_PATH
      NO_SYSTEM_ENVIRONMENT_PATH
      NO_CMAKE_SYSTEM_PATH
      NO_CMAKE_FIND_ROOT_PATH
    )

    # Look for the conduit specific headers
    set(GASNet_CONDUITS)
    set(GASNet_THREADING_OPTS)
    file(GLOB _GASNet_CONDUIT_MAKEFILES ${GASNet_INCLUDE_DIR}/*-conduit/*.mak)
    foreach(CMF IN LISTS _GASNet_CONDUIT_MAKEFILES)
      # Extract the component name from the makefile
      get_filename_component(_COMPONENT ${CMF} NAME_WE)

      # Seperate the filename components 
      _GASNet_parse_conduit_and_threading_names("${CMF}"
        GASNet_CONDUITS GASNet_THREADING_OPTS
      )

      # Look for the conduit specific library
      find_library(GASNet_${_COMPONENT}_LIBRARY gasnet-${_COMPONENT}
        ${_GASNet_FIND_LIBRARY_OPTS}
      )
      mark_as_advanced(GASNet_${_COMPONENT}_LIBRARY)

      # Create the component imported target
      _GASNet_create_component_target("${CMF}" ${_COMPONENT}
        "${GASNet_${_COMPONENT}_LIBRARY}"
      )
      if(NOT TARGET GASNet::${_COMPONENT})
        message(WARNING "Unable to create GASNet::${_COMPONENT} target")
      endif()
    endforeach()

    # Restore the existing prefix options
    set(CMAKE_PREFIX_PATH ${_CMAKE_PREFIX_PATH})
    set(CMAKE_LIBRARY_PATH ${_CMAKE_LIBRARY_PATH})
  endif()

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(GASNet
    FOUND_VAR GASNet_FOUND
    REQUIRED_VARS GASNet_INCLUDE_DIR GASNet_CONDUITS GASNet_THREADING_OPTS
  )
  if(GASNet_FOUND)
    set(GASNet_CONDUITS ${GASNet_CONDUITS} CACHE INTERNAL "")
    set(GASNet_THREADING_OPTS ${GASNet_THREADING_OPTS} CACHE INTERNAL "")
    message(STATUS "Found GASNet Conduits: ${GASNet_CONDUITS}")
    message(STATUS "Found GASNet Threading models: ${GASNet_THREADING_OPTS}")
  endif()
endif()

# If found, use the CONDUIT and THREADING options to determine which target to
# use
if(GASNet_FOUND AND NOT TARGET GASNet::GASNet)
  if(NOT GASNet_CONDUIT)
    list(GET GASNet_CONDUITS 0 GASNet_CONDUIT)
    set(GASNet_CONDUIT "${GASNet_CONDUIT}")
  endif()
  list(FIND GASNet_CONDUITS "${GASNet_CONDUIT}" _I)
  if(_I EQUAL -1)
    message(FATAL_ERROR "Invalid GASNet_CONDUIT setting.  Valid options are: ${GASNet_CONDUITS}")
  endif()

  if(NOT GASNet_THREADING)
    list(GET GASNet_THREADING_OPTS 0 GASNet_THREADING)
    set(GASNet_THREADING "${GASNet_THREADING}")
  endif()
  list(FIND GASNet_THREADING_OPTS "${GASNet_THREADING}" _I)
  if(_I EQUAL -1)
    message(FATAL_ERROR "Invalid GASNet_THREADING setting.  Valid options are: ${GASNet_THREADINGS}")
  endif()

  if(NOT TARGET GASNet::${GASNet_CONDUIT}-${GASNet_THREADING})
    message(FATAL_ERROR "Unable to use selected CONDUIT-THREADING combination: ${GASNet_CONDUIT}-${GASNet_THREADING}")
  endif()
  message(STATUS "GASNet: Using ${GASNet_CONDUIT}-${GASNet_THREADING}")
  add_library(GASNet::GASNet INTERFACE IMPORTED)
  set_target_properties(GASNet::GASNet PROPERTIES
    INTERFACE_COMPILE_DEFINITIONS GASNETI_BUG1389_WORKAROUND=1
    INTERFACE_LINK_LIBRARIES GASNet::${GASNet_CONDUIT}-${GASNet_THREADING}
  )
endif()
