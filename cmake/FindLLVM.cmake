#=============================================================================
# Copyright 2023 Kitware, Inc.
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

# This module produces the "LLVM" link target which carries with it all the
# necessary interface properties.  The following options guide the search
# GASNet backend configuration get's used:
#
# LLVM_CONFIG_EXECUTABLE - Path to llvm-config
#
# Once llvm-config is found, it will be used to query available components
#
if(NOT LLVM_FOUND AND NOT TARGET_LLVM)
  if(NOT LLVM_CONFIG_EXECUTABLE)
    # if an explicitly-versioned llvm-config (that we've tested with) is
    #  available, use that
    find_program(LLVM_CONFIG_EXECUTABLE NAMES llvm-config-9
                                              llvm-config-10
                                              llvm-config-11
                                              llvm-config-12
                                              llvm-config-13
                                              llvm-config-14
                                              llvm-config)
  endif(NOT LLVM_CONFIG_EXECUTABLE)
  if(LLVM_CONFIG_EXECUTABLE)
    # Check components
    execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --version
      OUTPUT_VARIABLE LLVM_VERSION
      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
    )
    execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --components
      OUTPUT_VARIABLE LLVM_AVAILABLE_COMPONENTS
      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
    )
    string(REGEX REPLACE "^ +" ""
      LLVM_AVAILABLE_COMPONENTS "${LLVM_AVAILABLE_COMPONENTS}"
    )
    string(REPLACE " " ";"
      LLVM_AVAILABLE_COMPONENTS "${LLVM_AVAILABLE_COMPONENTS}"
    )
    # for LLVM 3.6 and above, ignore jit in requested components
    if(${LLVM_VERSION} VERSION_GREATER 3.5.99)
      list(REMOVE_ITEM LLVM_FIND_COMPONENTS jit)
    endif()
    foreach(_component IN LISTS LLVM_FIND_COMPONENTS)
      list(FIND LLVM_AVAILABLE_COMPONENTS ${_component} C_IDX)
      if(C_IDX EQUAL -1)
        message(FATAL_ERROR "${_component} is not an available component for LLVM found at ${LLVM_CONFIG_EXECUTABLE}")
      endif()
    endforeach()

    # LLVM libs
    execute_process(
      COMMAND ${LLVM_CONFIG_EXECUTABLE} --libfiles ${LLVM_FIND_COMPONENTS}
      OUTPUT_VARIABLE _LLVM_LIBS
      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
    )
    # some versions of LLVM have a bug with libfiles where they produce
    #  /path/to/liblibfoo.so.so - do some string replaces to fix this
    string(REPLACE "liblib" "lib" _LLVM_LIBS ${_LLVM_LIBS})
    string(REPLACE ".so.so" ".so" _LLVM_LIBS ${_LLVM_LIBS})
    string(REPLACE " " ";" _LLVM_LIBS "${_LLVM_LIBS}")

    # LLVM system libs
    execute_process(
      COMMAND ${LLVM_CONFIG_EXECUTABLE} --system-libs
      OUTPUT_VARIABLE _LLVM_SYSTEM_LIBS
      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
    )
    # llvm-config --system-libs gives you all the libraries you might need for anything,
    #  which includes things we don't need, and might not be installed
    # for example, filter out libedit
    string(REPLACE "-ledit" "" _LLVM_SYSTEM_LIBS "${_LLVM_SYSTEM_LIBS}")

    string(REPLACE "-l" "" _LLVM_SYSTEM_LIBS "${_LLVM_SYSTEM_LIBS}")
    string(REPLACE " " ";" _LLVM_SYSTEM_LIBS "${_LLVM_SYSTEM_LIBS}")
    list(APPEND _LLVM_LIBS ${_LLVM_SYSTEM_LIBS})

    # LLVM includes
    execute_process(
      COMMAND ${LLVM_CONFIG_EXECUTABLE} --includedir
      OUTPUT_VARIABLE _LLVM_INCLUDE
      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
    )
  endif()

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(LLVM
    FOUND_VAR LLVM_FOUND
    REQUIRED_VARS LLVM_CONFIG_EXECUTABLE
  )
endif()

if(LLVM_FOUND AND NOT TARGET LLVM::LLVM)
  add_library(LLVM::LLVM INTERFACE IMPORTED)
  set_target_properties(LLVM::LLVM PROPERTIES
    INTERFACE_COMPILE_OPTIONS "--std=c++11"
    INTERFACE_INCLUDE_DIRECTORIES "${_LLVM_INCLUDE}"
    INTERFACE_LINK_LIBRARIES "${_LLVM_LIBS}"
  )
endif()
