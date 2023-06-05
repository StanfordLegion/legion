# Copyright (c) 2015-2020, Lawrence Livermore National Security, LLC.
#
# Try to find LIBDW headers and libraries.
#
# Usage of this module as follows:
#
#     find_package(LIBDW)
#
# Variables used by this module, they can change the default behaviour and need
# to be set before calling find_package:
#
#  LIBDW_PATH          Set this variable to the root installation of
#                      libdw if the module has problems finding the
#                      proper installation path.
#
# Variables defined by this module:
#
#  LIBDW_FOUND              System has LIBDW libraries and headers
#  LIBDW_LIBRARY          The LIBDW library
#  LIBDW_INCLUDE_DIR       The location of LIBDW headers

if(DEFINED ENV{LIBDW_PATH})
  set(LIBDW_PATH $ENV{LIBDW_PATH})
endif()

if (LIBDW_INCLUDE_DIR AND LIBDW_LIBRARY)
  set(LIBDW_FIND_QUIETLY true)
endif()

find_path(LIBDW_INCLUDE_DIR elfutils/libdwfl.h
  HINTS ${LIBDW_PATH}/include
)

find_library(LIBDW_LIBRARY 
  NAMES dw
  HINTS ${LIBDW_PATH}/lib
)

if (LIBDW_INCLUDE_DIR AND LIBDW_LIBRARY)
  message(STATUS "Found elfutils/libdwfl.h header: ${LIBDW_INCLUDE_DIR}")
  message(STATUS "Found libdw library: ${LIBDW_LIBRARY}")
  set(LIBDW_FOUND TRUE)

  mark_as_advanced(
    LIBDW_INCLUDE_DIR
    LIBDW_LIBRARY
  )
else()
  set(LIBDW_FOUND FALSE)
  message(FATAL_ERROR "Can not find libdw in: ${LIBDW_PATH}")
endif()