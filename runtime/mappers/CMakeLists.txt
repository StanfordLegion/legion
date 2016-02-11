#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#

set(mappers_HEADERS
  default_mapper.h
  mapping_utilities.h
  shim_mapper.h
)

set(mappers_SOURCES
  default_mapper.cc
  mapping_utilities.cc
  shim_mapper.cc
)

add_library(legion_mappers ${mappers_SOURCES})
target_include_directories(legion_mappers
  PRIVATE
    ${CMAKE_SOURCE_DIR}/runtime
    ${CMAKE_SOURCE_DIR}/runtime/legion
    ${CMAKE_SOURCE_DIR}/runtime/realm)

set ( PACKAGE_LIB_DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")
set ( PACKAGE_BIN_DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
set ( PACKAGE_INCLUDE_DESTINATION "${CMAKE_INSTALL_PREFIX}/include")

install(TARGETS legion_mappers
  ARCHIVE DESTINATION ${PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${PACKAGE_BIN_DESTINATION}
)

foreach (file ${mappers_HEADERS})
   install (FILES ${file} DESTINATION ${PACKAGE_INCLUDE_DESTINATION})
endforeach()

#------------------------------------------------------------------------------#
# vim: set tabstop=2 shiftwidth=2 expandtab :
#------------------------------------------------------------------------------#
