#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#

set(mappers_HEADERS
  debug_mapper.h
  default_mapper.h
  mapping_utilities.h
  replay_mapper.h
  shim_mapper.h
  test_mapper.h
)

set(mappers_SOURCES
  debug_mapper.cc
  default_mapper.cc
  mapping_utilities.cc
  replay_mapper.cc
  shim_mapper.cc
  test_mapper.cc
)

add_library(legion_mappers ${mappers_SOURCES})
target_include_directories(legion_mappers
  PRIVATE
    ${CMAKE_SOURCE_DIR}/runtime
    ${CMAKE_SOURCE_DIR}/runtime/legion
    ${CMAKE_SOURCE_DIR}/runtime/realm)

set ( PACKAGE_LIB_DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")
set ( PACKAGE_BIN_DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
set ( PACKAGE_INCLUDE_DESTINATION "${CMAKE_INSTALL_PREFIX}/include/mappers")

install(TARGETS legion_mappers EXPORT legion_mappersTargets
  ARCHIVE DESTINATION ${PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${PACKAGE_BIN_DESTINATION}
)
if(Legion_USE_GASNet)
   target_compile_definitions(legion_mappers PUBLIC USE_GASNET)
   target_link_libraries(legion_mappers PUBLIC GASNet::GASNet)
endif(Legion_USE_GASNet)


foreach (file ${mappers_HEADERS})
   install (FILES ${file} DESTINATION ${PACKAGE_INCLUDE_DESTINATION})
endforeach()

#------------------------------------------------------------------------------#
# vim: set tabstop=2 shiftwidth=2 expandtab :
#------------------------------------------------------------------------------#
