#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#

set(legion_HEADERS
  field_tree.h
  garbage_collection.h
  interval_tree.h
  legion.h
  legion.inl
  legion_allocation.h
  legion_analysis.h
  legion_c.h
  legion_c_util.h
  legion_config.h
  legion_constraint.h
  legion_mapping.h
  legion_instances.h
  legion_ops.h
  legion_profiling.h
  legion_realm.h
  legion_spy.h
  legion_tasks.h
  legion_trace.h
  legion_types.h
  legion_utilities.h
  legion_views.h
  mapper_manager.h
  rectangle_set.h
  region_tree.h
  runtime.h
)

set(legion_SOURCES
  garbage_collection.cc
  legion.cc
  legion_analysis.cc
  legion_c.cc
  legion_constraint.cc
  legion_instances.cc
  legion_mapping.cc
  legion_ops.cc
  legion_profiling.cc
  legion_spy.cc
  legion_tasks.cc
  legion_trace.cc
  legion_views.cc
  mapper_manager.cc
  region_tree.cc
  runtime.cc
)

SET_SOURCE_FILES_PROPERTIES(legion.inl PROPERTIES HEADER_FILE_ONLY 1)

add_library(legion ${legion_SOURCES})
if(Legion_USE_GASNet)
   target_compile_definitions(legion PUBLIC USE_GASNET)
   target_link_libraries(legion PUBLIC GASNet::GASNet)
endif(Legion_USE_GASNet)

target_include_directories(legion
  PRIVATE
    ${CMAKE_SOURCE_DIR}/runtime
    ${CMAKE_SOURCE_DIR}/runtime/legion
    ${CMAKE_SOURCE_DIR}/runtime/mappers
    ${CMAKE_SOURCE_DIR}/runtime/realm)

#target_link_libraries( legion legion_mappers)

set ( PACKAGE_LIB_DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")
set ( PACKAGE_BIN_DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
set ( PACKAGE_INCLUDE_DESTINATION "${CMAKE_INSTALL_PREFIX}/include/legion")


install(TARGETS legion EXPORT legionTargets
  ARCHIVE DESTINATION ${PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${PACKAGE_BIN_DESTINATION}
)

foreach (file ${legion_HEADERS})
   #get_filename_component (dir ${file} DIRECTORY)
   install (FILES ${file} DESTINATION ${PACKAGE_INCLUDE_DESTINATION})
endforeach()

#----------------------------------------------------------------------------~-#
# vim: set tabstop=2 shiftwidth=2 expandtab :
#----------------------------------------------------------------------------~-#
