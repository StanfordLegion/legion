add_subdirectory(realm)
add_subdirectory(legion)
add_subdirectory(mappers)


set(runtime_HEADERS
   lowlevel_config.h
   common.h
   accessor.h
   activemsg.h
   arrays.h
   atomics.h
   lowlevel_dma.h
   lowlevel.h
   utilities.h
)

set ( PACKAGE_INCLUDE_DESTINATION "${CMAKE_INSTALL_PREFIX}/include")

foreach (file ${runtime_HEADERS})
   install (FILES ${file} DESTINATION ${PACKAGE_INCLUDE_DESTINATION})
endforeach()

