set(hls_dep $ENV{HLS_CONFIG} CACHE STRING "set hls config .cmake path for module")

get_filename_component(hls_dir ${hls_dep} DIRECTORY)
get_filename_component(hls_module ${hls_dep} NAME_WE)

list(APPEND CMAKE_MODULE_PATH ${hls_dir})

include(${hls_module})
link_directories(${XRT_LIB_DIR})
target_link_libraries(RealmRuntime PRIVATE ${hls_module})
install(TARGETS ${hls_module} EXPORT LegionTargets)
install(TARGETS miniglog EXPORT LegionTargets)
