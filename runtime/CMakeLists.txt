#------------------------------------------------------------------------------#
# Copyright 2022 Kitware, Inc.
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

# Install headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.h"
)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.inl"
)

# Realm runtime
list(APPEND REALM_SRC
  realm/realm_c.h
  realm/transfer/lowlevel_disk.cc
  realm/transfer/channel.h                 realm/transfer/channel.cc
  realm/transfer/channel_disk.h            realm/transfer/channel_disk.cc
  realm/transfer/transfer.h                realm/transfer/transfer.cc
  realm/transfer/lowlevel_dma.h            realm/transfer/lowlevel_dma.cc
  realm/transfer/ib_memory.h               realm/transfer/ib_memory.cc
  realm/deppart/byfield.h                  realm/deppart/byfield.cc
  realm/deppart/deppart_config.h
  realm/deppart/image.h
  realm/deppart/inst_helper.h
  realm/deppart/partitions.h               realm/deppart/partitions.cc
  realm/deppart/preimage.h
  realm/deppart/rectlist.h
  realm/deppart/rectlist.inl
  realm/deppart/setops.h                   realm/deppart/setops.cc
  realm/deppart/sparsity_impl.h            realm/deppart/sparsity_impl.cc
  realm/deppart/sparsity_impl.inl
  realm/atomics.h           realm/atomics.inl
  realm/bgwork.h            realm/bgwork.cc
  realm/event_impl.h        realm/event_impl.cc
  realm/event_impl.inl
  realm/faults.h            realm/faults.cc
  realm/faults.inl
  realm/idx_impl.h          realm/idx_impl.cc
  realm/inst_impl.h         realm/inst_impl.cc
  realm/inst_layout.h       realm/inst_layout.inl
  realm/inst_layout.cc
  realm/interval_tree.h     realm/interval_tree.inl
  realm/machine_impl.h      realm/machine_impl.cc
  realm/mem_impl.h          realm/mem_impl.cc
  realm/metadata.h          realm/metadata.cc
  realm/module.h            realm/module.cc
  realm/nodeset.h           realm/nodeset.inl      realm/nodeset.cc
  realm/numa/numa_module.h  realm/numa/numa_module.cc
  realm/numa/numasysif.h    realm/numa/numasysif.cc
  realm/operation.h         realm/operation.cc
  realm/operation.inl
  realm/proc_impl.h         realm/proc_impl.cc
  realm/procset/procset_module.h realm/procset/procset_module.cc
  realm/rsrv_impl.h         realm/rsrv_impl.cc
  realm/runtime_impl.h      realm/runtime_impl.cc
  realm/sampling_impl.h     realm/sampling_impl.cc
  realm/subgraph_impl.h     realm/subgraph_impl.cc
  realm/tasks.h             realm/tasks.cc
  realm/threads.h           realm/threads.cc
  realm/threads.inl
)

if(Legion_USE_CUDA)
  list(APPEND REALM_SRC
    realm/cuda/cuda_module.h    realm/cuda/cuda_module.cc
    realm/cuda/cuda_internal.h  realm/cuda/cuda_internal.cc
    realm/cuda/cuda_access.h    realm/cuda/cuda_access.cc
  )
  if(REALM_USE_CUDART_HIJACK)
    list(APPEND REALM_SRC
      realm/cuda/cudart_hijack.h
    )
  endif()
endif()

if(Legion_USE_HIP)
  list(APPEND REALM_SRC
    realm/hip/hip_module.h    realm/hip/hip_module.cc
    realm/hip/hip_internal.h  realm/hip/hip_internal.cc
    realm/hip/hip_access.h    realm/hip/hip_access.cc
		realm/hip/hiphijack_api.h
  )
  if(REALM_USE_HIP_HIJACK)
    list(APPEND REALM_SRC
      realm/hip/hip_hijack.h
    )
  endif()
endif()

if(Legion_USE_LLVM)
  list(APPEND REALM_SRC
    realm/llvmjit/llvmjit.h
    realm/llvmjit/llvmjit.inl
    realm/llvmjit/llvmjit_internal.h  realm/llvmjit/llvmjit_internal.cc
    realm/llvmjit/llvmjit_module.h    realm/llvmjit/llvmjit_module.cc
  )
endif()

if(Legion_USE_HDF5)
  list(APPEND REALM_SRC
    realm/hdf5/hdf5_module.h realm/hdf5/hdf5_module.cc
    realm/hdf5/hdf5_internal.h realm/hdf5/hdf5_internal.cc
    realm/hdf5/hdf5_access.h   realm/hdf5/hdf5_access.inl
    realm/hdf5/hdf5_access.cc
  )
endif()

if (Legion_USE_OpenMP)
  list(APPEND REALM_SRC
    realm/openmp/openmp_module.h realm/openmp/openmp_module.cc
    realm/openmp/openmp_internal.h
  )
  if(NOT Legion_OpenMP_SYSTEM_RUNTIME)
    list(APPEND REALM_SRC
      realm/openmp/openmp_threadpool.h realm/openmp/openmp_threadpool.cc
      realm/openmp/openmp_threadpool.inl
      realm/openmp/openmp_api.cc
    )
  endif()
endif()

if (Legion_USE_Python)
  list(APPEND REALM_SRC
    realm/python/python_module.h realm/python/python_module.cc
    realm/python/python_source.h realm/python/python_source.cc
    realm/python/python_source.inl
    realm/python/python_internal.h
  )
endif()

if(REALM_USE_GASNET1)
  list(APPEND REALM_SRC
    realm/gasnet1/gasnet1_module.h
    realm/gasnet1/gasnet1_module.cc
    realm/gasnet1/gasnetmsg.h
    realm/gasnet1/gasnetmsg.cc
  )
endif()

if(REALM_USE_GASNETEX)
  list(APPEND REALM_SRC
    realm/gasnetex/gasnetex_module.h
    realm/gasnetex/gasnetex_module.cc
    realm/gasnetex/gasnetex_internal.h
    realm/gasnetex/gasnetex_internal.cc
    realm/gasnetex/gasnetex_handlers.h
    realm/gasnetex/gasnetex_handlers.cc
  )
endif()

if(REALM_USE_MPI)
  list(APPEND REALM_SRC
    realm/mpi/mpi_module.h
    realm/mpi/mpi_module.cc
    realm/mpi/am_mpi.h
    realm/mpi/am_mpi.cc
  )
endif()

list(APPEND REALM_SRC
  realm.h
  realm/activemsg.h realm/activemsg.cc
  realm/network.h          realm/network.cc
  realm/bytearray.h
  realm/bytearray.inl
  realm/circ_queue.h
  realm/circ_queue.inl
  realm/cmdline.h          realm/cmdline.cc
  realm/cmdline.inl
  realm/codedesc.h         realm/codedesc.cc
  realm/codedesc.inl
  realm/custom_serdez.h
  realm/custom_serdez.inl
  realm/dynamic_table.h
  realm/dynamic_table.inl
  realm/event.h
  realm/id.h
  realm/id.inl
  realm/indexspace.h
  realm/instance.h
  realm/logging.h          realm/logging.cc
  realm/logging.inl
  realm/machine.h
  realm/machine.inl
  realm/memory.h
  realm/mutex.h     realm/mutex.inl     realm/mutex.cc
  realm/pri_queue.h
  realm/pri_queue.inl
  realm/processor.h
  realm/processor.inl
  realm/profiling.h        realm/profiling.cc
  realm/profiling.inl
  realm/realm_config.h
  realm/redop.h
  realm/reservation.h
  realm/reservation.inl
  realm/runtime.h
  realm/sampling.h
  realm/sampling.inl
  realm/serialize.h
  realm/serialize.inl
  realm/subgraph.h
  realm/subgraph.inl
  realm/timers.h           realm/timers.cc
  realm/timers.inl
  realm/utils.h            realm/utils.cc
)

# generate per-dimension object files for deppart stuff
foreach(INST_N1 RANGE 1 ${REALM_MAX_DIM})
  foreach(INST_N2 RANGE 1 ${REALM_MAX_DIM})
    foreach(SRCFILE realm/deppart/image realm/deppart/preimage realm/deppart/byfield)
      # use cmake's configure_file for a portable way of creating wrapper
      #  source files
      configure_file(${PROJECT_SOURCE_DIR}/cmake/deppart_tmpl.cc.in ${SRCFILE}_${INST_N1}_${INST_N2}.cc)
      list(APPEND REALM_SRC ${SRCFILE}_${INST_N1}_${INST_N2}.cc)
    endforeach()
  endforeach()
endforeach()

find_package(Threads REQUIRED)
cmake_policy(SET CMP0063 NEW) # visibility controls in all builds
add_library(RealmRuntime ${REALM_SRC})
add_library(Legion::RealmRuntime ALIAS RealmRuntime)
target_compile_options(RealmRuntime PRIVATE ${CXX_BUILD_WARNING_FLAGS})

# turn on warnings about partially-overloaded methods, if supported
check_cxx_compiler_flag("-Woverloaded-virtual" COMPILER_SUPORTS_OVERLOADED_VIRTUAL_WARNING)
if(COMPILER_SUPORTS_OVERLOADED_VIRTUAL_WARNING)
  target_compile_options(RealmRuntime PRIVATE "-Woverloaded-virtual")
endif()

if(COMPILER_SUPPORTS_DEFCHECK)
  # use the cxx_defcheck wrapper to make sure realm_defines.h is included
  #  any place it needs to be
  target_compile_options(RealmRuntime PRIVATE --defcheck realm_defines.h)
endif()
if(REALM_USE_LIBDL)
  target_link_libraries(RealmRuntime PRIVATE ${CMAKE_DL_LIBS})
  if(APPLE)
    target_link_libraries(RealmRuntime PRIVATE "-undefined dynamic_lookup")
  endif()
endif()
target_link_libraries(RealmRuntime PRIVATE ${CMAKE_THREAD_LIBS_INIT})
if(UNIX AND NOT APPLE)
  target_link_libraries(RealmRuntime PRIVATE rt)
endif()
set_target_properties(RealmRuntime PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(RealmRuntime PROPERTIES SOVERSION ${SOVERSION})

if(REALM_LIMIT_SYMBOL_VISIBILITY)
  set_target_properties(RealmRuntime PROPERTIES CXX_VISIBILITY_PRESET hidden)
  set_target_properties(RealmRuntime PROPERTIES VISIBILITY_INLINES_HIDDEN 1)
endif()

if(Legion_USE_PAPI)
  target_include_directories(RealmRuntime PRIVATE ${PAPI_INCLUDE_DIRS})
  target_link_libraries(RealmRuntime PUBLIC ${PAPI_LIBRARIES})
endif()

if(Legion_USE_HWLOC)
  target_link_libraries(RealmRuntime PRIVATE HWLOC::HWLOC)
endif()

if(REALM_USE_GASNET1 OR REALM_USE_GASNETEX)
  if(Legion_EMBED_GASNet)
    # we need to configure/build gasnet at cmake configuration time so that
    #  we can read in libraries and flags and stuff
    file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/embed-gasnet)
    set(GASNet_SOURCE_DIR ${PROJECT_BINARY_DIR}/embed-gasnet/source)
    set(GASNet_BUILD_DIR ${PROJECT_BINARY_DIR}/embed-gasnet/build)
    set(GASNet_INSTALL_DIR ${PROJECT_BINARY_DIR}/embed-gasnet/install)
    set(GASNet_BUILD_OUTPUT ${PROJECT_BINARY_DIR}/embed-gasnet/build.log)
    # if we're building shared libraries, we need -fPIC in the gasnet build
    if(BUILD_SHARED_LIBS)
      set(GASNET_CFLAGS -fPIC)
      set(GASNET_CXXFLAGS -fPIC)
    else()
      set(GASNET_CFLAGS "")
      set(GASNET_CXXFLAGS "")
    endif()
    # we'll write the GASNet configuration to a file so that we can rebuild
    #  (only) if it changes
    set(GASNet_CONFIG_FILE ${PROJECT_BINARY_DIR}/embed-gasnet/config.txt)
    set(GASNet_CONFIG_SETTINGS "CONDUIT=${GASNet_CONDUIT}" "GASNET_EXTRA_CONFIGURE_ARGS=${Legion_EMBED_GASNet_CONFIGURE_ARGS}" "GASNET_CFLAGS=${GASNET_CFLAGS}" "GASNET_CXXFLAGS=${GASNET_CXXFLAGS}")
    if(Legion_EMBED_GASNet_VERSION)
      # make the source directory version-specific
      set(GASNet_SOURCE_DIR ${PROJECT_BINARY_DIR}/embed-gasnet/${Legion_EMBED_GASNet_VERSION})
      list(APPEND GASNet_CONFIG_SETTINGS "GASNET_VERSION=${Legion_EMBED_GASNet_VERSION}")
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
      if(Legion_EMBED_GASNet_LOCALSRC)
	# relative paths are relative to the _build_ dir - this does the
	#  intuitive thing when you do:
	#     cmake ... -DLegion_EMBED_GASNet_LOCALSRC=relative/to/cwd (source_dir)
	#  but is a little nonobvious when you type:
	#     cmake -S (source_dir) -B (build_dir) ... -DLegion_EMBED_GASNet_LOCALSRC=still/relative/to/build_dir
	get_filename_component(
	  EMBEDDED_GASNET_SRC
	  "${Legion_EMBED_GASNet_LOCALSRC}"
	  ABSOLUTE BASE_DIR ${PROJECT_BINARY_DIR})
      else()
	# fetchcontent requires cmake 3.11+
	cmake_minimum_required(VERSION 3.11)
	include(FetchContent)
	FetchContent_Declare(embed-gasnet
	  GIT_REPOSITORY ${Legion_EMBED_GASNet_GITREPO}
	  GIT_TAG        ${Legion_EMBED_GASNet_GITREF}
	  )
	message(STATUS "Downloading StanfordLegion/gasnet repo from: ${Legion_EMBED_GASNet_GITREPO}")
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
      file(WRITE ${GASNet_CONFIG_FILE} "${GASNet_CONFIG_SETTINGS}")
    endif()
    # make sure GASNet_ROOT_DIR is defined at the top level
    set(GASNet_ROOT_DIR ${GASNet_INSTALL_DIR} CACHE STRING "Root directory for GASNet" FORCE)
    find_package(GASNet REQUIRED)
    # make a list of the gasnet libraries we need to install along with realm
    if(BUILD_SHARED_LIBS)
      # get the list of libraries and filter only those that are not in the
      #  (gasnet) install dir - the gasnet ones are sucked into librealm.so
      get_target_property(gasnet_ext_libs GASNet::${GASNet_CONDUIT}-par INTERFACE_LINK_LIBRARIES)
      list(FILTER gasnet_ext_libs EXCLUDE REGEX "^${GASNet_INSTALL_DIR}")
      set(GASNET_LINK_LIBRARIES_FOR_INSTALL ${gasnet_ext_libs} PARENT_SCOPE)
      # cmake won't include GASNet::GASNet in the exported link libraries
      #  because we're doing a shared build, so force that
      target_link_libraries(RealmRuntime INTERFACE $<LINK_ONLY:GASNet::GASNet>)
    else()
      # get the list of libraries and filter out those that are not in the
      #  (gasnet) install dir
      get_target_property(gasnet_all_libs GASNet::${GASNet_CONDUIT}-par INTERFACE_LINK_LIBRARIES)
      set(gasnet_libs "${gasnet_all_libs}")
      list(FILTER gasnet_libs INCLUDE REGEX "^${GASNet_INSTALL_DIR}")
      set(gasnet_installed_libs ${gasnet_libs})
      string(REGEX REPLACE "${GASNet_INSTALL_DIR}/([^/;]*/)*" "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/" gasnet_installed_libs "${gasnet_all_libs}")
      install(FILES ${gasnet_libs} DESTINATION ${CMAKE_INSTALL_LIBDIR})
      set(GASNET_LINK_LIBRARIES_FOR_INSTALL ${gasnet_installed_libs} PARENT_SCOPE)
    endif()
  else()
    # we're using an external GASNet, but we still need to compute the list
    #  of GASNet-related libraries that applications will need when using an
    #  installed Legion build
    # (this way, the application doesn't need to be able to find the original
    #  GASNet root and pick the right conduit and the like)
    get_target_property(gasnet_libs GASNet::${GASNet_CONDUIT}-par INTERFACE_LINK_LIBRARIES)
    if(BUILD_SHARED_LIBS)
      # filter out the actual gasnet libs because we suck those into librealm.so
      list(FILTER gasnet_libs EXCLUDE REGEX "^${GASNet_ROOT_DIR}")
    endif()
    set(GASNET_LINK_LIBRARIES_FOR_INSTALL ${gasnet_libs} PARENT_SCOPE)
  endif()

  target_link_libraries(RealmRuntime PRIVATE GASNet::GASNet)
  # cmake won't include GASNet::GASNet in the exported link libraries
  #  if we're building shared libraries, so force that
  if(BUILD_SHARED_LIBS)
    target_link_libraries(RealmRuntime INTERFACE $<LINK_ONLY:GASNet::GASNet>)
  endif()
endif()

if(REALM_USE_MPI)
  target_link_libraries(RealmRuntime PRIVATE MPI::MPI_CXX)
endif()

if(Legion_USE_LLVM)
  if(Legion_LINK_LLVM_LIBS)
    target_link_libraries(RealmRuntime PRIVATE LLVM::LLVM)
  else()
    # even if we dont link, we still need the LLVM include directories
    get_target_property(llvm_incdir LLVM::LLVM INTERFACE_INCLUDE_DIRECTORIES)
    target_include_directories(RealmRuntime PRIVATE ${llvm_incdir})
  endif()
endif()

if(Legion_USE_CUDA)
  if(REALM_USE_CUDART_HIJACK)
    if(Legion_USE_Kokkos)
      # we need kokkos to depend on the hijack without creating a circular
      #  dependency, so put it in an object library - this works only on newer
      #  cmakes, but we already need cmake 3.13 for Kokkos, so this is ok
      add_library(realm_cudarthijack OBJECT
	realm/cuda/cudart_hijack.cc
	realm/cuda/cudart_hijack.h)
      set_target_properties(realm_cudarthijack PROPERTIES POSITION_INDEPENDENT_CODE ON)
      if(REALM_LIMIT_SYMBOL_VISIBILITY)
        set_target_properties(realm_cudarthijack PROPERTIES CXX_VISIBILITY_PRESET hidden)
        set_target_properties(realm_cudarthijack PROPERTIES VISIBILITY_INLINES_HIDDEN 1)
      endif()
      target_include_directories(realm_cudarthijack PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}
	${PROJECT_BINARY_DIR}/runtime
	${CUDA_INCLUDE_DIRS})
      target_link_libraries(RealmRuntime PRIVATE realm_cudarthijack)
      install(TARGETS realm_cudarthijack EXPORT LegionTargets)
    else()
      # without Kokkos interop, we can just add the hijack directly
      target_sources(RealmRuntime PRIVATE realm/cuda/cudart_hijack.cc)
    endif()

    # filter anything referring to *cudart* out of the CUDA_LIBRARIES
    #  making sure our changes are visible to the scope above
    # so list(FILTER ...) doesn't exist until cmake 3.6 - do string regex instead
    string(REGEX REPLACE "[^\;]*cudart[^\;]*(\;?)" "" CUDA_LIBRARIES "${CUDA_LIBRARIES}")
    set(CUDA_LIBRARIES ${CUDA_LIBRARIES} PARENT_SCOPE)
  else()
    if(NOT REALM_CUDA_DYNAMIC_LOAD)
      target_link_libraries(RealmRuntime PRIVATE ${CUDA_CUDART_LIBRARY})
    endif()
  endif()

  target_include_directories(RealmRuntime PRIVATE ${CUDA_INCLUDE_DIRS})
  if(NOT REALM_CUDA_DYNAMIC_LOAD)
    target_link_libraries(RealmRuntime PRIVATE ${CUDA_CUDA_LIBRARY})
  endif()
  # for backwards compatibility in applications
  target_compile_definitions(RealmRuntime INTERFACE USE_CUDA)
endif()

if(Legion_USE_HIP)
  if (Legion_HIP_TARGET STREQUAL "CUDA")
    if(REALM_USE_HIP_HIJACK)
      target_sources(RealmRuntime PRIVATE realm/hip/hip_hijack.cc)
      set(CUDA_LIBRARIES ${CUDA_LIBRARIES} PARENT_SCOPE)
    endif()

    target_include_directories(RealmRuntime PRIVATE ${CUDA_INCLUDE_DIRS})
    target_include_directories(RealmRuntime PRIVATE ${HIP_ROOT_DIR}/include)
    target_link_libraries(RealmRuntime PRIVATE ${CUDA_CUDA_LIBRARY})
    # for backwards compatibility in applications
    target_compile_definitions(RealmRuntime INTERFACE USE_HIP)

    target_compile_definitions(RealmRuntime PRIVATE __HIP_PLATFORM_NVIDIA__)
  endif()

  if (Legion_HIP_TARGET STREQUAL "ROCM")
    if(REALM_USE_HIP_HIJACK)
      target_sources(RealmRuntime PRIVATE realm/hip/hip_hijack.cc)
    endif()

    set(HIP_LIBRARIES ${HIP_ROOT_DIR}/lib/libamdhip64.so)
    target_include_directories(RealmRuntime PRIVATE ${HIP_INCLUDE_DIRS})
    target_link_libraries(RealmRuntime PRIVATE ${HIP_LIBRARIES})
    # for backwards compatibility in applications
    target_compile_definitions(RealmRuntime INTERFACE USE_HIP)

    target_compile_definitions(RealmRuntime PRIVATE __HIP_PLATFORM_AMD__)
  endif()
endif()

# fetches a specified property list from the target, performs per-element
#  replacement of the items in the list, and then write the list back to
#  the target's property
# importantly, this works on properies of imported targets...
macro(edit_target_property tgtname propname pattern replacement)
  get_property(has_prop TARGET ${tgtname} PROPERTY ${propname} SET)
  if(has_prop)
    get_property(prop_list TARGET ${tgtname} PROPERTY ${propname})
    if(NOT "${replacement}" STREQUAL "")
      list(TRANSFORM prop_list REPLACE ${pattern} ${replacement})
    else()
      list(FILTER prop_list EXCLUDE REGEX ${pattern})
    endif()
    set_property(TARGET ${tgtname} PROPERTY ${propname} "${prop_list}")
  endif()
endmacro()

if(Legion_USE_Kokkos)
  # realm/kokkos_interop.cc needs to be compiled like a kokkos application, so
  #  get kokkos' compile options, include dirs - do this by creating a subtarget
  #  that can import Kokkos::kokkoscore

  # unfortunately, building like Kokkos means using Kokkos' choice of C++
  #  compiler, and cmake really doesn't want to make that easy - we have to
  #  do the build as a custom command and construct the entire command line
  #  ourselves

  # start with compile options exported by kokkos, but we have to parse the
  #  generator expressions ourselves...
  get_property(kokkos_compile_options TARGET Kokkos::kokkoscore PROPERTY INTERFACE_COMPILE_OPTIONS)
  string(REGEX REPLACE "\\$<\\$<COMPILE_LANGUAGE:CXX>:([^<>]*)>" \\1 kokkos_compile_options "${kokkos_compile_options}")
  string(REGEX REPLACE "\\$<\\$<COMPILE_LANGUAGE:.+>:([^<>]*)>" "" kokkos_compile_options "${kokkos_compile_options}")

  # also do kokkos' compile features (or at least the one we understand)
  get_property(kokkos_compile_features TARGET Kokkos::kokkoscore PROPERTY INTERFACE_COMPILE_FEATURES)
  foreach(feature IN LISTS kokkos_compile_features)
    if(feature MATCHES "cxx_std_([0-9]+)")
      list(APPEND kokkos_compile_options "-std=c++${CMAKE_MATCH_1}")
    else()
      message(WARNING "unrecognized Kokkos compile feature: ${feature}")
    endif()
  endforeach()

  # now add on our own cxx flags, dealing with the whole strings vs lists thing
  set(cxx_flags_as_list ${CMAKE_CXX_FLAGS} -fPIC ${CXX_BUILD_WARNING_FLAGS})
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    string(APPEND cxx_flags_as_list " ${CMAKE_CXX_FLAGS_DEBUG}")
  endif()
  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    string(APPEND cxx_flags_as_list " ${CMAKE_CXX_FLAGS_RELEASE}")
  endif()
  separate_arguments(cxx_flags_as_list)
  list(APPEND kokkos_compile_options ${cxx_flags_as_list})

  # next up is include directories...
  get_property(kokkos_include_dirs
               TARGET Kokkos::kokkoscore
               PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
  list(TRANSFORM kokkos_include_dirs PREPEND "-I")
  list(APPEND kokkos_compile_options ${kokkos_include_dirs}
                                     "-I${CMAKE_CURRENT_SOURCE_DIR}"
                                     "-I${PROJECT_BINARY_DIR}/runtime")

  if(REALM_LIMIT_SYMBOL_VISIBILITY)
    # control symbol visibility
    list(APPEND kokkos_compile_options "-fvisibility=hidden" "-fvisibility-inlines-hidden")
  endif()

  # and then finally, tell clang (if it is clang) where to find CUDA
  if(Legion_USE_CUDA AND (KOKKOS_CXX_COMPILER MATCHES ".*clang.*"))
    list(APPEND kokkos_compile_options "--cuda-path=${CUDA_TOOLKIT_ROOT_DIR}")
  endif()

  # now define our custom build command
  add_custom_command(OUTPUT realm_kokkos_interop.cc.o
                     COMMAND ${KOKKOS_CXX_COMPILER}
                             ${kokkos_compile_options}
                             -c ${CMAKE_CURRENT_SOURCE_DIR}/realm/kokkos_interop.cc
                             -o ${CMAKE_CURRENT_BINARY_DIR}/realm_kokkos_interop.cc.o
                     IMPLICIT_DEPENDS CXX ${CMAKE_CURRENT_SOURCE_DIR}/realm/kokkos_interop.cc
                     VERBATIM)

  # add our object file to the realm library
  target_sources(RealmRuntime PRIVATE
                 ${CMAKE_CURRENT_BINARY_DIR}/realm_kokkos_interop.cc.o)

  # and add kokkos libraries to our exported library requirements
  set_property(TARGET RealmRuntime APPEND
    PROPERTY INTERFACE_LINK_LIBRARIES "\$<LINK_ONLY:Kokkos::kokkoscore>")

  # if Kokkos' CUDA target is enabled, and we're performing runtime hijack,
  #  rewrite Kokkos' cudart dependency to be on Realm instead
  if(Kokkos_ENABLE_CUDA AND Legion_HIJACK_CUDART)
    edit_target_property(Kokkos::CUDA INTERFACE_LINK_LIBRARIES .*cudart.* realm_cudarthijack)
  endif()

  # similarly, if Kokkos' OpenMP target is enabled but we're using Realm's
  #  OpenMP runtime implementation, remove anything mentioning openmp from
  #  the exported link options (don't mess with compile options though)
  if(Kokkos_ENABLE_OPENMP AND Legion_USE_OpenMP AND NOT Legion_OpenMP_SYSTEM_RUNTIME)
    edit_target_property(Kokkos::kokkoscore INTERFACE_LINK_OPTIONS .*openmp.* "")
    edit_target_property(Kokkos::kokkoscontainers INTERFACE_LINK_OPTIONS .*openmp.* "")
  endif()
endif()

if(Legion_USE_HDF5)
  target_include_directories(RealmRuntime PRIVATE ${HDF5_INCLUDE_DIRS})
  target_link_libraries(RealmRuntime PRIVATE ${HDF5_LIBRARIES})
  # for backwards compatibility in applications
  target_compile_definitions(RealmRuntime INTERFACE USE_HDF)
endif()

if(Legion_USE_OpenMP)
  # only include -fopenmp (or equivalent) if we're using the
  #  system omp runtime libs
  if(Legion_OpenMP_SYSTEM_RUNTIME)
    target_link_libraries(RealmRuntime PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif()

set_target_properties(RealmRuntime PROPERTIES OUTPUT_NAME "realm${INSTALL_SUFFIX}")

target_include_directories(RealmRuntime
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}

  # Include paths for generated header files.
  INTERFACE
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/runtime>
  PRIVATE
    ${PROJECT_BINARY_DIR}/runtime
)

install(TARGETS RealmRuntime EXPORT LegionTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Mapper objects
list(APPEND MAPPER_SRC
  mappers/default_mapper.h     mappers/default_mapper.cc
  mappers/default_mapper.inl
  mappers/mapping_utilities.h  mappers/mapping_utilities.cc
  mappers/shim_mapper.h        mappers/shim_mapper.cc
  mappers/test_mapper.h        mappers/test_mapper.cc
  mappers/null_mapper.h        mappers/null_mapper.cc
  mappers/replay_mapper.h      mappers/replay_mapper.cc
  mappers/debug_mapper.h       mappers/debug_mapper.cc
  mappers/wrapper_mapper.h     mappers/wrapper_mapper.cc
  mappers/forwarding_mapper.h  mappers/forwarding_mapper.cc
  mappers/logging_wrapper.h    mappers/logging_wrapper.cc
)

# Legion runtime
list(APPEND LEGION_SRC
  legion.h
  legion/accessor.h
  legion/arrays.h
  legion/bitmask.h
  legion/field_tree.h
  legion/garbage_collection.h             legion/garbage_collection.cc
  legion/interval_tree.h
  legion/legion_allocation.h
  legion/legion_analysis.h                legion/legion_analysis.cc
  legion/legion_c.h                       legion/legion_c.cc
  legion/legion_config.h
  legion/legion_constraint.h              legion/legion_constraint.cc
  legion/legion_context.h                 legion/legion_context.cc
  legion/legion_c_util.h
  legion/legion.cc
  legion/legion.inl
  legion/legion_domain.h
  legion/legion_domain.inl
  legion/legion_instances.h               legion/legion_instances.cc
  legion/legion_mapping.h                 legion/legion_mapping.cc
  legion/legion_ops.h                     legion/legion_ops.cc
  legion/legion_profiling.h               legion/legion_profiling.cc
  legion/legion_profiling_serializer.h    legion/legion_profiling_serializer.cc
  legion/legion_spy.h                     legion/legion_spy.cc
  legion/legion_tasks.h                   legion/legion_tasks.cc
  legion/legion_trace.h                   legion/legion_trace.cc
  legion/legion_types.h
  legion/legion_utilities.h
  legion/legion_views.h                   legion/legion_views.cc
  legion/legion_redop.h                   legion/legion_redop.cc
  legion/mapper_manager.h                 legion/mapper_manager.cc
  legion/rectangle_set.h
  legion/region_tree.h                    legion/region_tree.cc
  legion/runtime.h                        legion/runtime.cc
)

if(Legion_USE_HIP)
  list(APPEND LEGION_SRC hip_cuda_compat/hip_cuda.h)
endif()

if(Legion_REDOP_HALF)
  list(APPEND LEGION_SRC mathtypes/half.h)
endif()

if(Legion_REDOP_COMPLEX)
  list(APPEND LEGION_SRC mathtypes/complex.h)
endif()

if(Legion_USE_CUDA)
  list(APPEND LEGION_CUDA_SRC
    legion/legion_redop.cu
  )
endif()

if(Legion_USE_HIP AND Legion_GPU_REDUCTIONS)
  list(APPEND LEGION_HIP_SRC
    legion/legion_redop.cu
  )
endif()

# generate per-dimension object files for deppart stuff
foreach(INST_N1 RANGE 1 ${LEGION_MAX_DIM})
  # use cmake's configure_file for a portable way of creating wrapper
  #  source files
  set(SRCFILE legion/region_tree)
  configure_file(${PROJECT_SOURCE_DIR}/cmake/deppart_tmpl.cc.in ${SRCFILE}_${INST_N1}.cc)
  list(APPEND LEGION_SRC ${SRCFILE}_${INST_N1}.cc)

  foreach(INST_N2 RANGE 1 ${LEGION_MAX_DIM})
    configure_file(${PROJECT_SOURCE_DIR}/cmake/deppart_tmpl.cc.in ${SRCFILE}_${INST_N1}_${INST_N2}.cc)
    list(APPEND LEGION_SRC ${SRCFILE}_${INST_N1}_${INST_N2}.cc)
  endforeach()
  unset(SRCFILE)
endforeach()

# Legion Fortran
if(Legion_USE_Fortran)
  list(APPEND LEGION_FORTRAN_SRC
    legion/legion_f_types.f90
    legion/legion_f_c_interface.f90
    legion/legion_f.f90
  )
endif()

# HACK: completely separating the Legion and Realm cmake stuff will take
#  a while, so just exclude LegionRuntime from the "all" target for the
#  Realm-only case for now
if(Legion_BUILD_REALM_ONLY)
  set(EXCLUDE_LEGION EXCLUDE_FROM_ALL)
endif()

add_library(LegionRuntime ${EXCLUDE_LEGION} ${MAPPER_SRC} ${LEGION_SRC} ${LEGION_FORTRAN_SRC})
add_library(Legion::LegionRuntime ALIAS LegionRuntime)
# do we have any cuda source files in Legion itself?
if(LEGION_CUDA_SRC)
  # Fill in the most compatible set of architectures we support
  if("${Legion_CUDA_ARCH}" STREQUAL "")
    if (${CUDA_VERSION_STRING} VERSION_LESS "11.5")
      # Prior to 11.5, we don't have -arch=all and have to list the architectures out manually
      if (${CUDA_VERSION_STRING} VERSION_GREATER_EQUAL "3.2" AND ${CUDA_VERSION_STRING} VERSION_LESS "10.0")
        list(APPEND Legion_CUDA_ARCH 20)
      endif ()
      if (${CUDA_VERSION_STRING} VERSION_GREATER_EQUAL "5.0" AND ${CUDA_VERSION_STRING} VERSION_LESS "11.0")
        list(APPEND Legion_CUDA_ARCH 30)
      endif ()
      if (${CUDA_VERSION_STRING} VERSION_GREATER_EQUAL "5.0" AND ${CUDA_VERSION_STRING} VERSION_LESS "12.0")
        list(APPEND Legion_CUDA_ARCH 35)
      endif ()
      if (${CUDA_VERSION_STRING} VERSION_GREATER_EQUAL "6.0" AND ${CUDA_VERSION_STRING} VERSION_LESS "12.0")
        list(APPEND Legion_CUDA_ARCH 50)
      endif ()
      if (${CUDA_VERSION_STRING} VERSION_GREATER_EQUAL "8.0")
        list(APPEND Legion_CUDA_ARCH 60)
      endif ()
      if (${CUDA_VERSION_STRING} VERSION_GREATER_EQUAL "9.0")
        list(APPEND Legion_CUDA_ARCH 70)
      endif ()
      if (${CUDA_VERSION_STRING} VERSION_GREATER_EQUAL "11.0")
        list(APPEND Legion_CUDA_ARCH 80)
      endif ()
    else ()
        # New with 11.5, nvcc supports 'all-major' which does all this for us
        set(CUDA_GENCODE "-arch all-major")
    endif ()
  else ()
      # If the user specified a specific gencode to use, use that
      list(SORT Legion_CUDA_ARCH)
      string(REPLACE "," ";" CUDA_GENCODE "${Legion_CUDA_ARCH}")
      # Add all the specified architectures we support
      string(REGEX REPLACE "([0-9]+)" "-gencode arch=compute_\\1,code=sm_\\1" CUDA_GENCODE "${CUDA_GENCODE}")
      list(GET Legion_CUDA_ARCH -1 COMPUTE_ARCH)
      # For the last architecture we support, add PTX
      list(APPEND CUDA_GENCODE "-gencode arch=compute_${COMPUTE_ARCH},code=compute_${COMPUTE_ARCH}")
  endif ()

  message("${CUDA_GENCODE}")

  # starting with NVCC 10.2, we can error out on all warnings
  if(Legion_BUILD_WARN_AS_ERROR AND CUDA_VERSION VERSION_GREATER "10.1")
    set(CUDA_WARNINGS "-Werror all-warnings")
  else()
    set(CUDA_WARNINGS "")
  endif()
  cuda_compile(LEGION_CUDA_OBJS ${LEGION_CUDA_SRC} OPTIONS
    -Xcudafe "--diag_suppress=boolean_controlling_expr_is_constant"
    ${CUDA_WARNINGS}
    ${CUDA_GENCODE}
    -I ${CMAKE_CURRENT_SOURCE_DIR}
    -I ${PROJECT_BINARY_DIR}/runtime
    -Xcompiler -fPIC
    DEBUG -g
    RELEASE -O2
    RELWITHDEBINFO -g -O2)
  target_sources(LegionRuntime PRIVATE ${LEGION_CUDA_OBJS})
  target_link_libraries(LegionRuntime PUBLIC ${CUDA_LIBRARIES})

  # complex reduction ops bring in a public dependency on cuda headers
  if(Legion_REDOP_COMPLEX)
    target_include_directories(LegionRuntime PUBLIC ${CUDA_INCLUDE_DIRS})
  endif()
endif()

if(LEGION_HIP_SRC)
  if (Legion_HIP_TARGET STREQUAL "CUDA")
    if("${Legion_CUDA_ARCH}" STREQUAL "")
      set(CUDA_GENCODE "")
    else()
      string(REPLACE "," ";" CUDA_GENCODE "${Legion_CUDA_ARCH}")
      string(REGEX REPLACE "([0-9]+)" "-gencode arch=compute_\\1,code=sm_\\1" CUDA_GENCODE "${CUDA_GENCODE}")
    endif()
    # starting with NVCC 10.2, we can error out on all warnings
    if(Legion_BUILD_WARN_AS_ERROR AND CUDA_VERSION VERSION_GREATER "10.1")
      set(CUDA_WARNINGS "-Werror all-warnings")
    else()
      set(CUDA_WARNINGS "")
    endif()
    cuda_compile(LEGION_HIP_OBJS ${LEGION_HIP_SRC} OPTIONS
      -Xcudafe "--diag_suppress=boolean_controlling_expr_is_constant"
      ${CUDA_WARNINGS}
      ${CUDA_GENCODE}
      ${HIPCC_FLAGS}
      -I ${CMAKE_CURRENT_SOURCE_DIR}
      -I ${PROJECT_BINARY_DIR}/runtime
      -Xcompiler -fPIC
      DEBUG -g
      RELEASE -O2
      RELWITHDEBINFO -g -O2)
    target_sources(LegionRuntime PRIVATE ${LEGION_HIP_OBJS})
    target_link_libraries(LegionRuntime PUBLIC ${CUDA_LIBRARIES})

    # complex reduction ops bring in a public dependency on cuda headers
    if(Legion_REDOP_COMPLEX)
      target_include_directories(LegionRuntime PUBLIC ${CUDA_INCLUDE_DIRS})
      target_compile_definitions(LegionRuntime PUBLIC __HIP_PLATFORM_NVIDIA__)
    endif()
  endif()

  if (Legion_HIP_TARGET STREQUAL "ROCM")
    set(HIP_LIBRARIES ${HIP_ROOT_DIR}/lib/libamdhip64.so)
    set_source_files_properties(${LEGION_HIP_SRC} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
    hip_compile(LEGION_HIP_OBJS ${LEGION_HIP_SRC} HIPCC_OPTIONS
      ${HIPCC_FLAGS}
      ${HIP_GENCODE}
      -I ${CMAKE_CURRENT_SOURCE_DIR}
      -I ${CMAKE_CURRENT_SOURCE_DIR}/legion
      -I ${PROJECT_BINARY_DIR}/runtime
      -fPIC
      DEBUG -g
      RELEASE -O2
      RELWITHDEBINFO -g -O2)
    target_sources(LegionRuntime PRIVATE ${LEGION_HIP_OBJS})
    target_link_libraries(LegionRuntime PUBLIC ${HIP_LIBRARIES})

    # complex reduction ops bring in a public dependency on hip thrust headers
    if(Legion_REDOP_COMPLEX)
      # set thrust package
      # please download thrust from https://github.com/ROCmSoftwarePlatform/Thrust
      if(NOT DEFINED THRUST_PATH)
          set(THRUST_PATH $ENV{THRUST_PATH} CACHE PATH "Path to where Thrust has been installed")
      endif()
      target_include_directories(LegionRuntime PUBLIC ${HIP_INCLUDE_DIRS})
      target_compile_definitions(LegionRuntime PUBLIC __HIP_PLATFORM_AMD__)
    else()
      target_include_directories(LegionRuntime PRIVATE ${HIP_INCLUDE_DIRS})
      target_compile_definitions(LegionRuntime PRIVATE __HIP_PLATFORM_AMD__)
    endif()
  endif()
endif()

target_link_libraries(LegionRuntime PUBLIC RealmRuntime)
if(Legion_USE_ZLIB)
  target_link_libraries(LegionRuntime PRIVATE ZLIB::ZLIB)
endif()
set_target_properties(LegionRuntime PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(LegionRuntime PROPERTIES OUTPUT_NAME "legion${INSTALL_SUFFIX}")
set_target_properties(LegionRuntime PROPERTIES SOVERSION ${SOVERSION})
target_compile_options(LegionRuntime PRIVATE ${CXX_BUILD_WARNING_FLAGS})
if(COMPILER_SUPPORTS_DEFCHECK)
  # use the cxx_defcheck wrapper to make sure legion_defines.h is included
  #  any place it needs to be
  target_compile_options(LegionRuntime PRIVATE --defcheck legion_defines.h)
endif()

target_include_directories(LegionRuntime
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/mappers>
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/mappers>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}

  # Include paths for generated header files.
  INTERFACE
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/runtime>
  PRIVATE
    ${PROJECT_BINARY_DIR}/runtime
)

install(TARGETS LegionRuntime EXPORT LegionTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

add_library(Legion INTERFACE)
set_target_properties(Legion PROPERTIES
  INTERFACE_LINK_LIBRARIES LegionRuntime
)
add_library(Realm INTERFACE)
set_target_properties(Realm PROPERTIES
  INTERFACE_LINK_LIBRARIES RealmRuntime
)
install(TARGETS Legion Realm EXPORT LegionTargets)
