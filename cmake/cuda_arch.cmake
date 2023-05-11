include_guard(GLOBAL)

function(populate_cuda_archs_list ARCHS)

  set(archs ${${ARCHS}})

  if(NOT archs)
    #
    # CMake 3.23 will figure out the list from `CUDA_ARCHITECTURES=all-major`,
    # but in older CMake we have to enumerate the architecture list manually.
    # https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
    #
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.23")
      set(${ARCHS} all-major PARENT_SCOPE)
      return()
    else()
      if(${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "3.2" AND ${CUDAToolkit_VERSION} VERSION_LESS "10.0")
        list(APPEND archs 20)
      endif()
      if(${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "5.0" AND ${CUDAToolkit_VERSION} VERSION_LESS "11.0")
        list(APPEND archs 30)
      endif()
      if(${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "5.0" AND ${CUDAToolkit_VERSION} VERSION_LESS "12.0")
        list(APPEND archs 35)
      endif()
      if(${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "6.0" AND ${CUDAToolkit_VERSION} VERSION_LESS "12.0")
        list(APPEND archs 50)
      endif()
      if(${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "8.0")
        list(APPEND archs 60)
      endif()
      if(${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "9.0")
        list(APPEND archs 70)
      endif()
      if(${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "11.0")
        list(APPEND archs 80)
      endif()
      if(${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "11.8.0")
        list(APPEND archs 90)
      endif()
    endif()
  endif()

  # Compile all supported major real architectures (SASS),
  # and the highest major virtual architecture (PTX+SASS).
  list(SORT archs)
  list(POP_BACK archs highest_arch)
  list(TRANSFORM archs APPEND "-real")
  list(APPEND archs ${highest_arch})

  set(${ARCHS} ${archs} PARENT_SCOPE)
endfunction(populate_cuda_archs_list)

function(archs_list_to_gencode_flags)
  set(options )
  set(oneValueArgs FLAGS TARGET)
  set(multiValueArgs ARCHS)
  cmake_parse_arguments(CUDA "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(flags )

  if(${CUDA_FLAGS})
    set(flags ${${CUDA_FLAGS}})
  endif()

  if(NOT CUDA_ARCHS)
    if(${CUDA_TARGET} AND (TARGET ${CUDA_TARGET}))
      get_target_property(CUDA_ARCHS ${CUDA_TARGET} CUDA_ARCHITECTURES)
    else()
      set(CUDA_ARCHS ${CMAKE_CUDA_ARCHITECTURES})
    endif()
  endif()

  list(REMOVE_DUPLICATES CUDA_ARCHS)

  # ARCH=75-real    : --generate-code=arch=compute_75,code=[sm_75]
  # ARCH=75-virtual : --generate-code=arch=compute_75,code=[compute_75]
  # ARCH=75         : --generate-code=arch=compute_75,code=[compute_75,sm_75]
  foreach(ARCH IN LISTS CUDA_ARCHS)
    set(codes "compute_XX" "sm_XX")
    if(ARCH MATCHES "-real")
      list(POP_FRONT codes) # remove "compute_XX"
      string(REPLACE "-real" "" ARCH "${ARCH}")
    elseif(ARCH MATCHES "-virtual")
      list(POP_BACK codes) # remove "sm_XX"
      string(REPLACE "-virtual" "" ARCH "${ARCH}")
    endif()
    list(TRANSFORM codes REPLACE "_XX" "_${ARCH}")
    list(JOIN codes "," codes)
    list(APPEND flags "--generate-code=arch=compute_${ARCH},code=[${codes}]")
  endforeach()

  set(${CUDA_FLAGS} ${flags} PARENT_SCOPE)

endfunction()
