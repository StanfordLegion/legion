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
