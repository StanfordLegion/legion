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
      if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "3.2" AND CUDAToolkit_VERSION VERSION_LESS "10.0")
        list(APPEND archs 20)
      endif()
      if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "5.0" AND CUDAToolkit_VERSION VERSION_LESS "11.0")
        list(APPEND archs 30)
      endif()
      if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "5.0" AND CUDAToolkit_VERSION VERSION_LESS "12.0")
        list(APPEND archs 35)
      endif()
      if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "6.0" AND CUDAToolkit_VERSION VERSION_LESS "12.0")
        list(APPEND archs 50)
      endif()
      if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "8.0")
        list(APPEND archs 60)
      endif()
      if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "9.0")
        list(APPEND archs 70)
      endif()
      if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.0")
        list(APPEND archs 80)
      endif()
      if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.8.0")
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

# Set a target's CUDA_ARCHITECTURES property, or translate the archs list
# to --generate-code flags for CMake < 3.18
function(set_target_cuda_architectures)
    set(options )
    set(oneValueArgs TARGET)
    set(multiValueArgs ARCHS)
    cmake_parse_arguments(cuda "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # Translate CUDA_ARCHITECTURES to -gencode flags for CMake < 3.18
  if(CMAKE_VERSION VERSION_LESS_EQUAL "3.17")
    set(flags )
    if(cuda_ARCHS)
      set(archs ${cuda_ARCHS})
    else()
      get_target_property(archs ${cuda_TARGET} CUDA_ARCHITECTURES)
    endif()

    # ARCH=75-real    : --generate-code=arch=compute_75,code=[sm_75]
    # ARCH=75-virtual : --generate-code=arch=compute_75,code=[compute_75]
    # ARCH=75         : --generate-code=arch=compute_75,code=[compute_75,sm_75]
    foreach(arch IN LISTS archs)
      set(codes "compute_XX" "sm_XX")
      if(arch MATCHES "-real")
        # remove "compute_XX"
        list(POP_FRONT codes)
        string(REPLACE "-real" "" arch "${arch}")
      elseif(arch MATCHES "-virtual")
        # remove "sm_XX"
        list(POP_BACK codes)
        string(REPLACE "-virtual" "" arch "${arch}")
      endif()
      list(TRANSFORM codes REPLACE "_XX" "_${arch}")
      list(JOIN codes "," codes)
      list(APPEND flags "--generate-code=arch=compute_${arch},code=[${codes}]")
    endforeach()

    target_compile_options(${cuda_TARGET} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${flags}>)
  elseif(cuda_ARCHS)
    set_property(TARGET ${cuda_TARGET} PROPERTY CUDA_ARCHITECTURES ${cuda_ARCHS})
  endif()
endfunction()
