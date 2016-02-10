#------------------------------------------------------------------------------#
# Copyright (c) 2014 Los Alamos National Security, LLC
# All rights reserved.
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# Options
#------------------------------------------------------------------------------#

set(GASNET_ROOT "" CACHE PATH "Root directory of GASNet installation")
set(GASNET_CONDUIT FALSE CACHE STRING "GASNet conduit to use")
set(GASNET_MODEL FALSE CACHE STRING "GASNet model to use")

#------------------------------------------------------------------------------#
# Find the header file
#------------------------------------------------------------------------------#

find_path(GASNET_INCLUDE_DIR gasnet.h
    HINTS ENV GASNET_ROOT
    PATHS ${GASNET_ROOT}
    PATH_SUFFIXES include)

#------------------------------------------------------------------------------#
# Try to find the library
#------------------------------------------------------------------------------#

set(GASNET_LIBRARY_FOUND False)

if(NOT GASNET_CONDUIT)

    #--------------------------------------------------------------------------#
    # Conduit not specified -> take the first one that we find
    #--------------------------------------------------------------------------#

    foreach(conduit aries gemini ibv mpi mxm pami portals4 shmem smp udp)

        if(NOT GASNET_MODEL)

            #------------------------------------------------------------------#
            # Model not specified -> take the first one that we find
            #------------------------------------------------------------------#

            foreach(model par seq parsync)
                find_library(GASNET_${conduit}_${model}
                    gasnet-${conduit}-${model}
                    HINTS ENV GASNET_ROOT
                    PATHS ${GASNET_ROOT}
                    PATH_SUFFIXES lib lib64)

                if(GASNET_${conduit}_${model})
                    if(NOT GASNET_LIBRARY_FOUND)
                        set(GASNET_LIBRARY_FOUND
                            ${GASNET_${conduit}_${model}})
                    endif()
                endif()
            endforeach(model)

        else()

            #------------------------------------------------------------------#
            # Model specified -> see if it exists
            #------------------------------------------------------------------#

            find_library(GASNET_${conduit}_${GASNET_MODEL}
                gasnet-${conduit}-${GASNET_MODEL}
                HINTS ENV GASNET_ROOT
                PATHS ${GASNET_ROOT}
                PATH_SUFFIXES lib lib64)

            if(GASNET_${conduit}_${GASNET_MODEL})
                if(NOT GASNET_LIBRARY_FOUND)
                    set(GASNET_LIBRARY_FOUND
                        ${GASNET_${conduit}_${GASNET_MODEL}})
                endif()
            endif()

        endif(NOT GASNET_MODEL)

    endforeach(conduit)

else()

    if(NOT GASNET_MODEL)

        #----------------------------------------------------------------------#
        # Conduit specified
        #----------------------------------------------------------------------#

        foreach(model par seq parsync)

            find_library(GASNET_${GASNET_CONDUIT}_${model}
                gasnet-${GASNET_CONDUIT}-${model}
                HINTS ENV GASNET_ROOT
                PATHS ${GASNET_ROOT}
                PATH_SUFFIXES lib lib64)

            if(GASNET_${GASNET_CONDUIT}_${model})
                if(NOT GASNET_LIBRARY_FOUND)
                    set(GASNET_LIBRARY_FOUND
                        ${GASNET_${GASNET_CONDUIT}_${model}})
                endif()
            endif()

        endforeach(model)

    else()

        #----------------------------------------------------------------------#
        # Conduit and model specified
        #----------------------------------------------------------------------#

        find_library(GASNET_${GASNET_CONDUIT}_${GASNET_MODEL}
            gasnet-${GASNET_CONDUIT}-${GASNET_MODEL}
            HINTS ENV GASNET_ROOT
            PATHS ${GASNET_ROOT}
            PATH_SUFFIXES lib lib64)

        if(GASNET_${GASNET_CONDUIT}_${GASNET_MODEL})
            set(GASNET_LIBRARY_FOUND
                ${GASNET_${GASNET_CONDUIT}_${GASNET_MODEL}})
        endif()

    endif(NOT GASNET_MODEL)

endif(NOT GASNET_CONDUIT)

#------------------------------------------------------------------------------#
# Set standard args stuff
#------------------------------------------------------------------------------#

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(GASNET
    REQUIRED_VARS GASNET_LIBRARY_FOUND GASNET_INCLUDE_DIR)

#------------------------------------------------------------------------------#
# Formatting options for emacs and vim.
# vim: set tabstop=4 shiftwidth=4 expandtab :
#------------------------------------------------------------------------------#
