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

cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(LegionExample_future_instance)

# Only search if were building stand-alone and not as part of Legion
if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()
if(NOT Legion_USE_CUDA)
  message(FATAL_ERROR
    "The future_instance example requires Legion to be built with CUDA")
endif()

set(GPU_SOURCES future_instance.cu)
add_executable(future_instance ${GPU_SOURCES})
set_target_cuda_standard(future_instance STANDARD ${Legion_CUDA_STANDARD})
set_target_cuda_architectures(future_instance ARCHITECTURES ${Legion_CUDA_ARCH})
set_target_cuda_warnings_and_errors(future_instance WARN_AS_ERROR ${Legion_BUILD_WARN_AS_ERROR})

target_link_libraries(future_instance Legion::Legion)
