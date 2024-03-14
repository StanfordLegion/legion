# Copyright 2024 Stanford University, NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script can be called as a custom cmake command to generate a C
# file based on a generated file in a platform agnostic way such that
# the platform doesn't need bin2c to be available.
cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

# Read the input file as raw hex values
file(READ "${IN_FILE}" FILE_CONTENTS_HEX HEX)
# Replace the hex values with
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," FILE_CONTENTS_HEX ${FILE_CONTENTS_HEX})
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/bin2c.template.in ${OUT_FILE} @ONLY)
