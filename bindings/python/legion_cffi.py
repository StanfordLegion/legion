#!/usr/bin/env python3

# Copyright 2023 Stanford University
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
#

# IMPORTANT:
#   * legion_cffi.py.in is used as an input to string.format()
#   * legion_builtin_cffi.py and legion_canonical_cffi.py are generated files 
#     and should not be modified by hand
#   * legion_cffi.py is the unified interface of two cffi.py files

from __future__ import absolute_import, division, print_function, unicode_literals

# check which python interpreter is launched and then import the correct cffi
is_legion_python = None
try:
    from legion_builtin_cffi import ffi, lib
    # as long as lib is not empty, we are running with legion_python
    lib.legion_runtime_has_context()
    is_legion_python = True
except AttributeError:
    is_legion_python = False
    from legion_canonical_cffi import ffi, lib
