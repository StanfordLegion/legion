#!/usr/bin/env python

# Copyright 2018 Stanford University
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

export LG_RT_DIR="$LEGION_HOME/runtime"
export TERRA_DIR="$LEGION_HOME/language/terra/"
export LUAJIT_DIR="$LEGION_HOME/language/terra/build/LuaJIT-2.0.3"
export PATH="$TERRA_DIR:$PATH"
export TERRA_BINDING_DIR="$LEGION_HOME/bindings/terra/"
export DYLD_LIBRARY_PATH="$TERRA_BINDING_DIR:$LUAJIT_DIR/src"
export INCLUDE_PATH="$LG_RT_DIR;$TERRA_BINDING_DIR"
export TERRA_PATH="$TERRA_BINDING_DIR/?.t;?.t"
