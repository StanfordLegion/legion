#!/usr/bin/env python

# Copyright 2013 Stanford University and Los Alamos National Security, LLC
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

import os, sys
_root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(_root_dir)
sys.path.append(os.path.join(_root_dir, 'lib'))

from lcomp import passes

if __name__ == '__main__':
    program = passes.parse(sys.stdin)
    print program
