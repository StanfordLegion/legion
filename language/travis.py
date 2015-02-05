#!/usr/bin/env python

# Copyright 2015 Stanford University
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

import os, subprocess

root_dir = os.path.realpath(os.path.dirname(__file__))

def test(install_args):
    subprocess.check_call(['./install.py'] + install_args, cwd = root_dir)
    subprocess.check_call(['./test.py'], cwd = root_dir)

if __name__ == '__main__':
    test(['--debug'])
    test([])
