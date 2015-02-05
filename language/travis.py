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

def test(root_dir, install_args, install_env):
    subprocess.check_call(
        ['./install.py'] + install_args,
        env = install_env,
        cwd = root_dir)
    subprocess.check_call(
        ['./test.py'],
        cwd = root_dir)

if __name__ == '__main__':
    root_dir = os.path.realpath(os.path.dirname(__file__))
    legion_dir = os.path.dirname(root_dir)
    runtime_dir = os.path.join(legion_dir, 'runtime')

    install_env = dict(os.environ.items() + [
        ('LG_RT_DIR', runtime_dir),
    ])

    test(root_dir, ['--debug'], install_env)
    test(root_dir, [], install_env)
