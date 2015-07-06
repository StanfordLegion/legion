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

import os, platform, subprocess

def install_dependencies():
    env = dict(os.environ.iteritems())

    if platform.system() == 'Darwin':
        clang_tarball = 'clang+llvm-3.4.2-x86_64-apple-darwin10.9.xz'
        clang_dir = os.path.abspath('clang+llvm-3.4.2-x86_64-apple-darwin10.9')

        print('http://llvm.org/releases/3.4.2/%s' % clang_tarball)
        subprocess.check_call(
            ['curl', '-O', 'http://llvm.org/releases/3.4.2/%s' % clang_tarball])
        shasum = subprocess.Popen(['shasum', '-c'], stdin=subprocess.PIPE)
        shasum.communicate(
            'b182ca49f8e4933041daa8ed466f1e4a589708bf  %s' % clang_tarball)
        assert shasum.wait() == 0
        subprocess.check_call(['tar', 'xfJ', clang_tarball])

        env.update({
            'PATH': ':'.join(
                [os.path.join(clang_dir, 'bin'), os.environ['PATH']]),
            'DYLD_LIBRARY_PATH': ':'.join(
                [os.path.join(clang_dir, 'lib')] +
                ([os.environ['DYLD_LIBRARY_PATH']]
                 if 'DYLD_LIBRARY_PATH' in os.environ else [])),
        })

    return env

def test(root_dir, install_args, install_env):
    subprocess.check_call(
        ['./install.py', '-j', '2'] + install_args,
        env = install_env,
        cwd = root_dir)
    subprocess.check_call(
        ['./test.py', '-j1'],
        cwd = root_dir)

if __name__ == '__main__':
    root_dir = os.path.realpath(os.path.dirname(__file__))
    legion_dir = os.path.dirname(root_dir)
    runtime_dir = os.path.join(legion_dir, 'runtime')

    env = install_dependencies()
    env.update({
        'LG_RT_DIR': runtime_dir,
    })

    test(root_dir, ['--debug'], env)
    test(root_dir, [], env)
