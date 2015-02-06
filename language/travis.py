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
    if platform.system() == 'Darwin':
        subprocess.check_call(['brew', 'update'])
        subprocess.check_call(['brew', 'install', 'xz'])

        clang_tarball = 'clang+llvm-3.5.0-macosx-apple-darwin.tar.xz'
        clang_dir = 'clang+llvm-3.5.0-macosx-apple-darwin'

        subprocess.check_call(
            ['curl', '-O', 'http://llvm.org/releases/3.5.0/%s' % clang_tarball])
        shasum = subprocess.Popen(['shasum', '-c'], stdin=subprocess.PIPE)
        shasum.communicate(
            'ea15cfe99022fb2abce219d7e8a4377b81f7b1fb  %s' % clang_tarball)
        assert shasum.wait() == 0
        subprocess.check_call(['tar', 'xfJ', clang_tarball])

        os.environ['PATH'] = ':'.join(
            [os.path.join(clang_dir, 'bin'), os.environ['PATH']])
        os.environ['DYLD_LIBRARY_PATH'] = ':'.join(
            [os.path.join(clang_dir, 'lib')] +
            ([os.environ['DYLD_LIBRARY_PATH']]
             if 'DYLD_LIBRARY_PATH' in os.environ else []))

        print 'clang is:'
        subprocess.check_call(
            ['which', 'clang'])
        print 'clang++ is:'
        subprocess.check_call(
            ['which', 'clang++'])

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

    install_dependencies()

    test(root_dir, ['--debug'], install_env)
    test(root_dir, [], install_env)
