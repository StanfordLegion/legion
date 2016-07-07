#!/usr/bin/env python

# Copyright 2016 Stanford University
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
        # root_url = 'http://llvm.org/releases/3.5.2'
        root_url = 'http://legion.stanford.edu/~eslaught/llvm-deb-mirror/releases/3.5.2'
        clang_tarball = 'clang+llvm-3.5.2-x86_64-apple-darwin.tar.xz'
        clang_dir = os.path.abspath('clang+llvm-3.5.2-x86_64-apple-darwin')

        print('%s/%s' % (root_url, clang_tarball))
        subprocess.check_call(
            ['curl', '-O', '%s/%s' % (root_url, clang_tarball)])
        shasum = subprocess.Popen(['shasum', '-c'], stdin=subprocess.PIPE)
        shasum.communicate(
            '547d9a359258ce918750fd8647cd6e1b47feaa51  %s' % clang_tarball)
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

def test(root_dir, debug, spy, env):
    subprocess.check_call(
        ['time', './install.py', '-j', '2', '--rdir=auto'] + (['--debug'] if debug else []),
        env = env,
        cwd = root_dir)
    subprocess.check_call(
        ['time', './test.py', '-q', '-j', '2'] + (['--debug'] if debug else []),
        env = env,
        cwd = root_dir)
    if spy:
        subprocess.check_call(
            ['time', './test.py', '-q', '-j', '2', '--spy'],
            env = env,
            cwd = root_dir)

if __name__ == '__main__':
    root_dir = os.path.realpath(os.path.dirname(__file__))
    legion_dir = os.path.dirname(root_dir)
    runtime_dir = os.path.join(legion_dir, 'runtime')

    env = install_dependencies()
    env.update({
        'LG_RT_DIR': runtime_dir,
        'LUAJIT_URL': 'http://legion.stanford.edu/~eslaught/mirror/LuaJIT-2.0.4.tar.gz',
    })
    # reduce output spewage by default
    if 'MAKEFLAGS' not in env:
        env['MAKEFLAGS'] = 's'

    test(root_dir, env['DEBUG'], 'TEST_SPY' in env and env['TEST_SPY'], env)
