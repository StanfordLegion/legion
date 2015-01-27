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

from __future__ import print_function
import multiprocessing, os, platform, subprocess

# Requires:
#   * Terra-compatible LLVM installation on PATH
#   * Boost including tools (Ubuntu packages libboost-dev and
#     libboost-tools-dev, or equivalent)

os_name = platform.system()

if os_name == 'Linux':
    dylib_ext = '.so'
elif os_name == 'Darwin':
    dylib_ext = '.dylib'
else:
    raise Exception('install.py script does not work on %s' % platform.system())

def download(dest_path, url, sha1):
    dest_dir = os.path.dirname(dest_path)
    dest_file = os.path.basename(dest_path)
    subprocess.check_call(['wget', '-O', dest_path, url])
    shasum = subprocess.Popen(
        ['shasum', '--check'], stdin = subprocess.PIPE, cwd = dest_dir)
    shasum.communicate('%s  %s' % (sha1, dest_file))
    assert shasum.wait() == 0

def git_clone(repo_dir, url):
    subprocess.check_call(['git', 'clone', url, repo_dir])

def git_update(repo_dir):
    subprocess.check_call(
        ['git', 'pull', '--ff-only'],
        cwd = repo_dir)

def build_terra(terra_dir):
    subprocess.check_call(
        ['make', 'all', '-j', str(multiprocessing.cpu_count())],
        cwd = terra_dir)

def install_terra(terra_dir):
    if not os.path.exists(terra_dir):
        git_clone(terra_dir, 'https://github.com/zdevito/terra.git')
    else:
        git_update(terra_dir)
    build_terra(terra_dir)

def symlink(from_path, to_path):
    if not os.path.lexists(to_path):
        os.symlink(from_path, to_path)

def build_luabind(bindings_dir, terra_dir, luabind_dir):
    diff_filename = os.path.join(bindings_dir, 'Jamroot.diff')
    # Check to see if the patch has been applied before applying it.
    with open(diff_filename, 'rb') as diff:
        patched = subprocess.call(
            ['patch', '-p0', '--forward', '--dry-run', '--silent', 'Jamroot'],
            stdin = diff,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            cwd = luabind_dir)
    if patched == 0:
        print('patching Jamroot')
        with open(diff_filename, 'rb') as diff:
            subprocess.check_call(
                ['patch', '-p0', 'Jamroot'],
                stdin = diff,
                cwd = luabind_dir)
    else:
        print('patch to Jamroot already applied')

    luajit_dir = os.path.join(terra_dir, 'build', 'LuaJIT-2.0.3', 'src')
    fake_luajit_dir = os.path.join(luabind_dir, 'fake_luajit_dir')

    if not os.path.exists(fake_luajit_dir):
        os.mkdir(fake_luajit_dir)
    symlink(luajit_dir, os.path.join(fake_luajit_dir, 'include'))
    symlink(luajit_dir, os.path.join(fake_luajit_dir, 'lib'))
    # So apparently, when LuaJIT gets built on Mac OS X, it completely
    # ignores the system and saves its dynamic libraries with a .so
    # extension. Great.
    symlink(os.path.join(luajit_dir, 'libluajit.so'),
            os.path.join(luajit_dir, 'libluajit-5.1%s' % dylib_ext))
    # For Linux:
    symlink(os.path.join(luajit_dir, 'libluajit.so'),
            os.path.join(luajit_dir, 'libluajit-5.1%s.2' % dylib_ext))
    # For Mac OS X:
    symlink(os.path.join(luajit_dir, 'libluajit.so'),
            os.path.join(luajit_dir, 'libluajit-5.1.2%s' % dylib_ext))

    luabind_install_dir = os.path.join(luabind_dir, 'install_dir')
    env = dict(os.environ.items() + [
        ('LUAJIT_PATH', fake_luajit_dir),                   # for luabind
        ('LUAJIT_DIR', fake_luajit_dir),                    # for bindings
        ('LUABIND_DIR', luabind_install_dir),               # for bindings
        ('WITH_TERRA', '1'),                                # for bindings
        ('TERRA_DIR', terra_dir),                           # for bindings
        ('WITH_TERRA_STATIC_LINKING', '1'),                 # for bindings
    ])

    subprocess.check_call(
        ['bjam', '--prefix=%s' % luabind_install_dir, 'install'],
        cwd = luabind_dir,
        env = env)
    symlink(os.path.join(luabind_install_dir, 'lib',
                         'libluabindd%s' % dylib_ext),
            os.path.join(luabind_install_dir, 'lib',
                         'libluabind%s' % dylib_ext))

    subprocess.check_call(
        ['make', 'clean'],
        cwd = bindings_dir,
        env = env)
    subprocess.check_call(
        ['make', '-j', str(multiprocessing.cpu_count())],
        cwd = bindings_dir,
        env = env)
    symlink(os.path.join(bindings_dir, 'libbinding.so'),
            os.path.join(bindings_dir, 'libbinding%s' % dylib_ext))

    # This last bit is necessary because Mac OS X shared libraries
    # have paths hard-coded into them, and in this case those paths
    # are coming out wrong. Therefore, we need to fix them to use the
    # search path again so our scripts can find them properly.
    #
    # You can sanity check that this step actually worked with the
    # commands:
    #
    # otool -L libbinding.so
    # ./legion.py
    #   =package.loadlib('libbinding.so', 'init')

    if os_name == 'Darwin':
        subprocess.check_call(
            ['install_name_tool', '-change',
             '/usr/local/lib/libluajit-5.1.2.dylib', 'libluajit-5.1.2.dylib',
             os.path.join(bindings_dir, 'libbinding.so')])

def install_luabind(legion_dir, terra_dir, luabind_dir):
    if not os.path.exists(luabind_dir):
        git_clone(luabind_dir, 'https://github.com/elliottslaughter/luabind.git')
    else:
        git_update(luabind_dir)
    build_luabind(legion_dir, terra_dir, luabind_dir)

def install():
    root_dir = os.path.realpath(os.path.dirname(__file__))
    legion_dir = os.path.dirname(root_dir)

    terra_dir = os.path.join(root_dir, 'terra')
    install_terra(terra_dir)

    bindings_dir = os.path.join(legion_dir, 'bindings', 'lua')
    luabind_dir = os.path.join(root_dir, 'luabind')
    install_luabind(bindings_dir, terra_dir, luabind_dir)

if __name__ == '__main__':
    install()
