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

from __future__ import print_function
import argparse, json, multiprocessing, os, platform, shutil, subprocess, sys

# Requires:
#   * Terra-compatible LLVM installation on PATH

_version = sys.version_info.major

if _version == 2: # Python 2.x:
    _input = raw_input
elif _version == 3: # Python 3.x:
    _input = input
else:
    raise Exception('Incompatible Python version')

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
        ['shasum', '--check'], stdin=subprocess.PIPE, cwd=dest_dir)
    shasum.communicate('%s  %s' % (sha1, dest_file))
    assert shasum.wait() == 0

def git_clone(repo_dir, url, branch=None):
    subprocess.check_call(['git', 'clone'] +
                          (['-b', branch] if branch else []) +
                          [url, repo_dir])

def git_update(repo_dir):
    subprocess.check_call(
        ['git', 'pull', '--ff-only'],
        cwd=repo_dir)

def git_submodule_update(repo_dir):
    subprocess.check_call(
        ['git', 'submodule', 'update', '--init'],
        cwd=repo_dir)

def load_json_config(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except IOError:
        return None

def dump_json_config(filename, value):
    with open(filename, 'w') as f:
        return json.dump(value, f)

prompt_text = '''
RDIR is an optional compiler plugin for Regent which provides support
for dataflow optimizations (most notably control replication). RDIR
support is opt-in because RDIR's license is different from that of
Regent (thus this prompt). Specifically:

  * portions of RDIR are licensed under BSD
  * other portions of RDIR are dual-licensed under BSD and Apache

(Regent itself is licensed entirely under Apache.)

You may choose to use RDIR automatically (select "auto" below),
manually, or not at all. Your preference will be saved. You can change
your mind at any time by re-running this script with the "--rdir"
parameter.
'''

def install_rdir(rdir, legion_dir, regent_dir):
    config_filename = os.path.join(regent_dir, '.rdir.json')
    if rdir is None:
        rdir = load_json_config(config_filename)
        if rdir is None: rdir = 'prompt'

    if rdir == 'prompt':
        print(prompt_text)
        while rdir not in ['auto', 'manual', 'never']:
            rdir = _input('Enable RDIR? (auto/manual/never) ')
    assert rdir in ['auto', 'manual', 'skip', 'never']

    if rdir == 'auto':
        git_submodule_update(legion_dir)

    if rdir != 'skip':
        dump_json_config(config_filename, rdir)

def build_terra(terra_dir, thread_count, llvm):
    subprocess.check_call(
        ['make', 'all', '-j', str(thread_count)] +
        (['REEXPORT_LLVM_COMPONENTS=irreader mcjit x86'] if llvm else []),
        cwd=terra_dir)

def install_terra(terra_dir, terra_url, terra_branch, external_terra_dir,
                  thread_count, llvm):
    if external_terra_dir is not None:
        if terra_url is not None or terra_branch is not None:
            raise Exception('Terra URL/branch are incompatible with setting an external installation directory')

        external_terra_dir = os.path.expanduser(external_terra_dir)
        if not os.path.isdir(external_terra_dir):
            print('Error: No such directory %s' %
                  external_terra_dir)
            sys.exit(1)
        if os.path.lexists(terra_dir):
            if not os.path.islink(terra_dir):
                print('Error: Attempting build with external Terra when internal Terra')
                print('already exists. Please remove the following directory to continue with')
                print('an external Terra installation.')
                print('    %s' % terra_dir)
                sys.exit(1)
            if os.path.realpath(terra_dir) != os.path.realpath(external_terra_dir):
                os.unlink(terra_dir)
                os.symlink(external_terra_dir, terra_dir)
        else:
            print(external_terra_dir, terra_dir)
            os.symlink(external_terra_dir, terra_dir)
        return
    elif os.path.islink(terra_dir):
        if terra_url is not None or terra_branch is not None:
            raise Exception('Terra URL/branch are incompatible with setting an external installation directory')

        print('Reusing existing external Terra:')
        print('    %s' % os.path.realpath(terra_dir))
        print()
        return

    if not os.path.exists(terra_dir):
        if terra_url is None:
            terra_url = 'https://github.com/zdevito/terra.git'
        if terra_branch is None:
            terra_branch = 'master'
        git_clone(terra_dir, terra_url, terra_branch)
    else:
        if terra_url is not None or terra_branch is not None:
            raise Exception('Terra URL/branch must be set on first install, please delete the terra directory and try again')
        git_update(terra_dir)
    build_terra(terra_dir, thread_count, llvm)

def symlink(from_path, to_path):
    if not os.path.lexists(to_path):
        os.symlink(from_path, to_path)

def install_bindings(regent_dir, legion_dir, bindings_dir, runtime_dir,
                     cmake, cmake_exe, build_dir,
                     debug, cuda, openmp, llvm, hdf, spy,
                     gasnet, gasnet_dir, conduit, clean_first,
                     extra_flags, thread_count, verbose):
    # Don't blow away an existing directory
    assert not (clean_first and build_dir is not None)

    if cmake:
        regent_build_dir = os.path.join(regent_dir, 'build')
        if build_dir is None:
            build_dir = regent_build_dir
        else:
            try:
                # check if the link is already there (and pointing at the right
                #  thing) first
                if not os.path.islink(regent_build_dir) or (os.readlink(regent_build_dir) != build_dir):
                    os.symlink(build_dir, regent_build_dir)
            except OSError:
                print('Error: Attempting to build with an external build directory when an')
                print('internal (or different external) build directory already exists. Please')
                print('remove the following directory to continue with the installation:')
                print('    %s' % regent_build_dir)
                sys.exit(1)
        if clean_first:
            shutil.rmtree(build_dir)
        if not os.path.exists(build_dir):
            os.mkdir(build_dir)
        flags = (
            ['-DCMAKE_BUILD_TYPE=%s' % ('Debug' if debug else 'Release'),
             '-DLegion_USE_CUDA=%s' % ('ON' if cuda else 'OFF'),
             '-DLegion_USE_OpenMP=%s' % ('ON' if openmp else 'OFF'),
             '-DLegion_USE_LLVM=%s' % ('ON' if llvm else 'OFF'),
             '-DLegion_USE_GASNet=%s' % ('ON' if gasnet else 'OFF'),
             '-DLegion_USE_HDF5=%s' % ('ON' if hdf else 'OFF'),
             '-DLegion_BUILD_BINDINGS=ON',
             '-DBUILD_SHARED_LIBS=ON',
            ] +
            extra_flags +
            (['-DGASNet_ROOT_DIR=%s' % gasnet_dir] if gasnet_dir is not None else []) +
            (['-DGASNet_CONDUIT=%s' % conduit] if conduit is not None else []) +
            (['-DCMAKE_CXX_COMPILER=%s' % os.environ['CXX']] if 'CXX' in os.environ else []) +
            (['-DCMAKE_CXX_FLAGS=%s' % os.environ['CC_FLAGS']] if 'CC_FLAGS' in os.environ else []))
        make_flags = ['VERBOSE=1'] if verbose else []
        assert not spy # unimplemented
        try:
            subprocess.check_output([cmake_exe, '--version'])
        except OSError:
            print('Error: CMake is not installed or otherwise not executable. Please check')
            print('your CMake installation and try again. You can use the --with-cmake flag')
            print('to specify the CMake executable if it is not on PATH.')
            print()
            print('Attempted to execute: %s' % cmake_exe)
            sys.exit(1)
        subprocess.check_call(
            [cmake_exe] + flags + [legion_dir],
            cwd=build_dir)
        subprocess.check_call(
            ['make'] + make_flags + ['-j', str(thread_count)],
            cwd=build_dir)
    else:
        flags = (
            ['LG_RT_DIR=%s' % runtime_dir,
             'DEBUG=%s' % (1 if debug else 0),
             'USE_CUDA=%s' % (1 if cuda else 0),
             'USE_OPENMP=%s' % (1 if openmp else 0),
             'USE_LLVM=%s' % (1 if llvm else 0),
             'USE_GASNET=%s' % (1 if gasnet else 0),
             'USE_HDF=%s' % (1 if hdf else 0),
             'USE_SPY=%s' % (1 if spy else 0),
             ] +
            extra_flags +
            (['GASNET=%s' % gasnet_dir] if gasnet_dir is not None else []) +
            (['CONDUIT=%s' % conduit] if conduit is not None else []) +
            (['GCC=%s' % os.environ['CXX']] if 'CXX' in os.environ else []))

        if clean_first:
            subprocess.check_call(
                ['make'] + flags + ['clean'],
                cwd=bindings_dir)
        subprocess.check_call(
            ['make'] + flags + ['-j', str(thread_count)],
            cwd=bindings_dir)
        symlink(os.path.join(bindings_dir, 'libregent.so'),
                os.path.join(bindings_dir, 'libregent%s' % dylib_ext))

        # This last bit is necessary because Mac OS X shared libraries
        # have paths hard-coded into them, and in this case those paths
        # are coming out wrong. Therefore, we need to fix them to use the
        # search path again so our scripts can find them properly.
        #
        # You can sanity check that this step actually worked with the
        # commands:
        #
        # otool -L libregent.so
        # ./regent.py
        #   =package.loadlib('libregent.so', 'init')

        if os_name == 'Darwin':
            subprocess.check_call(
                ['install_name_tool', '-change',
                 '/usr/local/lib/libluajit-5.1.2.dylib', 'libluajit-5.1.2.dylib',
                 os.path.join(bindings_dir, 'libregent.so')])

def get_cmake_config(cmake, regent_dir, default=None):
    config_filename = os.path.join(regent_dir, '.cmake.json')
    if cmake is None:
        cmake = load_json_config(config_filename)
        if cmake is None:
            cmake = default
    assert cmake in [True, False]
    dump_json_config(config_filename, cmake)
    return cmake

def install(gasnet=False, cuda=False, openmp=False, hdf=False, llvm=False,
            spy=False, conduit=None, cmake=None, rdir=None,
            cmake_exe=None, cmake_build_dir=None,
            terra_url=None, terra_branch=None, external_terra_dir=None,
            gasnet_dir=None, debug=False, clean_first=True, extra_flags=[],
            thread_count=None, verbose=False):
    regent_dir = os.path.dirname(os.path.realpath(__file__))
    legion_dir = os.path.dirname(regent_dir)

    cmake = get_cmake_config(cmake, regent_dir, default=False)

    if clean_first is None:
        clean_first = not cmake

    if not cmake and cmake_build_dir is not None:
        raise Exception('Build directory is only permitted when building with CMake')

    if clean_first and cmake_build_dir is not None:
        raise Exception('Cannot clean a pre-existing build directory')

    if spy and not debug:
        raise Exception('Debugging mode is required for detailed Legion Spy.')

    thread_count = thread_count
    if thread_count is None:
        thread_count = multiprocessing.cpu_count()

    # Grab LG_RT_DIR from the environment if available, otherwise
    # assume we're running relative to our own location.
    runtime_dir = os.path.join(legion_dir, 'runtime')
    if 'LG_RT_DIR' in os.environ:
        runtime_dir = os.path.realpath(os.environ['LG_RT_DIR'])

    install_rdir(rdir, legion_dir, regent_dir)

    terra_dir = os.path.join(regent_dir, 'terra')
    install_terra(terra_dir, terra_url, terra_branch, external_terra_dir,
                  thread_count, llvm)

    bindings_dir = os.path.join(legion_dir, 'bindings', 'regent')
    install_bindings(regent_dir, legion_dir, bindings_dir, runtime_dir,
                     cmake, cmake_exe, cmake_build_dir,
                     debug, cuda, openmp, llvm, hdf, spy,
                     gasnet, gasnet_dir, conduit, clean_first,
                     extra_flags, thread_count, verbose)

def driver():
    parser = argparse.ArgumentParser(
        description='Install Regent front end.')
    parser.add_argument(
        '--terra-url', dest='terra_url', metavar='URL', required=False,
        help='URL to Terra repository to clone (optional).')
    parser.add_argument(
        '--terra-branch', dest='terra_branch', metavar='BRANCH', required=False,
        help='Name of Terra branch to clone (optional).')
    parser.add_argument(
        '--with-terra', dest='external_terra_dir', metavar='DIR', required=False,
        help='Path to Terra installation directory (optional).')
    parser.add_argument(
        '--debug', dest='debug', action='store_true', required=False,
        default=os.environ.get('DEBUG') == '1',
        help='Build Legion with debugging enabled.')
    parser.add_argument(
        '--gasnet', dest='gasnet', action='store_true', required=False,
        default=os.environ.get('USE_GASNET') == '1',
        help='Build Legion with GASNet.')
    parser.add_argument(
        '--with-gasnet', dest='gasnet_dir', metavar='DIR', required=False,
        default=os.environ.get('GASNET'),
        help='Path to GASNet installation directory.')
    parser.add_argument(
        '--cuda', dest='cuda', action='store_true', required=False,
        default=os.environ.get('USE_CUDA') == '1',
        help='Build Legion with CUDA.')
    parser.add_argument(
        '--openmp', dest='openmp', action='store_true', required=False,
        default=os.environ.get('USE_OPENMP') == '1',
        help='Build Legion with OpenMP support.')
    parser.add_argument(
        '--llvm', dest='llvm', action='store_true', required=False,
        default=os.environ.get('USE_LLVM') == '1',
        help='Build Legion (and compatible Terra) with LLVM support.')
    parser.add_argument(
        '--hdf5', '--hdf', dest='hdf', action='store_true', required=False,
        default=os.environ.get('USE_HDF') == '1',
        help='Build Legion with HDF.')
    parser.add_argument(
        '--spy', dest='spy', action='store_true', required=False,
        default=os.environ.get('USE_SPY') == '1',
        help='Build Legion with detailed Legion Spy enabled.')
    parser.add_argument(
        '--conduit', dest='conduit', action='store', required=False,
        default=os.environ.get('CONDUIT'),
        help='Build Legion with specified GASNet conduit.')
    parser.add_argument(
        '--cmake', dest='cmake', action='store_true', required=False,
        default=os.environ['USE_CMAKE'] == '1' if 'USE_CMAKE' in os.environ else None,
        help='Build Legion with CMake.')
    parser.add_argument(
        '--no-cmake', dest='cmake', action='store_false', required=False,
        help="Don't build Legion with CMake (instead use GNU Make).")
    parser.add_argument(
        '--with-cmake', dest='cmake_exe', metavar='EXE', required=False,
        default='cmake',
        help='Path to CMake executable (if not on PATH).')
    parser.add_argument(
        '--with-cmake-build', dest='cmake_build_dir', metavar='DIR', required=False,
        help='Path to CMake build directory (optional).')
    parser.add_argument(
        '--rdir', dest='rdir', required=False,
        choices=['prompt', 'auto', 'manual', 'skip', 'never'], default=None,
        help='Enable RDIR compiler plugin.')
    parser.add_argument(
        '--clean', dest='clean_first', action='store_true', required=False,
        default=None,
        help='Clean before build.')
    parser.add_argument(
        '--no-clean', '--noclean', dest='clean_first', action='store_false', required=False,
        help='Skip clean before build.')
    parser.add_argument(
        '--extra', dest='extra_flags', action='append', required=False,
        default=[],
        help='Extra flags for make command.')
    parser.add_argument(
        '-j', dest='thread_count', nargs='?', type=int,
        help='Number threads used to compile.')
    parser.add_argument(
        '-v', '--verbose', dest='verbose', action='store_true', required=False,
        help='Enable verbose build output.')
    args = parser.parse_args()

    install(**vars(args))

if __name__ == '__main__':
    driver()
