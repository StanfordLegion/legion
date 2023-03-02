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

from __future__ import print_function
import argparse, json, multiprocessing, os, platform, re, shutil, subprocess, sys

# Requires:
#   * Terra-compatible LLVM installation on PATH

_version = sys.version_info.major

if _version == 2: # Python 2.x:
    _input = raw_input
elif _version == 3: # Python 3.x:
    _input = input
else:
    raise Exception('Incompatible Python version')

# allow the make executable name to be overridden by the environment
make_exe = os.environ.get('MAKE', 'make')

os_name = platform.system()

if os_name == 'Linux':
    dylib_ext = '.so'
elif os_name == 'Darwin':
    dylib_ext = '.dylib'
elif os_name == 'FreeBSD':
    dylib_ext = '.so'
    make_exe = os.environ.get('MAKE', 'gmake')  # default needs to be GNU make
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

def build_terra(terra_dir, terra_branch, use_cmake, cmake_exe, thread_count, llvm):
    build_dir = os.path.join(terra_dir, 'build')
    release_dir = os.path.join(terra_dir, 'release')
    if use_cmake is None:
        build_detected = os.path.exists(os.path.join(build_dir, 'main.o'))
        cmake_detected = os.path.exists(os.path.join(build_dir, 'CMakeCache.txt'))
        use_cmake = cmake_detected or not build_detected
        if not use_cmake:
            print('Detected previous Makefile build in Terra, disabling Terra CMake build...')

    flags = []
    if llvm:
        assert not use_cmake, "LLVM mode not supported with Terra CMake build, see https://github.com/zdevito/terra/issues/394"
        flags.extend(['REEXPORT_LLVM_COMPONENTS=irreader mcjit x86'])

    if use_cmake:
        if not os.path.exists(os.path.join(build_dir, 'CMakeCache.txt')):
            subprocess.check_call(
                [cmake_exe, '..', '-DCMAKE_INSTALL_PREFIX=%s' % release_dir],
                cwd=build_dir)
        subprocess.check_call(
            [make_exe, 'install', '-j', str(thread_count)],
            cwd=build_dir)
    else:
        subprocess.check_call(
            [make_exe, 'all', '-j', str(thread_count)] + flags,
            cwd=terra_dir)

def install_terra(terra_dir, terra_url, terra_branch, use_cmake, cmake_exe,
                  external_terra_dir, thread_count, llvm):
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
            terra_url = 'https://github.com/terralang/terra.git'
        if terra_branch is None:
            terra_branch = 'master'
        git_clone(terra_dir, terra_url, terra_branch)
    else:
        if terra_url is not None or terra_branch is not None:
            raise Exception('Terra URL/branch must be set on first install, please delete the terra directory and try again')
        git_update(terra_dir)
    build_terra(terra_dir, terra_branch, use_cmake, cmake_exe, thread_count, llvm)

def install_luarocks(terra_dir, luarocks_dir):
    if not os.path.exists(luarocks_dir):
        # For now we need to use Git until the following patch makes
        # it into a release:
        # https://github.com/luarocks/luarocks/commit/708fed20d013e69fd79d80f0b59a45a25eed3a00
        luarocks_url = 'https://github.com/luarocks/luarocks.git'
        luarocks_branch = 'master'
        git_clone(luarocks_dir, luarocks_url, luarocks_branch)

    if os.path.exists(os.path.join(terra_dir, 'bin', 'terra')):
        terra_prefix = os.path.join(terra_dir)
    elif os.path.exists(os.path.join(terra_dir, 'release', 'bin', 'terra')):
        terra_prefix = os.path.join(terra_dir, 'release')
    else:
        raise Exception('Unable to determine correct prefix for LuaRocks installation')

    luarocks_prefix = os.path.join(luarocks_dir, 'install')
    luarocks_exe = os.path.join(luarocks_prefix, 'bin', 'luarocks')
    if not os.path.exists(luarocks_exe):
        subprocess.check_call(
            [os.path.join(luarocks_dir, 'configure'),
             '--prefix=%s' % luarocks_prefix,
             '--with-lua=%s' % terra_prefix,
             '--with-lua-include=%s' % os.path.join(terra_prefix, 'include', 'terra'),
             '--with-lua-interpreter=terra'],
            cwd=luarocks_dir)
        # Hack: This throws an error but we'll keep going anyway...
        subprocess.call(['make'], cwd=luarocks_dir)
        subprocess.check_call(['make', 'install'], cwd=luarocks_dir)

    ldoc_exe = os.path.join(luarocks_prefix, 'bin', 'ldoc')
    if not os.path.exists(ldoc_exe):
        ldoc_url = 'https://raw.githubusercontent.com/StanfordLegion/LDoc/master/ldoc-scm-2.rockspec'
        subprocess.check_call([luarocks_exe, 'install', ldoc_url])

def symlink(from_path, to_path):
    if not os.path.lexists(to_path):
        os.symlink(from_path, to_path)

def install_bindings(regent_dir, legion_dir, bindings_dir, python_bindings_dir, runtime_dir,
                     cmake, cmake_exe, build_dir,
                     debug, cuda, hip, openmp, python, llvm, hdf, spy,
                     gasnet, gasnet_dir, conduit, clean_first,
                     extra_flags, thread_count, verbose):
    # Don't blow away an existing directory
    assert not (clean_first and build_dir is not None)

    # If building support for CUDA then check CUDA version is not blacklisted
    # CUDA 9.2 and 10.0 have thrust bugs that break complex support
    if cuda:
        try:
            nvcc_version = subprocess.check_output([os.environ['CUDA']+'/bin/nvcc', '--version']).decode('utf-8')
        except (KeyError,FileNotFoundError,subprocess.CalledProcessError):
            try:
                nvcc_version = subprocess.check_output(["nvcc", "--version"]).decode('utf-8')
            except (FileNotFoundError,subprocess.CalledProcessError):
                print('Error: Unable to verify CUDA version is not blacklisted for Regent')
                sys.exit(1)
        pattern = re.compile(' V(?P<major>[0-9]+)\.(?P<minor>[0-9]+)')
        major_version = None
        minor_version = None
        for line in nvcc_version.splitlines():
            match = pattern.search(line)
            if match is None:
                continue
            major_version = int(match.group('major'))
            minor_version = int(match.group('minor'))
            break
        if major_version is None:
            print('Error: Unabled to verify CUDA version is not blacklisted for Regent')
            sys.exit(1)
        elif (major_version == 9 and minor_version == 2) or \
                (major_version == 10 and minor_version == 0):
            print('Error: CUDA version '+str(major_version)+'.'+
                    str(minor_version)+' is blacklisted for Regent due '+
                    'to a Thrust bug that breaks complex number support. '+
                    'Please either upgrade or downgrade your version '+
                    'of CUDA to a version that is not 9.2 or 10.0')
            sys.exit(1)
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
             '-DLegion_USE_HIP=%s' % ('ON' if hip else 'OFF'),
             '-DLegion_USE_OpenMP=%s' % ('ON' if openmp else 'OFF'),
             '-DLegion_USE_Python=%s' % ('ON' if python else 'OFF'),
             '-DLegion_USE_LLVM=%s' % ('ON' if llvm else 'OFF'),
             '-DLegion_USE_GASNet=%s' % ('ON' if gasnet else 'OFF'),
             '-DLegion_USE_HDF5=%s' % ('ON' if hdf else 'OFF'),
             '-DLegion_SPY=%s' % ('ON' if spy else 'OFF'),
             '-DLegion_BUILD_BINDINGS=ON',
             '-DBUILD_SHARED_LIBS=ON',
            ] +
            (['-DHIP_THRUST_ROOT_DIR=%s' % os.environ['THRUST_PATH']] if 'THRUST_PATH' in os.environ else []) +
            extra_flags +
            (['-DGASNet_ROOT_DIR=%s' % gasnet_dir] if gasnet_dir is not None else []) +
            (['-DGASNet_CONDUIT=%s' % conduit] if conduit is not None else []) +
            (['-DLegion_NETWORKS=%s' % os.environ['REALM_NETWORKS']] if 'REALM_NETWORKS' in os.environ else []))
        if llvm:
            # mess with a few things so that Realm uses terra's LLVM
            flags.append('-DLegion_ALLOW_MISSING_LLVM_LIBS=ON')
            flags.append('-DLegion_LINK_LLVM_LIBS=OFF')
            # pass through LLVM_CONFIG, if set
            if 'LLVM_CONFIG' in os.environ:
                flags.append('-DLLVM_CONFIG_EXECUTABLE=%s' % os.environ['LLVM_CONFIG'])
        make_flags = ['VERBOSE=1'] if verbose else []
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
            [make_exe] + make_flags + ['-j', str(thread_count)],
            cwd=build_dir)
    else:
        flags = (
            ['LG_RT_DIR=%s' % runtime_dir,
             'DEFINE_HEADERS_DIR=%s' % bindings_dir, # otherwise Python build recompiles everything
             'DEBUG=%s' % (1 if debug else 0),
             'USE_CUDA=%s' % (1 if cuda else 0),
             'USE_HIP=%s' % (1 if hip else 0),
             'USE_OPENMP=%s' % (1 if openmp else 0),
             'USE_PYTHON=%s' % (1 if python else 0),
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
                [make_exe] + flags + ['clean'],
                cwd=bindings_dir)
            if python:
                subprocess.check_call(
                    [make_exe] + flags + ['clean'],
                    cwd=python_bindings_dir)
        subprocess.check_call(
            [make_exe] + flags + ['-j', str(thread_count)],
            cwd=bindings_dir)
        if python:
            subprocess.check_call(
                [make_exe] + flags + ['-j', str(thread_count)],
                cwd=python_bindings_dir)

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
                 os.path.join(bindings_dir, 'libregent.dylib')])

def get_cmake_config(cmake, regent_dir, default=None):
    config_filename = os.path.join(regent_dir, '.cmake.json')
    if cmake is None:
        cmake = load_json_config(config_filename)
        if cmake is None:
            cmake = default
    assert cmake in [True, False]
    dump_json_config(config_filename, cmake)
    return cmake

def get_legion_install_prefix(legion_install_prefix, regent_dir, default=None):
    config_filename = os.path.join(regent_dir, '.legion_install_prefix.json')
    if legion_install_prefix is None:
        legion_install_prefix = load_json_config(config_filename)
        if legion_install_prefix is None:
            legion_install_prefix = default
    assert isinstance(legion_install_prefix, str)
    legion_install_prefix = os.path.abspath(legion_install_prefix)
    dump_json_config(config_filename, legion_install_prefix)
    return legion_install_prefix

def install(gasnet=False, cuda=False, hip=False, openmp=False, python=False, llvm=False, hdf=False,
            spy=False, conduit=None, cmake=None, rdir=None,
            cmake_exe=None, cmake_build_dir=None,
            legion_install_prefix=None,
            terra_url=None, terra_branch=None, terra_use_cmake=None, external_terra_dir=None,
            gasnet_dir=None, debug=False, clean_first=True, extra_flags=[],
            thread_count=None, verbose=False):
    regent_dir = os.path.dirname(os.path.realpath(__file__))
    legion_dir = os.path.dirname(regent_dir)

    cmake = get_cmake_config(cmake, regent_dir, default=False)
    legion_install_prefix = get_legion_install_prefix(legion_install_prefix, regent_dir)

    if clean_first is None:
        clean_first = not cmake

    if not cmake and cmake_build_dir is not None:
        raise Exception('Build directory is only permitted when building with CMake')

    if clean_first and cmake_build_dir is not None:
        raise Exception('Cannot clean a pre-existing build directory')

    if legion_install_prefix:
        if cmake:
            raise Exception('Cannot build with CMake, Legion is already installed')
        if len(extra_flags) > 0:
            raise Exception('Cannot build with extra flags, Legion is already installed')

    if thread_count is None:
        try:
            # this correctly considers the current affinity mask
            thread_count = len(os.sched_getaffinity(0))
        except AttributeError:
            # this works on macos
            thread_count = multiprocessing.cpu_count()

    # Grab LG_RT_DIR from the environment if available, otherwise
    # assume we're running relative to our own location.
    runtime_dir = os.path.join(legion_dir, 'runtime')
    if 'LG_RT_DIR' in os.environ:
        runtime_dir = os.path.realpath(os.environ['LG_RT_DIR'])

    install_rdir(rdir, legion_dir, regent_dir)

    terra_dir = os.path.join(regent_dir, 'terra')
    install_terra(terra_dir, terra_url, terra_branch, terra_use_cmake, cmake_exe,
                  external_terra_dir, thread_count, llvm)
    # luarocks_dir = os.path.join(regent_dir, 'luarocks')
    # install_luarocks(terra_dir, luarocks_dir)

    if legion_install_prefix is None:
        bindings_dir = os.path.join(legion_dir, 'bindings', 'regent')
        python_bindings_dir = os.path.join(legion_dir, 'bindings', 'python')
        install_bindings(regent_dir, legion_dir, bindings_dir, python_bindings_dir, runtime_dir,
                         cmake, cmake_exe, cmake_build_dir,
                         debug, cuda, hip, openmp, python, llvm, hdf, spy,
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
        '--terra-cmake', dest='terra_use_cmake', action='store_true', required=False,
        default=None,
        help='Build Terra with CMake.')
    parser.add_argument(
        '--no-terra-cmake', dest='terra_use_cmake', action='store_false', required=False,
        default=None,
        help="Don't build Terra with CMake (instead use GNU Make).")
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
        '--hip', dest='hip', action='store_true', required=False,
        default=os.environ.get('USE_HIP') == '1',
        help='Build Legion with HIP.')
    parser.add_argument(
        '--openmp', dest='openmp', action='store_true', required=False,
        default=os.environ.get('USE_OPENMP') == '1',
        help='Build Legion with OpenMP support.')
    parser.add_argument(
        '--python', dest='python', action='store_true', required=False,
        default=os.environ.get('USE_PYTHON') == '1',
        help='Build Legion with Python support.')
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
        '--legion-install-prefix', dest='legion_install_prefix', metavar='DIR', required=False,
        help='Do NOT build Legion. Just use the specified installation.')
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
