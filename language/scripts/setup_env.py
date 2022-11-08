#!/usr/bin/env python3

# Copyright 2022 Stanford University
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
import argparse, hashlib, multiprocessing, os, platform, re, subprocess, sys, tempfile, traceback

def discover_llvm_version():
    # standardize on LLVM 13.0 everywhere
    return '130'

def discover_skip_certificate_check():
    # Elliott: I don't think any systems are sensitive to this anymore
    return False

def discover_conduit():
    if 'CONDUIT' in os.environ:
        return os.environ['CONDUIT']
    elif platform.node().startswith('cori'):
        return 'aries'
    elif platform.node().startswith('daint'):
        return 'aries'
    elif platform.node().startswith('excalibur'):
        return 'aries'
    elif platform.node().startswith('quartz'):
        return 'psm'
    elif os.environ.get('LMOD_SYSTEM_NAME') == 'summit': # Summit doesn't set hostname
        return 'ibv'
    elif os.environ.get('LMOD_SYSTEM_NAME') == 'crusher': # Crusher doesn't set hostname
        return 'ofi-slingshot11'
    elif os.environ.get('NERSC_HOST') == 'perlmutter':
        return 'ofi-slingshot11'
    else:
        raise Exception('Please set CONDUIT in your environment')

def short_conduit(conduit):
    if conduit is not None and conduit.startswith('ofi-'):
        return 'ofi'
    return conduit

def gasnet_enabled():
    if 'USE_GASNET' in os.environ:
        return os.environ['USE_GASNET'] == '1'
    return platform.system() != 'Darwin'

def hdf_enabled():
    return 'USE_HDF' in os.environ and os.environ['USE_HDF'] == '1'

def check_sha256(file_path, sha256):
    with open(file_path, 'rb') as f:
        assert hashlib.sha256(f.read()).hexdigest() == sha256

def download(dest_path, url, sha256, insecure=False):
    dest_dir = os.path.dirname(dest_path)
    dest_file = os.path.basename(dest_path)
    insecure_flag = []
    if insecure:
        insecure_flag = ['--insecure']

    if os.path.exists(dest_path):
        check_sha256(dest_path, sha256)
        return

    subprocess.check_call(['curl', '-L'] + insecure_flag + ['-o', dest_path, url])
    check_sha256(dest_path, sha256)

def extract(dest_dir, archive_path, format):
    if format == 'gz':
        subprocess.check_call(['tar', 'xfz', archive_path], cwd=dest_dir)
    elif format == 'xz':
        subprocess.check_call(['tar', 'xfJ', archive_path], cwd=dest_dir)
    else:
        raise Exception('Unknown format %s' % format)

def apply_patch(dest_dir, diff_path, strip_levels=1):
    subprocess.check_call(['patch', '-p%d' % strip_levels, '-i', diff_path], cwd=dest_dir)

def git_clone(repo_dir, url, branch=None, commit_id=None):
    if branch is not None:
        assert commit_id is None
        subprocess.check_call(['git', 'clone', '-b', branch, url, repo_dir])
    elif commit_id is not None:
        subprocess.check_call(['git', 'clone', url, repo_dir])
        subprocess.check_call(['git', '-C', repo_dir, 'checkout', commit_id])
    else:
        subprocess.check_call(['git', 'clone', url, repo_dir])

def git_update(repo_dir):
    subprocess.check_call(
        ['git', 'pull', '--ff-only'],
        cwd=repo_dir)

def build_gasnet(gasnet_dir, conduit, gasnet_version):
    subprocess.check_call(
        ['make',
         'CONDUIT=%s' % conduit,
         'GASNET_VERSION=%s' % gasnet_version],
        cwd=gasnet_dir)

def build_llvm(source_dir, build_dir, install_dir, is_project_build, cmake_exe, thread_count, is_cray):
    env = None
    if is_cray:
        env = dict(list(os.environ.items()) + [
            ('CC', os.environ['HOST_CC']),
            ('CXX', os.environ['HOST_CXX']),
        ])
    extra_flags = []
    if is_project_build:
        extra_flags = [
            '-DLLVM_ENABLE_PROJECTS=clang;lld',
            '-DLLVM_ENABLE_RUNTIMES=libunwind',
        ]
    subprocess.check_call(
        [cmake_exe,
         '-DCMAKE_INSTALL_PREFIX=%s' % install_dir,
         '-DCMAKE_BUILD_TYPE=Release',
         '-DLLVM_ENABLE_ASSERTIONS=OFF',
         '-DLLVM_ENABLE_ZLIB=OFF',
         '-DLLVM_ENABLE_LIBXML2=OFF',
         '-DLLVM_ENABLE_TERMINFO=OFF',
         '-DLLVM_ENABLE_LIBEDIT=OFF'] +
        extra_flags +
        [source_dir],
        cwd=build_dir,
        env=env)
    subprocess.check_call(['make', '-j', str(thread_count)], cwd=build_dir)
    subprocess.check_call(['make', 'install'], cwd=build_dir)

def build_terra(terra_dir, terra_branch, terra_lua, use_cmake, cmake_exe, llvm_dir, cache, is_cray, thread_count):
    if cache:
        assert not use_cmake
        subprocess.check_call(['make', 'download'], cwd=terra_dir)
        return

    env = dict(list(os.environ.items()))
    if is_cray:
        env.update(dict([
            ('CC', os.environ['HOST_CC']),
            ('CXX', os.environ['HOST_CXX']),
        ]))

    if terra_lua is not None:
        assert use_cmake

    if use_cmake:
        flags = [
            '-DCMAKE_PREFIX_PATH=%s' % llvm_dir,
            '-DCMAKE_INSTALL_PREFIX=%s' % os.path.join(terra_dir, 'release'),
            '-DTERRA_LUA=%s' % (terra_lua or 'moonjit'),
        ]
        subprocess.check_call(
            [cmake_exe] + flags + [terra_dir],
            cwd=os.path.join(terra_dir, 'build'),
            env=env)
        subprocess.check_call(
            ['make', 'install', '-j', str(thread_count)],
            cwd=os.path.join(terra_dir, 'build'),
            env=env)
    else:
        flags = [
            'LLVM_CONFIG=%s' % os.path.join(llvm_dir, 'bin', 'llvm-config'),
            'CLANG=%s' % os.path.join(llvm_dir, 'bin', 'clang'),
        ]
        if platform.system() != 'Darwin':
            flags.append('REEXPORT_LLVM_COMPONENTS=irreader mcjit x86')
        flags.extend(['-j', str(thread_count)])

        subprocess.check_call(
            ['make'] + flags,
            cwd=terra_dir,
            env=env)

def build_hdf(source_dir, install_dir, thread_count, is_cray):
    env = None
    if is_cray:
        env = dict(list(os.environ.items()) + [
            ('CC', os.environ['HOST_CC']),
            ('CXX', os.environ['HOST_CXX']),
        ])
    subprocess.check_call(
        ['./configure',
         '--prefix=%s' % install_dir,
         '--enable-threadsafe',
         '--disable-hl'],
        cwd=source_dir,
        env=env)
    subprocess.check_call(['make', '-j', str(thread_count)], cwd=source_dir)
    subprocess.check_call(['make', 'install'], cwd=source_dir)

def build_regent(root_dir, use_cmake, cmake_exe, extra_flags,
                 gasnet_dir, llvm_dir, terra_dir, hdf_dir, conduit, thread_count):
    conduit = short_conduit(conduit)

    env = dict(list(os.environ.items()) +
        ([('CONDUIT', conduit),
          ('GASNET', gasnet_dir),
          ('USE_GASNET', '1')]
         if gasnet_enabled() else []) +
        ([('HDF_ROOT', hdf_dir),
          ('USE_HDF', '1')]
         if hdf_enabled() else []) +
        [('LLVM_CONFIG', os.path.join(llvm_dir, 'bin', 'llvm-config')),
         ('CMAKE_PREFIX_PATH', llvm_dir)]
    )

    subprocess.check_call(
        [os.path.join(root_dir, 'install.py'),
         '--with-terra', terra_dir,
         '--rdir', 'auto',
         '-j', str(thread_count),
        ] + (['--cmake', '--with-cmake', cmake_exe]
             if use_cmake else ['--no-cmake']) +
        ['--extra=%s' % flag for flag in extra_flags],
        env=env)

def install_llvm(llvm_dir, llvm_install_dir, scratch_dir, llvm_version, cmake_exe, thread_count, cache, is_cray, insecure):
    try:
        os.mkdir(llvm_dir)
    except OSError:
        pass # Hope this means it already exists
    assert(os.path.isdir(llvm_dir))

    mirror = 'https://github.com/llvm/llvm-project/releases/download'
    if llvm_version == '110':
        llvm_tarball = os.path.join(llvm_dir, 'llvm-11.1.0.src.tar.xz')
        llvm_source_dir = os.path.join(llvm_dir, 'llvm-11.1.0.src')
        clang_tarball = os.path.join(llvm_dir, 'clang-11.1.0.src.tar.xz')
        clang_source_dir = os.path.join(llvm_dir, 'clang-11.1.0.src')
        download(llvm_tarball, '%s/llvmorg-11.1.0/llvm-11.1.0.src.tar.xz' % mirror, 'ce8508e318a01a63d4e8b3090ab2ded3c598a50258cc49e2625b9120d4c03ea5', insecure=insecure)
        download(clang_tarball, '%s/llvmorg-11.1.0/clang-11.1.0.src.tar.xz' % mirror, '0a8288f065d1f57cb6d96da4d2965cbea32edc572aa972e466e954d17148558b', insecure=insecure)
    elif llvm_version == '130':
        llvm_tarball = os.path.join(llvm_dir, 'llvm-project-13.0.0.src.tar.xz')
        llvm_source_dir = os.path.join(llvm_dir, 'llvm-project-13.0.0.src', 'llvm')
        clang_tarball = None
        download(llvm_tarball, '%s/llvmorg-13.0.0/llvm-project-13.0.0.src.tar.xz' % mirror, '6075ad30f1ac0e15f07c1bf062c1e1268c241d674f11bd32cdf0e040c71f2bf3', insecure=insecure)
    else:
        assert False

    if not cache:
        extract(llvm_dir, llvm_tarball, 'xz')
        if clang_tarball:
            extract(llvm_dir, clang_tarball, 'xz')
        if clang_tarball:
            os.rename(clang_source_dir, os.path.join(llvm_source_dir, 'tools', 'clang'))

        llvm_build_dir = tempfile.mkdtemp(prefix='setup_env_llvm_build', dir=scratch_dir or llvm_dir)
        os.mkdir(llvm_install_dir)
        build_llvm(llvm_source_dir, llvm_build_dir, llvm_install_dir, clang_tarball is None, cmake_exe, thread_count, is_cray)

def install_hdf(hdf_dir, hdf_install_dir, thread_count, cache, is_cray, insecure):
    try:
        os.mkdir(hdf_dir)
    except OSError:
        pass # Hope this means it already exists
    assert(os.path.isdir(hdf_dir))
    hdf_tarball = os.path.join(hdf_dir, 'hdf5-1_10_1.tar.gz')
    hdf_source_dir = os.path.join(hdf_dir, 'hdf5-hdf5-1_10_1')
    download(hdf_tarball, 'https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5-1_10_1.tar.gz', '1658e734d209b653c84461cd198366fdb741384c31e49490612f26028c3d3258', insecure=insecure)
    if not cache:
        extract(hdf_dir, hdf_tarball, 'gz')
        build_hdf(hdf_source_dir, hdf_install_dir, thread_count, is_cray)

def print_advice(component_dir):
    print('Given the number of things that could potentially have gone')
    print('wrong, this script is not designed to handle this situation.')
    print('You will need to fix the problem on your own (or ask for help')
    print('fixing it).')
    print()
    print('The files are located here:')
    print()
    print(component_dir)
    print()
    print('Once you have fixed the problem, you have two options:')
    print()
    print(' 1. Go to the directory and rebuild it yourself. This')
    print('    script will not touch the directory again.')
    print()
    print(' 2. Remove the directory. Then rerun this script.')
    print()
    print('Good luck and please ask for help if you get stuck!')

def report_build_failure(name, component_dir, exception):
    print()
    print('#' * 68)
    print('## Build Failed')
    print('#' * 68)
    print()
    print('It appears that %s has failed to build. The failure was:' % name)
    print()
    traceback.print_exc()
    print()
    print_advice(component_dir)
    sys.exit(1)

def check_dirty_build(name, build_result, component_dir):
    if not os.path.exists(build_result):
        print()
        print('#' * 68)
        print('## Dirty Previous Build Detected')
        print('#' * 68)
        print()
        print('It appears that %s was not built successfully on a' % name)
        print('previous invocation of this script.')
        print()
        print_advice(component_dir)
        sys.exit(1)

def driver(prefix_dir=None, scratch_dir=None, cache=False,
           legion_use_cmake=False, extra_flags=[], llvm_version=None,
           terra_url=None, terra_branch=None, terra_lua=None, terra_use_cmake=None,
           gasnet_version=None, gasnet_config_version=None,
           thread_count=None, insecure=False):
    if not cache:
        if 'CC' not in os.environ:
            raise Exception('Please set CC in your environment')
        if 'CXX' not in os.environ:
            raise Exception('Please set CXX in your environment')
        if 'LG_RT_DIR' in os.environ:
            raise Exception('Please unset LG_RT_DIR in your environment')

    is_cray = 'CRAYPE_VERSION' in os.environ

    if not cache and is_cray:
        print('This system has been detected as a Cray system.')
        print()
        print('Note: The Cray wrappers are broken for various purposes')
        print('(particularly, dynamically linked libraries). For this')
        print('reason this script requires that HOST_CC and HOST_CXX')
        print('be set to the underlying compilers (GCC and G++, etc.).')
        print()
        if 'HOST_CC' not in os.environ:
            raise Exception('Please set HOST_CC in your environment')
        if 'HOST_CXX' not in os.environ:
            raise Exception('Please set HOST_CXX in your environment')

    if terra_use_cmake is None:
        terra_use_cmake = True

    root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    legion_dir = os.path.dirname(root_dir)

    if prefix_dir is None:
        prefix_dir = root_dir
    else:
        prefix_dir = os.path.abspath(prefix_dir)

    if thread_count is None:
        try:
            # this correctly considers the current affinity mask
            thread_count = len(os.sched_getaffinity(0))
        except AttributeError:
            # this works on macos
            thread_count = multiprocessing.cpu_count()

    gasnet_release_dir = None
    conduit = None
    if gasnet_enabled():
        gasnet_dir = os.path.realpath(os.path.join(prefix_dir, 'gasnet'))
        if not os.path.exists(gasnet_dir):
            git_clone(
                gasnet_dir,
                'https://github.com/StanfordLegion/gasnet.git',
                commit_id=gasnet_config_version)
        if not cache:
            conduit = discover_conduit()
            conduit_short = short_conduit(conduit)
            gasnet_release_dir = os.path.join(gasnet_dir, 'release')
            gasnet_build_result = os.path.join(
                gasnet_release_dir, '%s-conduit' % conduit_short,
                'libgasnet-%s-par.a' % conduit_short)
            if not os.path.exists(gasnet_release_dir):
                try:
                    build_gasnet(gasnet_dir, conduit, gasnet_version)
                except Exception as e:
                    report_build_failure('gasnet', gasnet_dir, e)
            else:
                check_dirty_build('gasnet', gasnet_build_result, gasnet_dir)
            assert os.path.exists(gasnet_build_result)

    cmake_exe = None
    try:
        cmake_version = subprocess.check_output(['cmake', '--version']).decode('utf-8')
    except:
        pass # Can't find CMake, continue to download
    else:
        m = re.match(r'cmake version (\d+)[.](\d+)', cmake_version)
        # LLVM requires at least CMake 13: https://github.com/llvm/llvm-project/blob/main/llvm/CMakeLists.txt
        if m is not None and (int(m.group(1)) < 3 or int(m.group(2)) < 13):
            pass # CMake is too old, continue to download
        elif m is None:
            raise Exception('Cannot parse CMake version:\n\n%s' % cmake_version)
        else:
            cmake_exe = 'cmake' # CMake is ok, use it
    if cache or (cmake_exe is None):
        cmake_system = platform.system()
        if cmake_system == 'Darwin':
            cmake_system = 'macos'
        else:
            cmake_system = cmake_system.lower()
        cmake_processor = platform.machine()

        if cmake_system == 'macos':
            cmake_processor = 'universal'

        cmake_stem = 'cmake-3.23.4-%s-%s' % (cmake_system, cmake_processor)
        cmake_basename = '%s.tar.gz' % cmake_stem
        cmake_url = 'https://github.com/Kitware/CMake/releases/download/v3.23.4/%s' % cmake_basename
        if cmake_stem == 'cmake-3.23.4-linux-x86_64':
            cmake_shasum = '3fbcbff85043d63a8a83c8bdf8bd5b1b2fd5768f922de7dc4443de7805a2670d'
        elif cmake_stem == 'cmake-3.23.4-macos-universal':
            cmake_shasum = '98cac043cdf321caa4fd07f27da3316db6c8bc48c39997bf78e27e5c46c4eb68'
        else:
            raise Exception("Don't know how to download CMake binary for %s %s" % (cmake_system, cmake_processor))

        cmake_dir = os.path.realpath(os.path.join(prefix_dir, 'cmake'))
        cmake_install_dir = os.path.join(cmake_dir, cmake_stem)
        if not os.path.exists(cmake_dir):
            os.mkdir(cmake_dir)
            cmake_tarball = os.path.join(cmake_dir, cmake_basename)
            download(cmake_tarball, cmake_url, cmake_shasum, insecure=insecure)
            extract(cmake_dir, cmake_tarball, 'gz')
        assert os.path.exists(cmake_install_dir)
        cmake_exe = os.path.join(cmake_install_dir, 'bin', 'cmake')
        if cmake_system == 'macos':
            cmake_exe = os.path.join(cmake_install_dir, 'CMake.app', 'Contents', 'bin', 'cmake')

    llvm_dir = os.path.realpath(os.path.join(prefix_dir, 'llvm'))
    llvm_install_dir = os.path.join(llvm_dir, 'install')
    llvm_build_result = os.path.join(llvm_install_dir, 'bin', 'llvm-config')
    if not os.path.exists(llvm_install_dir):
        try:
            install_llvm(llvm_dir, llvm_install_dir, scratch_dir, llvm_version, cmake_exe, thread_count, cache, is_cray, insecure)
        except Exception as e:
            report_build_failure('llvm', llvm_dir, e)
    else:
        check_dirty_build('llvm', llvm_build_result, llvm_dir)
    if not cache:
        assert os.path.exists(llvm_build_result)

    terra_dir = os.path.join(prefix_dir, 'terra.build')
    terra_build_result = os.path.join(terra_dir, 'release', 'bin', 'terra')
    if not os.path.exists(terra_dir):
        git_clone(terra_dir, terra_url, terra_branch)
    if not os.path.exists(terra_build_result):
        try:
            build_terra(terra_dir, terra_branch, terra_lua, terra_use_cmake, cmake_exe, llvm_install_dir, cache, is_cray, thread_count)
        except Exception as e:
            report_build_failure('terra', terra_dir, e)
    else:
        check_dirty_build('terra', terra_build_result, terra_dir)
    if not cache:
        assert os.path.exists(terra_build_result)

    hdf_install_dir = None
    if hdf_enabled():
        hdf_dir = os.path.join(prefix_dir, 'hdf')
        hdf_install_dir = os.path.join(hdf_dir, 'install')
        hdf_build_result = os.path.join(hdf_install_dir, 'lib', 'libhdf5.a')
        if not os.path.exists(hdf_install_dir):
            try:
                install_hdf(hdf_dir, hdf_install_dir, thread_count, cache, is_cray, insecure)
            except Exception as e:
                report_build_failure('hdf', hdf_dir, e)
        else:
            check_dirty_build('hdf', hdf_build_result, hdf_dir)
        if not cache:
            assert os.path.exists(hdf_build_result)

    if not cache:
        build_regent(root_dir, legion_use_cmake, cmake_exe, extra_flags,
                     gasnet_release_dir, llvm_install_dir, terra_dir, hdf_install_dir,
                     conduit, thread_count)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Setup tool for Regent.')
    parser.add_argument(
        '--prefix', dest='prefix_dir', required=False,
        help='Directory in which to install dependencies.')
    parser.add_argument(
        '--scratch', dest='scratch_dir', required=False,
        help='Directory in which to store temporary build files.')
    parser.add_argument(
        '--cache-only', dest='cache', action='store_true',
        help='Only cache downloads (do not install).')
    parser.add_argument(
        '--skip-certificate-check', dest='insecure', action='store_true',
        default=discover_skip_certificate_check(),
        help='Skip certificate checks on downloads.')
    parser.add_argument(
        '--cmake', dest='legion_use_cmake', action='store_true',
        default=os.environ.get('USE_CMAKE') == '1',
        help='Use CMake to build Legion.')
    parser.add_argument(
        '--extra', dest='extra_flags', action='append', required=False,
        default=[],
        help='Extra flags for Make/CMake command.')
    parser.add_argument(
        '--llvm-version', dest='llvm_version', required=False, choices=('110', '130'),
        default=discover_llvm_version(),
        help='Select LLVM version.')
    parser.add_argument(
        '--terra-url', dest='terra_url', required=False,
        default='https://github.com/terralang/terra.git',
        help='URL of Terra repository to clone from.')
    parser.add_argument(
        '--terra-branch', dest='terra_branch', required=False,
        default='release-1.0.6',
        help='Branch of Terra repository to checkout.')
    parser.add_argument(
        '--terra-lua', dest='terra_lua', required=False,
        default=None,
        help='Lua implementation to use for Terra (luajit or moonjit).')
    parser.add_argument(
        '--terra-cmake', dest='terra_use_cmake', action='store_true', default=None,
        help='Use CMake to build Terra.')
    parser.add_argument(
        '--no-terra-cmake', dest='terra_use_cmake', action='store_false', default=None,
        help='Use CMake to build Terra.')
    parser.add_argument(
        '--gasnet-version', dest='gasnet_version', required=False,
        default=os.environ.get('GASNET_VERSION', 'GASNet-2022.9.0'),
        help='Select GASNet version.')
    parser.add_argument(
        '--gasnet-config-version', dest='gasnet_config_version', required=False,
        default='e0b5ff403b289689285b0396e191340fe4a50d0c',
        help='Select version of the GASNet configuration/build tool.')
    parser.add_argument(
        '-j', dest='thread_count', nargs='?', type=int,
        help='Number threads used to compile.')
    args = parser.parse_args()
    driver(**vars(args))
