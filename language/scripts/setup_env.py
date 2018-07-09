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
import argparse, hashlib, multiprocessing, os, platform, re, subprocess, sys, traceback

def discover_llvm_version():
    if platform.node().startswith('titan'):
        return '38'
    elif os.environ.get('LMOD_SYSTEM_NAME') == 'summit': # Summit doesn't set hostname
        return '38'
    else:
        return '39'

def discover_skip_certificate_check():
    if platform.node().startswith('titan'):
        return True
    else:
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
    elif platform.node().startswith('titan'):
        return 'gemini'
    elif os.environ.get('LMOD_SYSTEM_NAME') == 'summit': # Summit doesn't set hostname
        return 'ibv'
    else:
        raise Exception('Please set CONDUIT in your environment')

def gasnet_enabled():
    return 'USE_GASNET' not in os.environ or os.environ['USE_GASNET'] == '1'

def hdf_enabled():
    return 'USE_HDF' in os.environ and os.environ['USE_HDF'] == '1'

def check_sha1(file_path, sha1):
    with open(file_path, 'rb') as f:
        assert hashlib.sha1(f.read()).hexdigest() == sha1

def download(dest_path, url, sha1, insecure=False):
    dest_dir = os.path.dirname(dest_path)
    dest_file = os.path.basename(dest_path)
    insecure_flag = []
    if insecure:
        insecure_flag = ['--no-check-certificate']

    if os.path.exists(dest_path):
        check_sha1(dest_path, sha1)
        return

    subprocess.check_call(['wget'] + insecure_flag + ['-O', dest_path, url])
    check_sha1(dest_path, sha1)

def extract(dest_dir, archive_path, format):
    if format == 'gz':
        subprocess.check_call(['tar', 'xfz', archive_path], cwd=dest_dir)
    elif format == 'xz':
        subprocess.check_call(['tar', 'xfJ', archive_path], cwd=dest_dir)
    else:
        raise Exception('Unknown format %s' % format)

def apply_patch(dest_dir, diff_path, strip_levels=1):
    subprocess.check_call(['patch', '-p%d' % strip_levels, '-i', diff_path], cwd=dest_dir)

def git_clone(repo_dir, url, branch=None):
    if branch is not None:
        subprocess.check_call(['git', 'clone', '-b', branch, url, repo_dir])
    else:
        subprocess.check_call(['git', 'clone', url, repo_dir])

def git_update(repo_dir):
    subprocess.check_call(
        ['git', 'pull', '--ff-only'],
        cwd=repo_dir)

def build_gasnet(gasnet_dir, conduit):
    subprocess.check_call(['make', 'CONDUIT=%s' % conduit], cwd=gasnet_dir)

def build_llvm(source_dir, build_dir, install_dir, use_cmake, cmake_exe, thread_count, is_cray):
    env = None
    if is_cray:
        env = dict(list(os.environ.items()) + [
            ('CC', os.environ['HOST_CC']),
            ('CXX', os.environ['HOST_CXX']),
        ])
    if use_cmake:
        subprocess.check_call(
            [cmake_exe,
             '-DCMAKE_INSTALL_PREFIX=%s' % install_dir,
             '-DCMAKE_BUILD_TYPE=Release',
             '-DLLVM_ENABLE_ZLIB=OFF',
             '-DLLVM_ENABLE_TERMINFO=OFF',
             source_dir],
            cwd=build_dir,
            env=env)
    else:
        subprocess.check_call(
            [os.path.join(source_dir, 'configure'),
             '--prefix=%s' % install_dir,
             '--enable-optimized',
             '--disable-assertions',
             '--disable-zlib',
             '--disable-terminfo'],
            cwd=build_dir,
            env=env)
    subprocess.check_call(['make', '-j', str(thread_count)], cwd=build_dir)
    subprocess.check_call(['make', 'install'], cwd=build_dir)

def build_terra(terra_dir, llvm_dir, cache, is_cray, thread_count):
    if cache:
        subprocess.check_call(['make', 'download'], cwd=terra_dir)
        return

    env = None
    if is_cray:
        env = dict(list(os.environ.items()) + [
            ('CC', os.environ['HOST_CC']),
            ('CXX', os.environ['HOST_CXX']),
        ])

    subprocess.check_call(
        ['make',
         'LLVM_CONFIG=%s' % os.path.join(llvm_dir, 'bin', 'llvm-config'),
         'CLANG=%s' % os.path.join(llvm_dir, 'bin', 'clang'),
         'REEXPORT_LLVM_COMPONENTS=irreader mcjit x86',
         '-j', str(thread_count),
        ],
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

def build_regent(root_dir, use_cmake, cmake_exe,
                 gasnet_dir, llvm_dir, terra_dir, hdf_dir, conduit, thread_count):
    env = dict(list(os.environ.items()) +
        ([('CONDUIT', conduit),
          ('GASNET', gasnet_dir),
          ('USE_GASNET', '1')]
         if gasnet_enabled() else []) +
        ([('HDF_ROOT', hdf_dir),
          ('USE_HDF', '1')]
         if hdf_enabled() else []) +
        [('LLVM_CONFIG', os.path.join(llvm_dir, 'bin', 'llvm-config'))]
    )

    subprocess.check_call(
        [os.path.join(root_dir, 'install.py'),
         '--with-terra', terra_dir,
         '--rdir', 'auto',
         '-j', str(thread_count),
        ] + (['--cmake', '--with-cmake', cmake_exe]
             if use_cmake else ['--no-cmake']),
        env=env)

def install_llvm(llvm_dir, llvm_install_dir, llvm_version, llvm_use_cmake, cmake_exe, thread_count, cache, is_cray, insecure):
    try:
        os.mkdir(llvm_dir)
    except OSError:
        pass # Hope this means it already exists
    assert(os.path.isdir(llvm_dir))

    if llvm_version == '35':
        llvm_tarball = os.path.join(llvm_dir, 'llvm-3.5.2.src.tar.xz')
        llvm_source_dir = os.path.join(llvm_dir, 'llvm-3.5.2.src')
        clang_tarball = os.path.join(llvm_dir, 'cfe-3.5.2.src.tar.xz')
        clang_source_dir = os.path.join(llvm_dir, 'cfe-3.5.2.src')
        download(llvm_tarball, 'http://sapling.stanford.edu/~eslaught/llvm/3.5.2/llvm-3.5.2.src.tar.xz', '85faf7cbd518dabeafc4d3f7e909338fc1dab3c4', insecure=insecure)
        download(clang_tarball, 'http://sapling.stanford.edu/~eslaught/llvm/3.5.2/cfe-3.5.2.src.tar.xz', '50291e4c4ced8fcee3cca40bff0afb19fcc356e2', insecure=insecure)
    elif llvm_version == '38':
        llvm_tarball = os.path.join(llvm_dir, 'llvm-3.8.1.src.tar.xz')
        llvm_source_dir = os.path.join(llvm_dir, 'llvm-3.8.1.src')
        clang_tarball = os.path.join(llvm_dir, 'cfe-3.8.1.src.tar.xz')
        clang_source_dir = os.path.join(llvm_dir, 'cfe-3.8.1.src')
        download(llvm_tarball, 'http://sapling.stanford.edu/~eslaught/llvm/3.8.1/llvm-3.8.1.src.tar.xz', 'e0c48c4c182424b99999367d688cd8ce7876827b', insecure=insecure)
        download(clang_tarball, 'http://sapling.stanford.edu/~eslaught/llvm/3.8.1/cfe-3.8.1.src.tar.xz', 'b5ff24dc6ad8f84654f4859389990bace1cfb6d5', insecure=insecure)
    elif llvm_version == '39':
        llvm_tarball = os.path.join(llvm_dir, 'llvm-3.9.1.src.tar.xz')
        llvm_source_dir = os.path.join(llvm_dir, 'llvm-3.9.1.src')
        clang_tarball = os.path.join(llvm_dir, 'cfe-3.9.1.src.tar.xz')
        clang_source_dir = os.path.join(llvm_dir, 'cfe-3.9.1.src')
        download(llvm_tarball, 'http://sapling.stanford.edu/~eslaught/llvm/3.9.1/llvm-3.9.1.src.tar.xz', 'ce801cf456b8dacd565ce8df8288b4d90e7317ff', insecure=insecure)
        download(clang_tarball, 'http://sapling.stanford.edu/~eslaught/llvm/3.9.1/cfe-3.9.1.src.tar.xz', '95e4be54b70f32cf98a8de36821ea5495b84add8', insecure=insecure)
    else:
        assert False

    if not cache:
        extract(llvm_dir, llvm_tarball, 'xz')
        extract(llvm_dir, clang_tarball, 'xz')
        if llvm_version == '35':
            apply_patch(llvm_source_dir, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'llvm-3.5-gcc.patch'))
        os.rename(clang_source_dir, os.path.join(llvm_source_dir, 'tools', 'clang'))

        llvm_build_dir = os.path.join(llvm_dir, 'build')
        os.mkdir(llvm_build_dir)
        os.mkdir(llvm_install_dir)
        build_llvm(llvm_source_dir, llvm_build_dir, llvm_install_dir, llvm_use_cmake, cmake_exe, thread_count, is_cray)

def install_hdf(hdf_dir, hdf_install_dir, thread_count, cache, is_cray, insecure):
    try:
        os.mkdir(hdf_dir)
    except OSError:
        pass # Hope this means it already exists
    assert(os.path.isdir(hdf_dir))
    hdf_tarball = os.path.join(hdf_dir, 'hdf5-1.10.1.tar.gz')
    hdf_source_dir = os.path.join(hdf_dir, 'hdf5-1.10.1')
    download(hdf_tarball, 'http://sapling.stanford.edu/~manolis/hdf/hdf5-1.10.1.tar.gz', '73b77a23ca099ac47d8241f633bf67430007c430', insecure=insecure)
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

def driver(prefix_dir=None, cache=False, legion_use_cmake=False, llvm_version=None,
           terra_url=None, terra_branch=None, insecure=False):
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

    if llvm_version == '35':
        llvm_use_cmake = False
    elif llvm_version == '38':
        llvm_use_cmake = False
    elif llvm_version == '39':
        llvm_use_cmake = True
    else:
        raise Exception('Unrecognized LLVM version %s' % llvm_version)

    root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    legion_dir = os.path.dirname(root_dir)

    if prefix_dir is None:
        prefix_dir = root_dir
    else:
        prefix_dir = os.path.abspath(prefix_dir)

    thread_count = multiprocessing.cpu_count()

    gasnet_release_dir = None
    conduit = None
    if gasnet_enabled():
        gasnet_dir = os.path.realpath(os.path.join(prefix_dir, 'gasnet'))
        if not os.path.exists(gasnet_dir):
            git_clone(gasnet_dir, 'https://github.com/StanfordLegion/gasnet.git')
        if not cache:
            conduit = discover_conduit()
            gasnet_release_dir = os.path.join(gasnet_dir, 'release')
            gasnet_build_result = os.path.join(
                gasnet_release_dir, '%s-conduit' % conduit,
                'libgasnet-%s-par.a' % conduit)
            if not os.path.exists(gasnet_release_dir):
                try:
                    build_gasnet(gasnet_dir, conduit)
                except Exception as e:
                    report_build_failure('gasnet', gasnet_dir, e)
            else:
                check_dirty_build('gasnet', gasnet_build_result, gasnet_dir)
            assert os.path.exists(gasnet_build_result)

    cmake_exe = None
    try:
        cmake_version = subprocess.check_output(['cmake', '--version'])
    except:
        pass # Can't find CMake, continue to download
    else:
        m = re.match(r'cmake version (\d+)[.](\d+)', cmake_version)
        if m is not None and (int(m.group(1)) < 3 or int(m.group(2)) < 1):
            pass # CMake is too old, continue to download
        elif m is None:
            raise Exception('Cannot parse CMake version:\n\n%s' % cmake_version)
        else:
            cmake_exe = 'cmake' # CMake is ok, use it
    if cache or ((legion_use_cmake or llvm_use_cmake) and cmake_exe is None):
        cmake_dir = os.path.realpath(os.path.join(prefix_dir, 'cmake'))
        cmake_install_dir = os.path.join(cmake_dir, 'cmake-3.7.2-Linux-x86_64')
        if not os.path.exists(cmake_dir):
            os.mkdir(cmake_dir)

            proc_type = subprocess.check_output(['uname', '-p']).strip()
            if proc_type != 'x86_64':
                raise Exception("Don't know how to download CMake binary for %s" % proc_type)

            cmake_tarball = os.path.join(cmake_dir, 'cmake-3.7.2-Linux-x86_64.tar.gz')
            download(cmake_tarball, 'https://cmake.org/files/v3.7/cmake-3.7.2-Linux-x86_64.tar.gz', '915bc981aab354821fb9fd28374a720fdb3aa180', insecure=insecure)
            extract(cmake_dir, cmake_tarball, 'gz')
        assert os.path.exists(cmake_install_dir)
        cmake_exe = os.path.join(cmake_install_dir, 'bin', 'cmake')

    llvm_dir = os.path.realpath(os.path.join(prefix_dir, 'llvm'))
    llvm_install_dir = os.path.join(llvm_dir, 'install')
    llvm_build_result = os.path.join(llvm_install_dir, 'bin', 'llvm-config')
    if not os.path.exists(llvm_install_dir):
        try:
            install_llvm(llvm_dir, llvm_install_dir, llvm_version, llvm_use_cmake, cmake_exe, thread_count, cache, is_cray, insecure)
        except Exception as e:
            report_build_failure('llvm', llvm_dir, e)
    else:
        check_dirty_build('llvm', llvm_build_result, llvm_dir)
    if not cache:
        assert os.path.exists(llvm_build_result)

    terra_dir = os.path.join(prefix_dir, 'terra.build')
    terra_build_dir = os.path.join(terra_dir, 'build', 'bin')
    terra_build_result = os.path.join(terra_dir, 'release', 'bin', 'terra')
    if not os.path.exists(terra_dir):
        git_clone(terra_dir, terra_url, terra_branch)
    if not os.path.exists(terra_build_dir):
        try:
            build_terra(terra_dir, llvm_install_dir, cache, is_cray, thread_count)
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
        hdf_build_result = os.path.join(hdf_install_dir, 'lib', 'libhdf5.so')
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
        build_regent(root_dir, legion_use_cmake, cmake_exe,
                     gasnet_release_dir, llvm_install_dir, terra_dir, hdf_install_dir,
                     conduit, thread_count)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Setup tool for Regent.')
    parser.add_argument(
        '--prefix', dest='prefix_dir', required=False,
        help='Directory in which to install dependencies.')
    parser.add_argument(
        '--cache-only', dest='cache', action='store_true',
        help='Only cache downloads (do not install).')
    parser.add_argument(
        '--skip-certificate-check', dest='insecure', action='store_true',
        default=discover_skip_certificate_check(),
        help='Skip certificate checks on downloads.')
    parser.add_argument(
        '--cmake', dest='legion_use_cmake', action='store_true',
        default=os.environ.get('USE_CMAKE') == 1,
        help='Use CMake to build Legion.')
    parser.add_argument(
        '--llvm-version', dest='llvm_version', required=False, choices=('35', '38', '39'),
        default=discover_llvm_version(),
        help='Select LLVM version.')
    parser.add_argument(
        '--terra-url', dest='terra_url', required=False,
        default='https://github.com/elliottslaughter/terra.git',
        help='URL of Terra repository to clone from.')
    parser.add_argument(
        '--terra-branch', dest='terra_branch', required=False,
        default='compiler-snapshot',
        help='Branch of Terra repository to checkout.')
    args = parser.parse_args()
    driver(**vars(args))
