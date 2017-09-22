#!/usr/bin/env python

# Copyright 2017 Stanford University
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
import argparse, hashlib, multiprocessing, os, platform, subprocess, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import install # from ./install.py

def discover_llvm_version():
    if platform.node().startswith('titan'):
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
    else:
        raise Exception('Please set CONDUIT in your environment')

def check_sha1(file_path, sha1):
    with open(file_path, 'rb') as f:
        assert hashlib.sha1(f.read()).hexdigest() == sha1

def download(dest_path, url, sha1, insecure=False):
    dest_dir = os.path.dirname(dest_path)
    dest_file = os.path.basename(dest_path)
    insecure_flag = []
    if insecure:
        insecure_flag = ['--no-check-certificate']
    subprocess.check_call(['wget'] + insecure_flag + ['-O', dest_path, url])
    check_sha1(dest_path, sha1)

def extract(dest_dir, archive_path, format):
    if format == 'gz':
        subprocess.check_call(['tar', 'xfz', archive_path], cwd=dest_dir)
    elif format == 'xz':
        subprocess.check_call(['tar', 'xfJ', archive_path], cwd=dest_dir)
    else:
        raise Exception('Unknown format %s' % format)

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

def build_terra(terra_dir, llvm_dir, is_cray, thread_count):
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
         '-j', str(thread_count),
        ],
        cwd=terra_dir,
        env=env)

def driver(llvm_version, insecure):
    if 'CC' not in os.environ:
        raise Exception('Please set CC in your environment')
    if 'CXX' not in os.environ:
        raise Exception('Please set CXX in your environment')
    if 'LG_RT_DIR' in os.environ:
        raise Exception('Please unset LG_RT_DIR in your environment')

    is_cray = 'CRAYPE_VERSION' in os.environ

    if is_cray:
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

    if llvm_version == '38':
        llvm_use_cmake = False
    elif llvm_version == '39':
        llvm_use_cmake = True
    else:
        raise Exception('Unrecognized LLVM version %s' % llvm_version)

    root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    legion_dir = os.path.dirname(root_dir)

    thread_count = multiprocessing.cpu_count()

    conduit = discover_conduit()
    gasnet_dir = os.path.realpath(os.path.join(root_dir, 'gasnet'))
    gasnet_release_dir = os.path.join(gasnet_dir, 'release')
    if not os.path.exists(gasnet_dir):
        git_clone(gasnet_dir, 'https://github.com/StanfordLegion/gasnet.git')
    build_gasnet(gasnet_dir, conduit)

    cmake_exe = None
    if llvm_use_cmake:
        cmake_dir = os.path.realpath(os.path.join(root_dir, 'cmake'))
        cmake_install_dir = os.path.join(cmake_dir, 'cmake-3.7.2-Linux-x86_64')
        if not os.path.exists(cmake_dir):
            os.mkdir(cmake_dir)
            cmake_tarball = os.path.join(cmake_dir, 'cmake-3.7.2-Linux-x86_64.tar.gz')
            download(cmake_tarball, 'https://cmake.org/files/v3.7/cmake-3.7.2-Linux-x86_64.tar.gz', '915bc981aab354821fb9fd28374a720fdb3aa180', insecure=insecure)
            extract(cmake_dir, cmake_tarball, 'gz')
        assert os.path.exists(cmake_install_dir)
        cmake_exe = os.path.join(cmake_install_dir, 'bin', 'cmake')

    llvm_dir = os.path.realpath(os.path.join(root_dir, 'llvm'))
    llvm_build_dir = os.path.join(llvm_dir, 'build')
    llvm_install_dir = os.path.join(llvm_dir, 'install')
    if llvm_version == '38':
        llvm_source_dir = os.path.join(llvm_dir, 'llvm-3.8.1.src')
        clang_source_dir = os.path.join(llvm_dir, 'cfe-3.8.1.src')
    elif llvm_version == '39':
        llvm_source_dir = os.path.join(llvm_dir, 'llvm-3.9.1.src')
        clang_source_dir = os.path.join(llvm_dir, 'cfe-3.9.1.src')
    else:
        assert(False)
    if not os.path.exists(llvm_dir):
        os.mkdir(llvm_dir)
        if llvm_version == '38':
            llvm_tarball = os.path.join(llvm_dir, 'llvm-3.8.1.src.tar.xz')
            clang_tarball = os.path.join(llvm_dir, 'cfe-3.8.1.src.tar.xz')
            download(llvm_tarball, 'http://llvm.org/releases/3.8.1/llvm-3.8.1.src.tar.xz', 'e0c48c4c182424b99999367d688cd8ce7876827b', insecure=insecure)
            download(clang_tarball, 'http://llvm.org/releases/3.8.1/cfe-3.8.1.src.tar.xz', 'b5ff24dc6ad8f84654f4859389990bace1cfb6d5', insecure=insecure)
        elif llvm_version == '39':
            llvm_tarball = os.path.join(llvm_dir, 'llvm-3.9.1.src.tar.xz')
            clang_tarball = os.path.join(llvm_dir, 'cfe-3.9.1.src.tar.xz')
            download(llvm_tarball, 'http://llvm.org/releases/3.9.1/llvm-3.9.1.src.tar.xz', 'ce801cf456b8dacd565ce8df8288b4d90e7317ff', insecure=insecure)
            download(clang_tarball, 'http://llvm.org/releases/3.9.1/cfe-3.9.1.src.tar.xz', '95e4be54b70f32cf98a8de36821ea5495b84add8', insecure=insecure)
        else:
            assert(False)
        extract(llvm_dir, llvm_tarball, 'xz')
        extract(llvm_dir, clang_tarball, 'xz')
        os.rename(clang_source_dir, os.path.join(llvm_source_dir, 'tools', 'clang'))
        os.mkdir(llvm_build_dir)
        os.mkdir(llvm_install_dir)
    build_llvm(llvm_source_dir, llvm_build_dir, llvm_install_dir, llvm_use_cmake, cmake_exe, thread_count, is_cray)

    terra_dir = os.path.join(root_dir, 'terra.build')
    if not os.path.exists(terra_dir):
        git_clone(terra_dir, 'https://github.com/elliottslaughter/terra.git', 'compiler-sc17-snapshot')
    build_terra(terra_dir, llvm_install_dir, is_cray, thread_count)

    use_cuda = 'USE_CUDA' in os.environ and os.environ['USE_CUDA'] == '1'
    use_openmp = 'USE_OPENMP' in os.environ and os.environ['USE_OPENMP'] == '1'
    use_hdf = 'USE_HDF' in os.environ and os.environ['USE_HDF'] == '1'
    debug = 'DEBUG' in os.environ and os.environ['DEBUG'] == '1'
    install.install(
        gasnet=True, cuda=use_cuda, openmp=use_openmp, hdf=use_hdf,
        external_terra_dir=terra_dir, gasnet_dir=gasnet_release_dir, conduit=conduit,
        debug=debug, thread_count=thread_count)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Setup tool for Regent.')
    parser.add_argument(
        '--skip-certificate-check', dest='skip_certificate_check', action='store_true',
        default=discover_skip_certificate_check(),
        help='Skip certificate checks on downloads.')
    parser.add_argument(
        '--llvm-version', dest='llvm_version', required=False, choices=('38', '39'),
        default=discover_llvm_version(),
        help='Select LLVM version.')
    args = parser.parse_args()
    driver(args.llvm_version, args.skip_certificate_check)
