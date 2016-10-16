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

from __future__ import print_function
import hashlib, multiprocessing, os, platform, subprocess, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import install # from ./install.py

def check_sha1(file_path, sha1):
    with open(file_path, 'rb') as f:
        assert hashlib.sha1(f.read()).hexdigest() == sha1

def download(dest_path, url, sha1):
    dest_dir = os.path.dirname(dest_path)
    dest_file = os.path.basename(dest_path)
    subprocess.check_call(['wget', '-O', dest_path, url])
    check_sha1(dest_path, sha1)

def extract(dest_dir, archive_path, format):
    if format == 'xz':
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

def discover_conduit():
    if 'CONDUIT' in os.environ:
        return os.environ['CONDUIT']
    elif platform.node().startswith('daint'):
        return 'aries'
    elif platform.node().startswith('excalibur'):
        return 'aries'
    else:
        raise Exception('Please set CONDUIT in your environment')

def build_gasnet(gasnet_dir, conduit):
    subprocess.check_call(['make', 'CONDUIT=%s' % conduit], cwd=gasnet_dir)

def build_llvm(source_dir, build_dir, install_dir, thread_count):
    subprocess.check_call(
        [os.path.join(source_dir, 'configure'),
         '--prefix=%s' % install_dir,
         '--enable-optimized',
         '--disable-zlib', '--disable-terminfo', '--disable-libedit'],
        cwd=build_dir)
    subprocess.check_call(['make', '-j', str(thread_count)], cwd=build_dir)
    subprocess.check_call(['make', 'install'], cwd=build_dir)

def build_terra(terra_dir, llvm_dir, thread_count):
    subprocess.check_call(
        ['make',
         'CC=%s -dynamic' % os.environ['CC'],
         'CXX=%s -dynamic' % os.environ['CXX'],
         'LLVM_CONFIG=%s' % os.path.join(llvm_dir, 'bin', 'llvm-config'),
         'CLANG=%s' % os.path.join(llvm_dir, 'bin', 'clang'),
         'LLVM_VERSION=35',
         '-j', str(thread_count)],
        cwd = terra_dir)

if __name__ == '__main__':
    if 'CC' not in os.environ:
        raise Exception('Please set CC in your environment')
    if 'CXX' not in os.environ:
        raise Exception('Please set CXX in your environment')
    if 'LG_RT_DIR' in os.environ:
        raise Exception('Please unset LG_RT_DIR in your environment')

    root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    legion_dir = os.path.dirname(root_dir)

    thread_count = multiprocessing.cpu_count()

    conduit = discover_conduit()
    gasnet_dir = os.path.realpath(os.path.join(root_dir, 'gasnet'))
    gasnet_release_dir = os.path.join(gasnet_dir, 'release')
    if not os.path.exists(gasnet_dir):
        git_clone(gasnet_dir, 'https://github.com/StanfordLegion/gasnet.git')
        build_gasnet(gasnet_dir, conduit)
    assert os.path.exists(gasnet_release_dir)

    llvm_dir = os.path.realpath(os.path.join(root_dir, 'llvm'))
    llvm_install_dir = os.path.join(llvm_dir, 'install')
    if not os.path.exists(llvm_dir):
        os.mkdir(llvm_dir)

        llvm_tarball = os.path.join(llvm_dir, 'llvm-3.5.2.src.tar.xz')
        llvm_source_dir = os.path.join(llvm_dir, 'llvm-3.5.2.src')
        clang_tarball = os.path.join(llvm_dir, 'cfe-3.5.2.src.tar.xz')
        clang_source_dir = os.path.join(llvm_dir, 'cfe-3.5.2.src')
        download(llvm_tarball, 'http://llvm.org/releases/3.5.2/llvm-3.5.2.src.tar.xz', '85faf7cbd518dabeafc4d3f7e909338fc1dab3c4')
        download(clang_tarball, 'http://llvm.org/releases/3.5.2/cfe-3.5.2.src.tar.xz', '50291e4c4ced8fcee3cca40bff0afb19fcc356e2')
        extract(llvm_dir, llvm_tarball, 'xz')
        extract(llvm_dir, clang_tarball, 'xz')
        os.rename(clang_source_dir, os.path.join(llvm_source_dir, 'tools', 'clang'))

        llvm_build_dir = os.path.join(llvm_dir, 'build')
        os.mkdir(llvm_build_dir)
        os.mkdir(llvm_install_dir)
        build_llvm(llvm_source_dir, llvm_build_dir, llvm_install_dir, thread_count)
    assert os.path.exists(llvm_install_dir)

    terra_dir = os.path.join(root_dir, 'terra.build')
    if not os.path.exists(terra_dir):
        git_clone(terra_dir, 'https://github.com/elliottslaughter/terra.git', 'compiler-ppopp17-snapshot')
        build_terra(terra_dir, llvm_install_dir, thread_count)

    install.install(
        shared_llr=False, general_llr=True, gasnet=True, cuda=False,
        external_terra_dir=terra_dir, gasnet_dir=gasnet_release_dir,
        debug=False, thread_count=thread_count)
