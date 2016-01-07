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
import argparse, multiprocessing, os, platform, subprocess, sys

# Requires:
#   * Terra-compatible LLVM installation on PATH

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

def build_terra(terra_dir, thread_count):
    subprocess.check_call(
        ['make', 'all', '-j', str(thread_count)] +
        (['LUAJIT_URL=%s' % os.environ['LUAJIT_URL']] if 'LUAJIT_URL' in os.environ else []),
        cwd = terra_dir)

def install_terra(terra_dir, external_terra_dir, thread_count):
    if external_terra_dir is not None:
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
    else:
        if os.path.islink(terra_dir):
            print('Error: Attempting build with internal Terra when external Terra')
            print('already exists. Please remove the following symlink to continue with')
            print('an internal Terra installation.')
            print('    %s' % terra_dir)
            sys.exit(1)

    if not os.path.exists(terra_dir):
        git_clone(terra_dir, 'https://github.com/zdevito/terra.git')
    else:
        git_update(terra_dir)
    build_terra(terra_dir, thread_count)

def symlink(from_path, to_path):
    if not os.path.lexists(to_path):
        os.symlink(from_path, to_path)

def install_bindings(bindings_dir, runtime_dir, terra_dir, debug, general_llr,
                     cuda, gasnet, thread_count):
    env = dict(list(os.environ.items()) + [
        ('LG_RT_DIR', runtime_dir),
        ('TERRA_DIR', terra_dir),                           # for bindings
    ])

    flags = (
        ['DEBUG=%s' % (1 if debug else 0),
         'SHARED_LOWLEVEL=%s' % (0 if general_llr else 1),
         'USE_CUDA=%s' % (1 if cuda else 0),
         'USE_GASNET=%s' % (1 if gasnet else 0),
         ] +
        (['GCC=%s' % os.environ['CXX']] if 'CXX' in os.environ else []))

    subprocess.check_call(
        ['make'] +
        flags +
        ['clean'],
        cwd = bindings_dir,
        env = env)
    subprocess.check_call(
        ['make'] +
        flags +
        ['-j', str(thread_count)],
        cwd = bindings_dir,
        env = env)
    symlink(os.path.join(bindings_dir, 'liblegion_terra.so'),
            os.path.join(bindings_dir, 'liblegion_terra%s' % dylib_ext))

    # This last bit is necessary because Mac OS X shared libraries
    # have paths hard-coded into them, and in this case those paths
    # are coming out wrong. Therefore, we need to fix them to use the
    # search path again so our scripts can find them properly.
    #
    # You can sanity check that this step actually worked with the
    # commands:
    #
    # otool -L liblegion_terra.so
    # ./regent.py
    #   =package.loadlib('liblegion_terra.so', 'init')

    if os_name == 'Darwin':
        subprocess.check_call(
            ['install_name_tool', '-change',
             '/usr/local/lib/libluajit-5.1.2.dylib', 'libluajit-5.1.2.dylib',
             os.path.join(bindings_dir, 'liblegion_terra.so')])

def install():
    parser = argparse.ArgumentParser(
        description = 'Install Regent front end.')
    parser.add_argument(
        '--with-terra', dest = 'terra', required = False,
        help = 'Path to Terra installation directory (optional).')
    parser.add_argument(
        '--debug', dest = 'debug', action = 'store_true', required = False,
        help = 'Build Legion with debugging enabled.')
    parser.add_argument(
        '--shared', dest = 'shared_llr', action = 'store_true', required = False,
        help = 'Build Legion with the shared low-level runtime.')
    parser.add_argument(
        '--general', dest = 'general_llr', action = 'store_true', required = False,
        help = 'Build Legion with the general low-level runtime.')
    parser.add_argument(
        '--gasnet', dest = 'gasnet', action = 'store_true', required = False,
        help = 'Build Legion with GASNet.')
    parser.add_argument(
        '--cuda', dest = 'cuda', action = 'store_true', required = False,
        help = 'Build Legion with CUDA.')
    parser.add_argument(
        '-j', dest = 'thread_count', nargs = '?', type = int,
        help = 'Number threads used to compile.')
    args = parser.parse_args()

    if args.shared_llr and args.general_llr:
        raise Exception('Shared LLR is mutually exclusive with general LLR.')

    general = not args.shared_llr

    if args.gasnet and not general:
        raise Exception('General LLR is required for GASNet.')

    if args.cuda and not general:
        raise Exception('General LLR is required for CUDA.')

    thread_count = args.thread_count
    if thread_count is None:
        thread_count = multiprocessing.cpu_count()

    root_dir = os.path.dirname(os.path.realpath(__file__))
    legion_dir = os.path.dirname(root_dir)

    # Grab LG_RT_DIR from the environment if available, otherwise
    # assume we're running relative to our own location.
    runtime_dir = os.path.join(legion_dir, 'runtime')
    if 'LG_RT_DIR' in os.environ:
        runtime_dir = os.path.realpath(os.environ['LG_RT_DIR'])

    terra_dir = os.path.join(root_dir, 'terra')
    install_terra(terra_dir, args.terra, thread_count)

    bindings_dir = os.path.join(legion_dir, 'bindings', 'terra')
    install_bindings(bindings_dir, runtime_dir, terra_dir, args.debug,
                     general, args.cuda, args.gasnet,
                     thread_count)

if __name__ == '__main__':
    install()
