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
import argparse, multiprocessing, os, shutil, subprocess, sys, traceback, tempfile

tutorial = [
    ['tutorial/00_hello_world/hello_world', []],
    ['tutorial/01_tasks_and_futures/tasks_and_futures', []],
    ['tutorial/02_index_tasks/index_tasks', []],
    ['tutorial/03_global_vars/global_vars', []],
    ['tutorial/04_logical_regions/logical_regions', []],
    ['tutorial/05_physical_regions/physical_regions', []],
    ['tutorial/06_privileges/privileges', []],
    ['tutorial/07_partitioning/partitioning', []],
    ['tutorial/08_multiple_partitions/multiple_partitions', []],
    ['tutorial/09_custom_mapper/custom_mapper', []],
]

examples = [
    #['examples/attach_file/attach_file', []],
    ['examples/circuit/circuit', []],
    ['examples/dynamic_registration/dynamic_registration', []],
    ['examples/ghost/ghost', ['-ll:cpu', '4']],
    ['examples/ghost_pull/ghost_pull', ['-ll:cpu', '4']],
    ['examples/realm_saxpy/realm_saxpy', []],
    ['examples/spmd_cgsolver/spmd_cgsolver', ['-ll:cpu', '4', '-perproc']],
]

def cmd(command, env=None, cwd=None):
    print(' '.join(command))
    return subprocess.check_call(command, env=env, cwd=cwd)

def run_test_regent(launcher, root_dir, tmp_dir, bin_dir, env, thread_count):
    cmd([os.path.join(root_dir, 'language/travis.py')], env=env)

def run_cxx(tests, flags, launcher, root_dir, bin_dir, env, thread_count):
    for test_file, test_flags in tests:
        test_dir = os.path.dirname(os.path.join(root_dir, test_file))
        if bin_dir:
            test_path = os.path.join(bin_dir, os.path.basename(test_file))
        else:
            test_path = os.path.join(root_dir, test_file)
            cmd(['make', '-C', test_dir, '-j', str(thread_count)], env=env)
        cmd(launcher + [test_path] + flags + test_flags, env=env, cwd=test_dir)

def run_test_tutorial(launcher, root_dir, tmp_dir, bin_dir, env, thread_count):
    flags = ['-logfile', 'out_%.log']
    run_cxx(tutorial, flags, launcher, root_dir, bin_dir, env, thread_count)

def run_test_examples(launcher, root_dir, tmp_dir, bin_dir, env, thread_count):
    flags = ['-logfile', 'out_%.log']
    run_cxx(examples, flags, launcher, root_dir, bin_dir, env, thread_count)

def run_test_fuzzer(launcher, root_dir, tmp_dir, bin_dir, env, thread_count):
    env = dict(list(env.items()) + [('WARN_AS_ERROR', '0')])
    fuzz_dir = os.path.join(tmp_dir, 'fuzz-tester')
    cmd(['git', 'clone', 'https://github.com/StanfordLegion/fuzz-tester', fuzz_dir])
    cmd(['python', 'main.py'], env=env, cwd=fuzz_dir)

def run_test_realm(launcher, root_dir, tmp_dir, bin_dir, env, thread_count):
    test_dir = os.path.join(root_dir, 'test/realm')
    cmd(['make', '-C', test_dir, 'DEBUG=0', 'SHARED_LOWLEVEL=0', 'USE_CUDA=0', 'USE_GASNET=0', 'clean'], env=env)
    cmd(['make', '-C', test_dir, 'DEBUG=0', 'SHARED_LOWLEVEL=0', 'USE_CUDA=0', 'USE_GASNET=0', 'run_all'], env=env)

    perf_dir = os.path.join(root_dir, 'test/performance/realm')
    cmd(['make', '-C', perf_dir, 'DEBUG=0', 'SHARED_LOWLEVEL=0', 'clean_all'], env=env)
    cmd(['make', '-C', perf_dir, 'DEBUG=0', 'SHARED_LOWLEVEL=0', 'run_all'], env=env)

def run_test_external(launcher, root_dir, tmp_dir, bin_dir, env, thread_count):
    flags = ['-logfile', 'out_%.log']

    # Fast Direct Solver
    # Contact: Chao Chen <cchen10@stanford.edu>
    solver_dir = os.path.join(tmp_dir, 'fastSolver2')
    cmd(['git', 'clone', 'https://github.com/Charles-Chao-Chen/fastSolver2.git', solver_dir])
    solver = [[os.path.join(solver_dir, 'spmd_benchMark/solver'),
               ['-machine', '1', '-core', '8', '-mtxlvl', '6', '-ll:cpu', '8']]]
    run_cxx(solver, flags, launcher, root_dir, None, env, thread_count)

    # Parallel Research Kernels: Stencil
    # Contact: Wonchan Lee <wonchan@cs.stanford.edu>
    prk_dir = os.path.join(tmp_dir, 'prk')
    cmd(['git', 'clone', 'https://github.com/magnatelee/PRK.git', prk_dir])
    stencil_dir = os.path.join(prk_dir, 'LEGION', 'Stencil')
    stencil_env = dict(list(env.items()) + [
        ('OUTFILE', 'stencil'),
        ('GEN_SRC', 'stencil.cc'),
        ('CC_FLAGS', (env['CC_FLAGS'] if 'CC_FLAGS' in env else '') +
         ' -DRADIUS=2 -DRESTRICT_KEYWORD -DDISABLE_BARRIER_MIGRATION'),
    ])
    makefile = os.path.join(root_dir, 'apps/Makefile.template')
    cmd(['make', '-f', makefile, '-C', stencil_dir, '-j', str(thread_count)], env=stencil_env)
    stencil = os.path.join(stencil_dir, 'stencil')
    cmd([stencil, '4', '10', '1000'])

def run_test_private(launcher, root_dir, tmp_dir, bin_dir, env, thread_count):
    flags = ['-logfile', 'out_%.log']

    # MiniAero
    # Contact: Wonchan Lee <wonchan@cs.stanford.edu>
    miniaero_dir = os.path.join(tmp_dir, 'miniaero-spmd')
    cmd(['git', 'clone', '-b', 'spmd_flattened_superblocks',
         'git@github.com:magnatelee/miniaero-spmd.git', miniaero_dir])
    cmd(['make', '-C', miniaero_dir, '-j', str(thread_count)], env=env,
        cwd=miniaero_dir)
    for test in ['3D_Sod', '3D_Sod_2nd_Order'
                 # These tests take a long time so skip them by default.
                 # , 'FlatPlate', 'Ramp'
                ]:
        test_dir = os.path.join(miniaero_dir, 'tests', test)
        cmd([os.path.join(test_dir, 'test.sh')], env=env, cwd=test_dir)

def build_cmake(root_dir, tmp_dir, env, thread_count,
                test_tutorial, test_examples):
    build_dir = os.path.join(tmp_dir, 'build')
    install_dir = os.path.join(tmp_dir, 'install')
    os.mkdir(build_dir)
    os.mkdir(install_dir)
    cmd(['cmake', '-DCMAKE_INSTALL_PREFIX=%s' % install_dir] +
        (['-DLegion_BUILD_EXAMPLES=ON'] if test_tutorial or test_examples else []) +
        [root_dir],
        env=env, cwd=build_dir)
    cmd(['make', '-C', build_dir, '-j', str(thread_count)], env=env)
    cmd(['make', '-C', build_dir, 'install'], env=env)
    return os.path.join(build_dir, 'bin')

def clean_cxx(tests, root_dir, env, thread_count):
    for test_file, test_flags in tests:
        test_dir = os.path.dirname(os.path.join(root_dir, test_file))
        cmd(['make', '-C', test_dir, 'clean'], env=env)

def build_make_clean(root_dir, env, thread_count, test_tutorial, test_examples):
    if test_tutorial:
        clean_cxx(tutorial, root_dir, env, thread_count)
    if test_examples:
        clean_cxx(examples, root_dir, env, thread_count)

def option_enabled(option, options, var_prefix='', default=True):
    if options is not None: return option in options
    option_var = '%s%s' % (var_prefix, option.upper())
    if option_var in os.environ: return os.environ[option_var] == '1'
    return default

def run_tests(test_modules=None,
              debug=True,
              use_features=None,
              launcher=None,
              thread_count=None,
              root_dir=None,
              keep_tmp_dir=False,
              verbose=False):
    if thread_count is None:
        thread_count = multiprocessing.cpu_count()

    if root_dir is None:
        root_dir = os.path.dirname(os.path.realpath(__file__))

    # Determine which test modules to run.
    def module_enabled(module, default=True):
        return option_enabled(module, test_modules, 'TEST_', default)
    test_regent = module_enabled('regent')
    test_tutorial = module_enabled('tutorial')
    test_examples = module_enabled('examples')
    test_fuzzer = module_enabled('fuzzer', debug)
    test_realm = module_enabled('realm', not debug)
    test_external = module_enabled('external', False)
    test_private = module_enabled('private', False)

    # Determine which features to build with.
    def feature_enabled(feature, default=True):
        return option_enabled(feature, use_features, 'USE_', default)
    use_gasnet = feature_enabled('gasnet', False)
    use_cuda = feature_enabled('cuda', False)
    use_llvm = feature_enabled('llvm', False)
    use_hdf = feature_enabled('hdf', False)
    use_spy = feature_enabled('spy', False)
    use_cmake = feature_enabled('cmake', False)
    use_rdir = feature_enabled('rdir', True)

    if use_gasnet and launcher is None:
        raise Exception('GASNet is enabled but launcher is not set (use --launcher or LAUNCHER)')
    launcher = launcher.split() if launcher is not None else []

    # Normalize the test environment.
    env = dict(list(os.environ.items()) + [
        ('DEBUG', '1' if debug else '0'),
        ('LAUNCHER', ' '.join(launcher)),
        ('USE_GASNET', '1' if use_gasnet else '0'),
        ('USE_CUDA', '1' if use_cuda else '0'),
        ('USE_LLVM', '1' if use_llvm else '0'),
        ('USE_HDF', '1' if use_hdf else '0'),
        ('TEST_HDF', '1' if use_hdf else '0'),
        ('USE_SPY', '1' if use_spy else '0'),
        ('TEST_SPY', '1' if use_spy else '0'),
        ('USE_RDIR', '1' if use_rdir else '0'),
        ('LG_RT_DIR', os.path.join(root_dir, 'runtime')),
    ])

    tmp_dir = tempfile.mkdtemp(dir=root_dir)
    if verbose:
        print('Using build directory: %s' % tmp_dir)
        print()
    try:
        # Build tests.
        if use_cmake:
            bin_dir = build_cmake(root_dir, tmp_dir, env, thread_count,
                                  test_tutorial, test_examples)
        else:
            # With GNU Make, builds happen inline. But clean here.
            build_make_clean(root_dir, env, thread_count,
                             test_tutorial, test_examples)
            bin_dir = None

        # Run tests.
        if test_regent:
            run_test_regent(launcher, root_dir, tmp_dir, bin_dir, env, thread_count)
        if test_tutorial:
            run_test_tutorial(launcher, root_dir, tmp_dir, bin_dir, env, thread_count)
        if test_examples:
            run_test_examples(launcher, root_dir, tmp_dir, bin_dir, env, thread_count)
        if test_fuzzer:
            run_test_fuzzer(launcher, root_dir, tmp_dir, bin_dir, env, thread_count)
        if test_realm:
            run_test_realm(launcher, root_dir, tmp_dir, bin_dir, env, thread_count)
        if test_external:
            run_test_external(launcher, root_dir, tmp_dir, bin_dir, env, thread_count)
        if test_private:
            run_test_private(launcher, root_dir, tmp_dir, bin_dir, env, thread_count)
    finally:
        if keep_tmp_dir:
            print('Leaving build directory:')
            print('  %s' % tmp_dir)
        else:
            if verbose:
                print('Removing build directory:')
                print('  %s' % tmp_dir)
            shutil.rmtree(tmp_dir)

def driver():
    parser = argparse.ArgumentParser(
        description = 'Install Regent front end.')

    # What tests to run:
    parser.add_argument(
        '--test', dest='test_modules', action='append',
        choices=['regent', 'tutorial', 'examples', 'fuzzer', 'realm',
                 'external', 'private'],
        default=None,
        help='Test modules to run (also via TEST_*).')

    # Build options:
    parser.add_argument(
        '--debug', dest='debug', action='store_true',
        default=os.environ['DEBUG'] == '1' if 'DEBUG' in os.environ else True,
        help='Build Legion in debug mode (also via DEBUG).')
    parser.add_argument(
        '--use', dest='use_features', action='append',
        choices=['gasnet', 'cuda', 'llvm', 'hdf', 'spy', 'cmake', 'rdir'],
        default=None,
        help='Build Legion with features (also via USE_*).')
    parser.add_argument(
        '--launcher', dest='launcher', action='store',
        default=os.environ['LAUNCHER'] if 'LAUNCHER' in os.environ else None,
        help='Launcher for Legion tests (also via LAUNCHER).')

    parser.add_argument(
        '-C', '--directory', dest='root_dir', metavar='DIR', action='store', required=False,
        help='Legion root directory.')

    parser.add_argument(
        '-j', dest='thread_count', nargs='?', type=int,
        help='Number threads used to compile.')

    parser.add_argument(
        '--keep', dest='keep_tmp_dir', action='store_true',
        help='Keep temporary directory.')

    parser.add_argument(
        '-v', '--verbose', dest='verbose', action='store_true',
        help='Print more debugging information.')

    args = parser.parse_args()

    run_tests(**vars(args))

if __name__ == '__main__':
    driver()
