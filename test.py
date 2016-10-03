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
import argparse, multiprocessing, os, subprocess

def cmd(command, env=None, cwd=None):
    print(' '.join(command))
    return subprocess.check_call(command, env=env, cwd=cwd)

def run_test_regent(root_dir, env, thread_count):
    cmd([os.path.join(root_dir, 'language/travis.py')], env=env)

def run_test_cxx(tests, root_dir, env, thread_count):
    for test in tests:
        test_file, test_flags = test
        test_path = os.path.join(root_dir, test_file)
        test_dir = os.path.dirname(test_path)
        cmd(['make', '-s', '-C', test_dir, '-j', thread_count], env=env)
        cmd([test_path] + test_flags, env=env, cwd=test_dir)

def run_test_tutorial(root_dir, env, thread_count):
    flags = ['-logfile', 'out_%.log']
    tutorials = [
        ['tutorial/00_hello_world/hello_world', flags],
        ['tutorial/01_tasks_and_futures/task_and_futures', flags],
        ['tutorial/02_index_tasks/index_tasks', flags],
        ['tutorial/03_global_vars/global_vars', flags],
        ['tutorial/04_logical_regions/logical_regions', flags],
        ['tutorial/05_physical_regions/phsyical_regions', flags],
        ['tutorial/06_privileges/privileges', flags],
        ['tutorial/07_partitioning/partitioning', flags],
        ['tutorial/08_multiple_partitions/multiple_partitions', flags],
        ['tutorial/09_custom_mapper/custom_mapper', flags],
    ]
    run_test_cxx(tutorials, root_dir, env, thread_count)

def run_test_examples(root_dir, env, thread_count):
    flags = ['-logfile', 'out_%.log']
    tutorials = [
        ['examples/attach_file/attach_file', flags],
        ['examples/dynamic_registration/dynamic_registration', flags],
        ['examples/full_circuit/ckt_sim', flags],
        ['examples/full_ghost/ghost', ['-ll:cpu', '4'] + flags],
        ['examples/spmd_cgsolver/cgsolver', ['-ll:cpu', '4', '-perproc'] + flags],
    ]
    run_test_cxx(tutorials, root_dir, env, thread_count)

def run_test_realm(root_dir, env, thread_count):
    test_dir = os.path.join(root_dir, 'test/realm')
    cmd(['make', '-s', '-C', test_dir, 'DEBUG=0', 'SHARED_LOWLEVEL=0', 'USE_CUDA=0', 'USE_GASNET=0', 'clean'])
    cmd(['make', '-s', '-C', test_dir, 'DEBUG=0', 'SHARED_LOWLEVEL=0', 'USE_CUDA=0', 'USE_GASNET=0', 'run_all'])

    perf_dir = os.path.join(root_dir, 'test/performance/realm')
    cmd(['make', '-s', '-C', perf_dir, 'DEBUG=0', 'SHARED_LOWLEVEL=0', 'clean'])
    cmd(['make', '-s', '-C', perf_dir, 'DEBUG=0', 'SHARED_LOWLEVEL=0', 'run_all'])

def run_test_fuzzer(root_dir, env, thread_count):
    env = dict(list(env.items()) + [('WARN_AS_ERROR', '0')])
    fuzz_dir = os.path.join(root_dir, 'fuzz-tester')
    cmd(['git', 'clone', 'https://github.com/StanfordLegion/fuzz-tester', fuzz_dir], cwd=root_dir)
    cmd(['python', 'main.py'], env=env, cwd=fuzz_dir)

def option_enabled(option, options, var_prefix='', default=True):
    if options is not None: return option in options
    option_var = '%s%s' % (var_prefix, option.upper())
    if option_var in os.environ: return os.environ[option_var] == '1'
    return default

def run_tests(test_modules=None,
              debug=True,
              use_features=None,
              thread_count=None,
              root_dir=None):
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
    test_realm = module_enabled('realm', not debug)
    test_fuzzer = module_enabled('fuzzer', debug)
    test_spy = module_enabled('spy')

    # Determine which features to build with.
    def feature_enabled(feature, default=True):
        return option_enabled(feature, use_features, 'USE_', default)
    use_gasnet = feature_enabled('gasnet', False)
    use_cuda = feature_enabled('cuda', False)
    use_llvm = feature_enabled('llvm', False)
    use_hdf = feature_enabled('hdf', False)
    use_rdir = feature_enabled('rdir', True)

    # Normalize the test environment.
    env = dict(list(os.environ.items()) + [
        ('TEST_REGENT', '1' if test_regent else '0'),
        ('TEST_TUTORIAL', '1' if test_tutorial else '0'),
        ('TEST_EXAMPLES', '1' if test_examples else '0'),
        ('TEST_REALM', '1' if test_realm else '0'),
        ('TEST_FUZZER', '1' if test_fuzzer else '0'),
        ('TEST_SPY', '1' if test_spy else '0'),
        ('DEBUG', '1' if debug else '0'),
        ('USE_GASNET', '1' if use_gasnet else '0'),
        ('USE_CUDA', '1' if use_cuda else '0'),
        ('USE_LLVM', '1' if use_llvm else '0'),
        ('USE_HDF', '1' if use_hdf else '0'),
        ('USE_RDIR', '1' if use_rdir else '0'),
        ('LG_RT_DIR', os.path.join(root_dir, 'runtime')),
    ])

    # Run tests.
    if test_regent:
        run_test_regent(root_dir, env, thread_count)

    if test_tutorial:
        run_test_tutorial(root_dir, env, thread_count)

    if test_examples:
        run_test_examples(root_dir, env, thread_count)

    if test_realm:
        run_test_realm(root_dir, env, thread_count)

    if test_fuzzer:
        run_test_fuzzer(root_dir, env, thread_count)

def driver():
    parser = argparse.ArgumentParser(
        description = 'Install Regent front end.')

    # What tests to run:
    parser.add_argument(
        '--test', dest='test_modules', action='append',
        choices=['regent', 'tutorial', 'examples', 'realm', 'fuzzer', 'spy'],
        default=None,
        help='Test modules to run (also via TEST_*).')

    # Build options:
    parser.add_argument(
        '--debug', dest='debug', action='store_true',
        default=os.environ['DEBUG'] == '1' if 'DEBUG' in os.environ else True,
        help='Build Legion in debug mode (also via DEBUG).')
    parser.add_argument(
        '--use', dest='use_features', action='append',
        choices=['gasnet', 'cuda', 'llvm', 'hdf', 'rdir'],
        default=None,
        help='Build Legion with features (also via USE_*).')

    parser.add_argument(
        '-C', '--directory', dest='root_dir', metavar='DIR', action='store', required=False,
        help='Legion root directory.')

    parser.add_argument(
        '-j', dest='thread_count', nargs='?', type=int,
        help='Number threads used to compile.')

    args = parser.parse_args()

    run_tests(**vars(args))

if __name__ == '__main__':
    driver()
