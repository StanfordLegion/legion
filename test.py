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
import argparse, datetime, glob, json, multiprocessing, os, platform, shutil, subprocess, sys, traceback, tempfile

# Find physical core count of the machine.
if platform.system() == 'Linux':
    lines = subprocess.check_output(['lscpu', '--parse=core'])
    physical_cores = len(set(line for line in lines.strip().split('\n')
                             if not line.startswith('#')))
elif platform.system() == 'Darwin':
    physical_cores = int(
        subprocess.check_output(['sysctl', '-n', 'hw.physicalcpu']))
else:
    raise Exception('Unknown platform: %s' % platform.system())

# Choose a reasonable number of application cores given the
# available physical cores.
app_cores = max(physical_cores - 2, 1)

# for backward-compatibility, use app_cores if PERF_CORES_PER_NODE is not specified
perf_cores_per_node = os.environ.get('PERF_CORES_PER_NODE', str(app_cores))

legion_cxx_tests = [
    # Tutorial
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

    # Examples
    ['examples/circuit/circuit', []],
    ['examples/dynamic_registration/dynamic_registration', []],
    ['examples/ghost/ghost', ['-ll:cpu', '4']],
    ['examples/ghost_pull/ghost_pull', ['-ll:cpu', '4']],
    ['examples/realm_saxpy/realm_saxpy', []],
    ['examples/spmd_cgsolver/spmd_cgsolver', ['-ll:cpu', '4', '-perproc']],
    ['examples/virtual_map/virtual_map', []],

    # Tests
]

if platform.system() != 'Darwin':
    legion_cxx_tests = legion_cxx_tests + [
        # FIXME: Fails non-deterministically on Mac OS: https://github.com/StanfordLegion/legion/issues/213
        ['test/attach_file_mini/attach_file_mini', []],
    ]

legion_gasnet_cxx_tests = [
    # Examples
    ['examples/mpi_interop/mpi_interop', []],
]

legion_openmp_cxx_tests = [
    # Examples
    ['examples/omp_saxpy/omp_saxpy', []],
]

legion_hdf_cxx_tests = [
    # Examples
    ['examples/attach_file/attach_file', []],

    # Tests
    ['test/hdf_attach_subregion_parallel/hdf_attach_subregion_parallel', ['-ll:cpu', '4']],
]

legion_cxx_perf_tests = [
    # Circuit: Heavy Compute
    ['examples/circuit/circuit',
     ['-l', '10', '-p', lambda nodes : nodes * int(perf_cores_per_node), '-npp', '2500', '-wpp', '10000', '-ll:cpu', perf_cores_per_node]],

    # Circuit: Light Compute
    ['examples/circuit/circuit',
     ['-l', '10', '-p', '100', '-npp', '2', '-wpp', '4', '-ll:cpu', '2']],
]

regent_perf_tests = [
    # Circuit: Heavy Compute
    ['language/examples/circuit_sparse.rg',
     ['-l', '10', '-p', lambda nodes : nodes * int(perf_cores_per_node), '-npp', '2500', '-wpp', '10000', '-ll:cpu', perf_cores_per_node,
      '-fflow-spmd-shardsize', perf_cores_per_node]],

    # Circuit: Light Compute
    ['language/examples/circuit_sparse.rg',
     ['-l', '10', '-p', '100', '-npp', '2', '-wpp', '4', '-ll:cpu', '2',
      '-fflow-spmd-shardsize', '2']],

    # PENNANT: Heavy Compute
    ['language/examples/pennant_fast.rg',
     ['pennant.tests/sedovbig3x30/sedovbig.pnt',
      '-seq_init', '0', '-par_init', '1', '-print_ts', '1', '-prune', '5',
      '-npieces', perf_cores_per_node, '-numpcx', '1', '-numpcy', perf_cores_per_node,
      '-ll:csize', '8192', '-ll:cpu', perf_cores_per_node, '-fflow-spmd-shardsize', perf_cores_per_node,
      '-fvectorize-unsafe', '1']],
]

def cmd(command, env=None, cwd=None):
    print(' '.join(command))
    return subprocess.check_call(command, env=env, cwd=cwd)

def run_test_regent(launcher, root_dir, tmp_dir, bin_dir, env, thread_count):
    cmd([os.path.join(root_dir, 'language/travis.py')], env=env)

def launcher_is_batch(launcher):
    if 'aprun' in launcher:
        return True
    return False

def launcher_with_nodes(perf_launcher, nodes):
    if 'aprun' in perf_launcher:
        return perf_launcher + ' -n ' + str(nodes)
    else:
        return perf_launcher

def batch_execute(nodes, command, env, cwd, launcher_name, root_dir):
    batch_dir = './batch'
    cmd(['mkdir', '-p', batch_dir])
    file_name = str(nodes) + '_nodes' + '_'.join(command[2:])
    file_name = file_name.replace('/', '_')
    file_name = file_name.replace(':', '_')
    file_name = file_name[:200]
    file_path = batch_dir + '/' + file_name

    file = open(file_path, 'w')
    file.write('#!/bin/bash\n')
    account = 'csc103'
    file.write('#PBS -A ' + account + '\n')    
    file.write('#PBS -N ' + file_name + '\n')
    file.write('#PBS -j oe\n')
    file.write('#PBS -l walltime=0:10:00,nodes=' + str(nodes) + '\n')
    file.write('#PBS -l gres=atlas1%atlas2\n')
    for key in env.keys():
        file.write('export ' + key.replace('()', '') + '="' + env[key].replace('()', '') + '"\n')
    file.write('cd $MEMBERWORK/' + account + '\n')
    file.write('date\n')
    command_string = ' '.join(command)
    file.write(launcher_with_nodes(env[launcher_name], nodes) + ' ' + command_string + '\n')
    file.write(os.path.join(root_dir, 'tools', 'perf_chart.py') + ' https://github.com/StanfordLegion/perf-data.git'
             + ' https://github.com/StanfordLegion/perf-data.git')
    file.write('date\n')
    file.close()
    cmd(['qsub', file_name], cwd=batch_dir)

def evaluate_lambda(test_flags, nodes):
    result = []
    for i in test_flags:
        if type(i) == type('string'):
            result = result + [i]
        elif type(i) == type(0):
            result = result + [str(i)]
        else:
            result = result + [str(i(nodes))]
    return result

def run_cxx(tests, flags, launcher, root_dir, bin_dir, env, thread_count, nodes=1):
    for test_file, test_flags in tests:
        test_flags = evaluate_lambda(test_flags, nodes)
        test_dir = os.path.dirname(os.path.join(root_dir, test_file))
        if bin_dir:
            test_path = os.path.join(bin_dir, os.path.basename(test_file))
        else:
            test_path = os.path.join(root_dir, test_file)
            cmd(['make', '-C', test_dir, '-j', str(thread_count)], env=env)
        if launcher_is_batch(env['PERF_LAUNCHER']):
            batch_execute(nodes, launcher + [test_path] + flags + test_flags, env=env, cwd=test_dir, launcher_name='PERF_LAUNCHER', root_dir=root_dir)
        else:
            cmd(launcher + [test_path] + flags + test_flags, env=env, cwd=test_dir)

def run_regent(tests, flags, launcher, root_dir, env, thread_count, nodes=1):
    for test_file, test_flags in tests:
        test_flags = evaluate_lambda(test_flags, nodes)
        test_dir = os.path.dirname(os.path.join(root_dir, test_file))
        test_path = os.path.join(root_dir, test_file)
        if launcher_is_batch(env['LAUNCHER']):
            batch_execute(nodes, launcher + [test_path] + flags + test_flags, env=env, cwd=test_dir, launcher_name='LAUNCHER', root_dir=root_dir)
        else:
            cmd(launcher + [test_path] + flags + test_flags, env=env, cwd=test_dir)

def run_test_legion_cxx(launcher, root_dir, tmp_dir, bin_dir, env, thread_count):
    flags = ['-logfile', 'out_%.log']
    run_cxx(legion_cxx_tests, flags, launcher, root_dir, bin_dir, env, thread_count)

def run_test_legion_gasnet_cxx(launcher, root_dir, tmp_dir, bin_dir, env, thread_count):
    flags = ['-logfile', 'out_%.log']
    run_cxx(legion_gasnet_cxx_tests, flags, launcher, root_dir, bin_dir, env, thread_count)

def run_test_legion_openmp_cxx(launcher, root_dir, tmp_dir, bin_dir, env, thread_count):
    flags = ['-logfile', 'out_%.log']
    run_cxx(legion_openmp_cxx_tests, flags, launcher, root_dir, bin_dir, env, thread_count)

def run_test_legion_hdf_cxx(launcher, root_dir, tmp_dir, bin_dir, env, thread_count):
    flags = ['-logfile', 'out_%.log']
    run_cxx(legion_hdf_cxx_tests, flags, launcher, root_dir, bin_dir, env, thread_count)

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
    solver = [[os.path.join(solver_dir, 'spmd_driver/solver'),
               ['-machine', '1', '-core', '8', '-mtxlvl', '6', '-ll:cpu', '8']]]
    run_cxx(solver, flags, launcher, root_dir, None, env, thread_count)

    # Parallel Research Kernels: Stencil
    # Contact: Wonchan Lee <wonchan@cs.stanford.edu>
    prk_dir = os.path.join(tmp_dir, 'prk')
    cmd(['git', 'clone', 'https://github.com/magnatelee/PRK.git', prk_dir])
    # This uses a custom Makefile that requires additional
    # configuration. Rather than go to that trouble it's easier to
    # just use a copy of the standard Makefile template.
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

    # SNAP
    # Contact: Mike Bauer <mbauer@nvidia.com>
    snap_dir = os.path.join(tmp_dir, 'snap')
    cmd(['git', 'clone', 'https://github.com/StanfordLegion/Legion-SNAP.git', snap_dir])
    # This can't handle flags before application arguments, so place
    # them after.
    snap = [[os.path.join(snap_dir, 'src/snap'),
             [os.path.join(snap_dir, 'input/mms.in')] + flags]]
    run_cxx(snap, [], launcher, root_dir, None, env, thread_count)

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

    # PENNANT
    # Contact: Galen Shipman <gshipman@lanl.gov>
    pennant_dir = os.path.join(tmp_dir, 'pennant')
    cmd(['git', 'clone', '-b', 'spmdv2',
         'git@github.com:gshipman/pennant-legion.git', pennant_dir])
    # This uses a custom Makefile that requires additional
    # configuration. Rather than go to that trouble it's easier to
    # just use a copy of the standard Makefile template.
    pennant_env = dict(list(env.items()) + [
        ('OUTFILE', 'pennant'),
        ('GEN_SRC', ' '.join(glob.glob(os.path.join(pennant_dir, 'src/*.cc')))),
        ('CC_FLAGS', (env['CC_FLAGS'] if 'CC_FLAGS' in env else '') +
         ' -std=c++11 -Wno-sign-compare -Wno-unknown-pragmas -Wno-unused-variable' +
         ' -D__STDC_FORMAT_MACROS -DDISABLE_BARRIER_MIGRATION'),
        ('WARN_AS_ERROR', '0'),
    ])
    makefile = os.path.join(root_dir, 'apps/Makefile.template')
    # Previous build uses -DASSUME_UNALLOCABLE. Clean first to get a fresh environment.
    cmd(['make', '-f', makefile, '-C', pennant_dir, 'clean'], env=pennant_env)
    cmd(['make', '-f', makefile, '-C', pennant_dir, '-j', str(thread_count)], env=pennant_env)
    pennant = os.path.join(pennant_dir, 'pennant')
    cmd([pennant, str(app_cores), 'test/sedovsmall/sedovsmall.pnt', '-ll:cpu', str(app_cores)],
        cwd=pennant_dir)

def hostname():
    return subprocess.check_output(['hostname']).strip()

def git_commit_id(repo_dir):
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=repo_dir).strip()

def git_branch_name(repo_dir):
    proc = subprocess.Popen(
        ['git', 'symbolic-ref', '--short', 'HEAD'], cwd=repo_dir,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, errors = proc.communicate()
    if proc.returncode == 0:
        return output.strip()
    return None




def run_test_perf(launcher, root_dir, tmp_dir, bin_dir, env, thread_count):
    nodes = 1
    while nodes < int(env['PERF_MAX_NODES']):
        run_test_perf_one_configuration(launcher, root_dir, tmp_dir, bin_dir, env, thread_count, nodes)
        nodes = nodes * 2

def run_test_perf_one_configuration(launcher, root_dir, tmp_dir, bin_dir, env, thread_count, nodes):
    flags = ['-logfile', 'out_%.log']

    # Performance test configuration:
    metadata = {
        'host': (os.environ['CI_RUNNER_DESCRIPTION']
                 if 'CI_RUNNER_DESCRIPTION' in os.environ else hostname()),
        'commit': (os.environ['CI_BUILD_REF'] if 'CI_BUILD_REF' in os.environ
                   else git_commit_id(root_dir)),
        'branch': (os.environ['CI_BUILD_REF_NAME'] if 'CI_BUILD_REF_NAME' in os.environ
                   else git_branch_name(root_dir)),
    }
    cxx_measurements = {
        # Hack: Use the command name as the benchmark name.
        'benchmark': {
            'type': 'argv',
            'index': 0,
            'filter': 'basename',
        },
        # Capture command line arguments following flags.
        'argv': {
            'type': 'argv',
            'start': 1 + len(flags),
        },
        # Record running time in seconds.
        'time_seconds': {
            'type': 'regex',
            'pattern': r'^ELAPSED TIME\s*=\s*(.*) s$',
            'multiline': True,
        }
    }
    regent_measurements = {
        # Hack: Use the command name as the benchmark name.
        'benchmark': {
            'type': 'argv',
            'index': 1,
            'filter': 'basename',
        },
        # Capture command line arguments following flags.
        'argv': {
            'type': 'argv',
            'start': 2,# + len(flags), # FIXME: Skipping flags, see below.
        },
        # Record running time in seconds.
        'time_seconds': {
            'type': 'command',
            'args': [
                os.path.join(root_dir, 'language/scripts/summarize.py'),
                '--machine-readable', '-',
            ],
        }
    }
    env = dict(list(env.items()) + [
        ('PERF_OWNER', 'StanfordLegion'),
        ('PERF_REPOSITORY', 'perf-data'),
        ('PERF_METADATA', json.dumps(metadata)),
    ])
    cxx_env = dict(list(env.items()) + [
        ('PERF_MEASUREMENTS', json.dumps(cxx_measurements)),
        # Launch through perf.py
        ('PERF_LAUNCHER', ' '.join(launcher)),
        ('LAUNCHER', ''),
    ])
    regent_env = dict(list(env.items()) + [
        ('PERF_MEASUREMENTS', json.dumps(regent_measurements)),
        # Launch through regent.py
        ('PERF_LAUNCHER', ''),
        ('LAUNCHER', ' '.join(launcher)),
    ])

    # Build Regent first to avoid recompiling later.
    cmd([os.path.join(root_dir, 'language/travis.py'), '--install-only'], env=env)

    # Run Legion C++ performance tests.
    runner = os.path.join(root_dir, 'perf.py')
    run_cxx(legion_cxx_perf_tests, flags, [runner], root_dir, bin_dir, cxx_env, thread_count, nodes)

    # Run Regent performance tests.
    regent_path = os.path.join(root_dir, 'language/regent.py')
    # FIXME: PENNANT can't handle the -logfile flag coming first, so just skip it.
    run_regent(regent_perf_tests, [], [runner, regent_path], root_dir, regent_env, thread_count, nodes)

    if not launcher_is_batch(launcher):
        # Render the final charts.
        subprocess.check_call(
            [os.path.join(root_dir, 'tools', 'perf_chart.py'),
             'https://github.com/StanfordLegion/perf-data.git',
             'https://github.com/StanfordLegion/perf-data.git'],
            env=env)

def check_test_legion_cxx(root_dir):
    print('Checking that tests that SHOULD tested are ACTUALLY tested...')
    print()

    # These are the directories we SHOULD have coverage for.
    should_dirs = ['tutorial', 'examples', 'test']
    should_tests = []
    for dir in should_dirs:
        entries = os.listdir(os.path.join(root_dir, dir))
        for entry in entries:
            if os.path.isdir(os.path.join(root_dir, dir, entry)):
                should_tests.append(os.path.join(dir, entry))
    assert len(should_tests) > 0

    # These are the tests we ACTUALLY have coverage for.
    tests = legion_cxx_tests + legion_gasnet_cxx_tests + \
            legion_openmp_cxx_tests + legion_hdf_cxx_tests
    actual_tests = set()
    for test_file, test_flags in tests:
        actual_tests.add(os.path.dirname(test_file))

    actual_tests.add('test/realm') # We test Realm separately.
    actual_tests.add('test/performance') # We test performance separately.

    # Check that all tests that SHOULD be covered are ACTUALLY covered.
    not_tests = []
    for should_test in should_tests:
        if should_test not in actual_tests:
            not_tests.append(should_test)
    if len(not_tests) > 0:
        print('The following tests are NOT currently being tested:')
        print()
        for not_test in not_tests:
            print('   %s' % not_test)
        print()
        raise Exception('There are tests that are NOT in the test suite')

def build_cmake(root_dir, tmp_dir, env, thread_count, test_legion_cxx, test_perf):
    build_dir = os.path.join(tmp_dir, 'build')
    install_dir = os.path.join(tmp_dir, 'install')
    os.mkdir(build_dir)
    os.mkdir(install_dir)
    cmd(['cmake', '-DCMAKE_INSTALL_PREFIX=%s' % install_dir,
         '-DCMAKE_BUILD_TYPE=%s' % (
             'Debug' if env['DEBUG'] == '1' else 'Release'),
         '-DLegion_USE_GASNet=%s' % (
             'ON' if env['USE_GASNET'] == '1' else 'OFF'),
         '-DLegion_USE_CUDA=%s' % (
             'ON' if env['USE_CUDA'] == '1' else 'OFF'),
         '-DLegion_USE_LLVM=%s' % (
             'ON' if env['USE_LLVM'] == '1' else 'OFF'),
         '-DLegion_USE_HDF5=%s' % (
             'ON' if env['USE_HDF'] == '1' else 'OFF')] +
        (['-DCMAKE_CXX_FLAGS=%s' % env['CC_FLAGS']]
          if 'CC_FLAGS' in env else []) +
        (['-DLegion_BUILD_TUTORIAL=ON',
          '-DLegion_BUILD_EXAMPLES=ON',
          '-DLegion_BUILD_TESTS=ON',
         ] if test_legion_cxx or test_perf else []) +
        [root_dir],
        env=env, cwd=build_dir)
    cmd(['make', '-C', build_dir, '-j', str(thread_count)], env=env)
    cmd(['make', '-C', build_dir, 'install'], env=env)
    return os.path.join(build_dir, 'bin')

def clean_cxx(tests, root_dir, env, thread_count):
    env = dict(list(env.items()) + [
        ('MAKEFLAGS', 's'), # Always silence initial clean.
    ])
    for test_file, test_flags in tests:
        test_dir = os.path.dirname(os.path.join(root_dir, test_file))
        cmd(['make', '-C', test_dir, 'clean'], env=env)

def build_make_clean(root_dir, env, thread_count, test_legion_cxx, test_perf,
                     test_external, test_private):
    # External and private also require cleaning, even though they get
    # built separately.
    if test_legion_cxx or test_perf or test_external or test_private:
        clean_cxx(legion_cxx_tests, root_dir, env, thread_count)

def option_enabled(option, options, var_prefix='', default=True):
    if options is not None: return option in options
    option_var = '%s%s' % (var_prefix, option.upper())
    if option_var in os.environ: return os.environ[option_var] == '1'
    return default

class Stage(object):
    __slots__ = ['name', 'begin_time']
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.begin_time = datetime.datetime.now()
        print()
        print('#'*60)
        print('### Entering Stage: %s' % self.name)
        print('#'*60)
        print()
        sys.stdout.flush()
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.datetime.now()
        print()
        print('#'*60)
        print('### Exiting Stage: %s' % self.name)
        print('###   * Exception Type: %s' % exc_type)
        print('###   * Elapsed Time: %s' % (end_time - self.begin_time))
        print('#'*60)
        print()
        sys.stdout.flush()

def report_mode(debug, launcher,
                test_regent, test_legion_cxx, test_fuzzer, test_realm,
                test_external, test_private, test_perf, use_gasnet,
                use_cuda, use_openmp, use_llvm, use_hdf, use_spy, use_gcov,
                use_cmake, use_rdir):
    print()
    print('#'*60)
    print('### Test Suite Configuration')
    print('###')
    print('### Debug:          %s' % debug)
    print('### Launcher:       %s' % launcher)
    print('###')
    print('### Running Tests:')
    print('###   * Regent:     %s' % test_regent)
    print('###   * Legion C++: %s' % test_legion_cxx)
    print('###   * Fuzzer:     %s' % test_fuzzer)
    print('###   * Realm:      %s' % test_realm)
    print('###   * External:   %s' % test_external)
    print('###   * Private:    %s' % test_private)
    print('###   * Perf:       %s' % test_perf)
    print('###')
    print('### Build Flags:')
    print('###   * GASNet:     %s' % use_gasnet)
    print('###   * CUDA:       %s' % use_cuda)
    print('###   * OpenMP:     %s' % use_openmp)
    print('###   * LLVM:       %s' % use_llvm)
    print('###   * HDF5:       %s' % use_hdf)
    print('###   * Spy:        %s' % use_spy)
    print('###   * Gcov:       %s' % use_gcov)
    print('###   * CMake:      %s' % use_cmake)
    print('###   * RDIR:       %s' % use_rdir)
    print('#'*60)
    print()
    sys.stdout.flush()

def run_tests(test_modules=None,
              debug=True,
              use_features=None,
              launcher=None,
              perf_max_nodes=1,
              thread_count=None,
              root_dir=None,
              check_ownership=False,
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
    test_legion_cxx = module_enabled('legion_cxx')
    test_fuzzer = module_enabled('fuzzer', debug)
    test_realm = module_enabled('realm', not debug)
    test_external = module_enabled('external', False)
    test_private = module_enabled('private', False)
    test_perf = module_enabled('perf', False)

    # Determine which features to build with.
    def feature_enabled(feature, default=True):
        return option_enabled(feature, use_features, 'USE_', default)
    use_gasnet = feature_enabled('gasnet', False)
    use_cuda = feature_enabled('cuda', False)
    use_openmp = feature_enabled('openmp', False)
    use_llvm = feature_enabled('llvm', False)
    use_hdf = feature_enabled('hdf', False)
    use_spy = feature_enabled('spy', False)
    use_gcov = feature_enabled('gcov', False)
    use_cmake = feature_enabled('cmake', False)
    use_rdir = feature_enabled('rdir', True)

    if test_perf and debug:
        raise Exception('Performance tests requested but DEBUG is enabled')

    if use_gasnet and launcher is None:
        raise Exception('GASNet is enabled but launcher is not set (use --launcher or LAUNCHER)')
    launcher = launcher.split() if launcher is not None else []

    gcov_flags = ' -ftest-coverage -fprofile-arcs'

    # Normalize the test environment.
    env = dict(list(os.environ.items()) + [
        ('DEBUG', '1' if debug else '0'),
        ('LAUNCHER', ' '.join(launcher)),
        ('USE_GASNET', '1' if use_gasnet else '0'),
        ('TEST_GASNET', '1' if use_gasnet else '0'),
        ('USE_CUDA', '1' if use_cuda else '0'),
        ('USE_OPENMP', '1' if use_openmp else '0'),
        ('TEST_OPENMP', '1' if use_openmp else '0'),
        ('USE_LLVM', '1' if use_llvm else '0'),
        ('USE_HDF', '1' if use_hdf else '0'),
        ('TEST_HDF', '1' if use_hdf else '0'),
        ('USE_SPY', '1' if use_spy else '0'),
        ('TEST_SPY', '1' if use_spy else '0'),
        ('TEST_GCOV', '1' if use_gcov else '0'),
        ('USE_RDIR', '1' if use_rdir else '0'),
        ('PERF_MAX_NODES', perf_max_nodes),
        ('LG_RT_DIR', os.path.join(root_dir, 'runtime'))] + (

        # Gcov doesn't get a USE_GCOV flag, but instead stuff the GCC
        # options for Gcov on to the compile and link flags.
        [('CC_FLAGS', (os.environ['CC_FLAGS'] + gcov_flags
                       if 'CC_FLAGS' in os.environ else gcov_flags)),
         ('LD_FLAGS', (os.environ['LD_FLAGS'] + gcov_flags
                       if 'LD_FLAGS' in os.environ else gcov_flags)),
        ] if use_gcov else []))

    if check_ownership:
        check_test_legion_cxx(root_dir)
        return

    report_mode(debug, launcher,
                test_regent, test_legion_cxx, test_fuzzer, test_realm,
                test_external, test_private, test_perf, use_gasnet,
                use_cuda, use_openmp, use_llvm, use_hdf, use_spy, use_gcov,
                use_cmake, use_rdir)

    tmp_dir = tempfile.mkdtemp(dir=root_dir)
    if verbose:
        print('Using build directory: %s' % tmp_dir)
        print()
    try:
        # Build tests.
        with Stage('build'):
            if use_cmake:
                bin_dir = build_cmake(
                    root_dir, tmp_dir, env, thread_count, test_legion_cxx, test_perf)
            else:
                # With GNU Make, builds happen inline. But clean here.
                build_make_clean(
                    root_dir, env, thread_count, test_legion_cxx, test_perf,
                    # These configurations also need to be cleaned first.
                    test_external, test_private)
                bin_dir = None

        # Run tests.
        if test_regent:
            with Stage('regent'):
                run_test_regent(launcher, root_dir, tmp_dir, bin_dir, env, thread_count)
        if test_legion_cxx:
            with Stage('legion_cxx'):
                run_test_legion_cxx(launcher, root_dir, tmp_dir, bin_dir, env, thread_count)
                if use_gasnet:
                    run_test_legion_gasnet_cxx(launcher, root_dir, tmp_dir, bin_dir, env, thread_count)
                if use_openmp:
                    run_test_legion_openmp_cxx(launcher, root_dir, tmp_dir, bin_dir, env, thread_count)
                if use_hdf:
                    run_test_legion_hdf_cxx(launcher, root_dir, tmp_dir, bin_dir, env, thread_count)
        if test_fuzzer:
            with Stage('fuzzer'):
                run_test_fuzzer(launcher, root_dir, tmp_dir, bin_dir, env, thread_count)
        if test_realm:
            with Stage('realm'):
                run_test_realm(launcher, root_dir, tmp_dir, bin_dir, env, thread_count)
        if test_external:
            with Stage('external'):
                run_test_external(launcher, root_dir, tmp_dir, bin_dir, env, thread_count)
        if test_private:
            with Stage('private'):
                run_test_private(launcher, root_dir, tmp_dir, bin_dir, env, thread_count)
        if test_perf:
            with Stage('perf'):
                run_test_perf(launcher, root_dir, tmp_dir, bin_dir, env, thread_count)
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
        description = 'Legion test suite')

    # What tests to run:
    parser.add_argument(
        '--test', dest='test_modules', action='append',
        choices=['regent', 'legion_cxx', 'fuzzer', 'realm', 'external', 'private', 'perf'],
        default=None,
        help='Test modules to run (also via TEST_*).')

    # Build options:
    parser.add_argument(
        '--debug', dest='debug', action='store_true',
        default=os.environ['DEBUG'] == '1' if 'DEBUG' in os.environ else True,
        help='Build Legion in debug mode (also via DEBUG).')
    parser.add_argument(
        '--no-debug', dest='debug', action='store_false',
        help='Disable debug mode (equivalent to DEBUG=0).')
    parser.add_argument(
        '--use', dest='use_features', action='append',
        choices=['gasnet', 'cuda', 'openmp', 'llvm', 'hdf', 'spy', 'gcov',
                 'cmake', 'rdir'],
        default=None,
        help='Build Legion with features (also via USE_*).')
    parser.add_argument(
        '--launcher', dest='launcher', action='store',
        default=os.environ['LAUNCHER'] if 'LAUNCHER' in os.environ else None,
        help='Launcher for Legion tests (also via LAUNCHER).')

    parser.add_argument(
        '--perf_max_nodes', dest='perf_max_nodes', action='store', default='1',
        help='Max nodes for performance scaling runs')

    parser.add_argument(
        '-C', '--directory', dest='root_dir', metavar='DIR', action='store', required=False,
        help='Legion root directory.')

    parser.add_argument(
        '-j', dest='thread_count', nargs='?', type=int,
        help='Number threads used to compile.')

    parser.add_argument(
        '--check', dest='check_ownership', action='store_true',
        help='Check for tests that are being skipped.')

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
