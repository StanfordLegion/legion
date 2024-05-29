#!/usr/bin/env python3

# Copyright 2024 Stanford University
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
import argparse, datetime, glob, json, multiprocessing, os, platform, shlex, shutil, subprocess, sys, traceback, tempfile
import signal
from pathlib import Path

make_exe = os.environ.get('MAKE', 'make')

# Find physical core count of the machine.
if platform.system() == 'Linux':
    lines = subprocess.check_output(['lscpu', '--parse=core']).decode('utf-8')
    physical_cores = len(set(line for line in lines.strip().split('\n')
                             if not line.startswith('#')))
elif platform.system() == 'Darwin':
    physical_cores = int(
        subprocess.check_output(['sysctl', '-n', 'hw.physicalcpu']).decode('utf-8'))
elif platform.system() == 'FreeBSD':
    physical_cores = int(
        subprocess.check_output(['sysctl', '-n', 'hw.ncpu']).decode('utf-8'))
    make_exe = os.environ.get('MAKE', 'gmake')  # default needs to be GNU make
else:
    raise Exception('Unknown platform: %s' % platform.system())

# Choose a reasonable number of application cores given the
# available physical cores.
app_cores = max(physical_cores - 2, 1)

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
    ['examples/realm_stencil/realm_stencil', ['-ll:cpu', '4']],
    ['examples/spmd_cgsolver/spmd_cgsolver', ['-ll:cpu', '4', '-perproc']],
    ['examples/virtual_map/virtual_map', []],
    ['examples/attach_2darray_c_fortran_layout/attach_2darray', []],
    ['examples/attach_array_daxpy/attach_array_daxpy', []],
    ['examples/attach_index_space/index_space_attach', []],
    ['examples/predication/predication', []],
    ['examples/layout_constraints/transpose', []],
    ['examples/padded_instances/padded_instances', []],
    ['examples/inline_tasks/inline_tasks', []],
    ['examples/allreduce/allreduce', []],
    ['examples/tree_collectives/tree_collectives', []],
    ['examples/local_function_tasks/local_function_tasks', []],
    ['examples/provenance/provenance', []],
    ['examples/tiling/tiling', []],
    ['examples/machine_config/machine_config', []],
    ['examples/future_map_transforms/future_map_transforms', []],
    ['examples/collective_writes/collective_writes', ['-ll:cpu', '4']],
    ['examples/concurrent_tasks/concurrent', ['-ll:cpu', '4']],
    # Comment this test out until it works everywhere
    #['examples/implicit_top_task/implicit_top_task', []],

    # Tests
    ['test/rendering/rendering', ['-i', '2', '-n', '64', '-ll:cpu', '4']],
    ['test/output_requirements/output_requirements', []],
    ['test/output_requirements/output_requirements', ['-replicate']],
    ['test/output_requirements/output_requirements', ['-index']],
    ['test/output_requirements/output_requirements', ['-index', '-replicate']],
    ['test/output_requirements/output_requirements', ['-empty']],
    ['test/output_requirements/output_requirements', ['-empty', '-replicate']],
    ['test/output_requirements/output_requirements', ['-empty', '-index']],
    ['test/output_requirements/output_requirements', ['-empty', '-index', '-replicate']],
    ['test/disjoint_complete/disjoint_complete', []],
    ['test/reduce_future/reduce_future', ['-ll:cpu', '4']],
    ['test/nested_replication/nested_replication', ['-ll:cpu', '4']],
    ['test/ctrl_repl_safety/ctrl_repl_safety', [':0:0', '-ll:cpu', '4']],
    ['test/ctrl_repl_safety/ctrl_repl_safety', [':0:1', '-ll:cpu', '4', '-lg:safe_ctrlrepl', '1']],
    ['test/ctrl_repl_safety/ctrl_repl_safety', [':1:0', '-ll:cpu', '4']],
    ['test/ctrl_repl_safety/ctrl_repl_safety', [':1:1', '-ll:cpu', '4', '-lg:safe_ctrlrepl', '1']],
    ['test/mapper/mapper', []],

    # Tutorial/realm
    ['tutorial/realm/hello_world/realm_hello_world', []],
    ['tutorial/realm/machine_model/realm_machine_model', []],
    ['tutorial/realm/events/realm_events', []],
    ['tutorial/realm/region_instances/realm_region_instances', []],
    ['tutorial/realm/deferred_allocation/realm_deferred_allocation', []],
    ['tutorial/realm/index_space_ops/realm_index_space_ops', []],
    ['tutorial/realm/index_space_copy_fill/realm_index_space_copy_fill', []],
    ['tutorial/realm/reductions/realm_reductions', []],
    ['tutorial/realm/barrier/realm_barrier', []],
    ['tutorial/realm/subgraph/realm_subgraph', []],
    ['tutorial/realm/reservation/realm_reservation', []],
    ['tutorial/realm/completion_queue/realm_completion_queue', []],
    ['tutorial/realm/profiling/realm_profiling', []],
]

if 'USE_CUDA' in os.environ and os.environ['USE_CUDA'] == 1:
    legion_cxx_tests += [
        ['tutorial/realm/cuda_interop/realm_cuda_interop', []],
    ]

legion_cxx_prof_tests = [
    ['examples/provenance/provenance', []],
    ['test/gather_perf/gather_perf', ['-m', '1']],
    ['test/gather_perf/gather_perf', ['-m', '2']],
    ['test/gather_perf/gather_perf', ['-m', '3']],
    ['test/gather_perf/gather_perf', ['-m', '4']],
    ['test/gather_perf/gather_perf', ['-m', '5']],
    ['test/gather_perf/gather_perf', ['-m', '6']],
    ['test/gather_perf/gather_perf', ['-m', '7']],
]

legion_fortran_tests = [
    ['tutorial/fortran/00_hello_world/hello_world_fortran', []],
    ['tutorial/fortran/01_tasks_and_futures/tasks_and_futures_fortran', []],
    ['tutorial/fortran/02_index_tasks/index_tasks_fortran', []],
    ['tutorial/fortran/03_physical_regions/physical_regions_fortran', []],
    ['tutorial/fortran/04_privileges_accessor/privileges_accessor_fortran', []],
    ['tutorial/fortran/05_privileges_raw_ptr/privileges_raw_ptr_fortran', []],
    ['tutorial/fortran/06_partitioning/partitioning_fortran', []],
    ['tutorial/fortran/07_partitioning_fortran_task/partitioning_fortran_task_fortran', []],
    ['tutorial/fortran/08_multiple_partitions/multiple_partitions_fortran', []],
    ['tutorial/fortran/09_region_2d/region_2d_fortran', []],
    ['tutorial/fortran/10_attach_array/attach_array_fortran', []],
]

if platform.system() != 'Darwin':
    legion_cxx_tests += [
        # FIXME: Fails non-deterministically on Mac OS: https://github.com/StanfordLegion/legion/issues/213
        ['test/attach_file_mini/attach_file_mini', []],
    ]

legion_network_cxx_tests = [
    # Examples
    ['examples/mpi_interop/mpi_interop', []],
    ['examples/mpi_with_ctrl_repl/mpi_with_ctrl_repl', []],
    # Tests
    ['test/bug954/bug954', ['-ll:rsize', '1024']],
]

legion_openmp_cxx_tests = [
    # Examples
    ['examples/omp_saxpy/omp_saxpy', ['-ll:ocpu', '1']],
]

legion_kokkos_cxx_tests = [
    # Examples
    ['examples/kokkos_saxpy/kokkos_saxpy', []],
]

legion_python_cxx_tests = [
    # Bindings
    ['bindings/python/legion_python', ['examples/domain.py', '-ll:py', '1', '-ll:cpu', '0']],
    ['bindings/python/legion_python', ['examples/domain_point.py', '-ll:py', '1', '-ll:cpu', '0']],
    ['bindings/python/legion_python', ['examples/domain_transform.py', '-ll:py', '1', '-ll:cpu', '0']],
    ['bindings/python/legion_python', ['examples/future.py', '-ll:py', '1', '-ll:cpu', '0']],
    ['bindings/python/legion_python', ['examples/hello.py', '-ll:py', '1', '-ll:cpu', '0']],
    ['bindings/python/legion_python', ['examples/import.py', '-ll:py', '1', '-ll:cpu', '0']],
    ['bindings/python/legion_python', ['examples/index_launch.py', '-ll:py', '1', '-ll:cpu', '0']],
    ['bindings/python/legion_python', ['examples/ispace.py', '-ll:py', '1', '-ll:cpu', '0']],
    ['bindings/python/legion_python', ['examples/method.py', '-ll:py', '1', '-ll:cpu', '0']],
    ['bindings/python/legion_python', ['examples/must_epoch.py', '-ll:py', '1', '-ll:cpu', '0']],
    ['bindings/python/legion_python', ['examples/partition.py', '-ll:py', '1', '-ll:cpu', '0']],
    ['bindings/python/legion_python', ['examples/partition_by_field.py', '-ll:py', '1', '-ll:cpu', '0']],
    ['bindings/python/legion_python', ['examples/partition_by_image.py', '-ll:py', '1', '-ll:cpu', '0']],
    ['bindings/python/legion_python', ['examples/partition_by_image_range.py', '-ll:py', '1', '-ll:cpu', '0']],
    ['bindings/python/legion_python', ['examples/partition_by_preimage.py', '-ll:py', '1', '-ll:cpu', '0']],
    ['bindings/python/legion_python', ['examples/partition_by_preimage_range.py', '-ll:py', '1', '-ll:cpu', '0']],
    ['bindings/python/legion_python', ['examples/partition_by_restriction.py', '-ll:py', '1', '-ll:cpu', '0']],
    ['bindings/python/legion_python', ['examples/reduction.py', '-ll:py', '1', '-ll:cpu', '0']],
    ['bindings/python/legion_python', ['examples/region.py', '-ll:py', '1', '-ll:cpu', '0']],
    ['bindings/python/legion_python', ['examples/region_fields.py', '-ll:py', '1', '-ll:cpu', '0']],
    ['bindings/python/legion_python', ['examples/return_region.py', '-ll:py', '1', '-ll:cpu', '0']],
    ['bindings/python/legion_python', ['examples/single_launch.py', '-ll:py', '1', '-ll:cpu', '0']],
    ['bindings/python/legion_python', ['examples/struct.py', '-ll:py', '1', '-ll:cpu', '0']],
    ['bindings/python/legion_python', ['examples/trace.py', '-ll:py', '1', '-ll:cpu', '0']],
    ['bindings/python/legion_python', ['examples/tunable.py', '-ll:py', '1', '-ll:cpu', '0']],
    ['bindings/python/legion_python', ['examples/types.py', '-ll:py', '1', '-ll:cpu', '0']],

    ['bindings/python/legion_python', ['tests/fail/privileges.py', '-ll:py', '1', '-ll:cpu', '0']],

    ['bindings/python/legion_python', ['tests/pass/copy.py', '-ll:py', '1', '-ll:cpu', '0']],
    ['bindings/python/legion_python', ['tests/pass/empty_region.py', '-ll:py', '1', '-ll:cpu', '0']],
    ['bindings/python/legion_python', ['tests/pass/print_once.py', '-ll:py', '1', '-ll:cpu', '0']],
    ['bindings/python/legion_python', ['tests/pass/privileges.py', '-ll:py', '1', '-ll:cpu', '0']],
    ['bindings/python/legion_python', ['tests/pass/no_access.py', '-ll:py', '1', '-ll:cpu', '0']],

    # Tests for Package Import
    ['bindings/python/legion_python', ['-m', 'tests.pass.test_package1.a.b.c', '-ll:py', '1', '-ll:cpu', '0']],
    ['bindings/python/legion_python', ['-m', 'tests.pass.test_package2.a.b.c', '-ll:py', '1', '-ll:cpu', '0']],

    # Examples
    ['examples/python_interop/python_interop', ['-ll:py', '1']],

    # Tests
    ['test/python_bindings/python_bindings', ['-ll:py', '1', '-ll:cpu', '0']],
]

legion_hdf_cxx_tests = [
    # Tests
    ['test/hdf_attach_subregion_parallel/hdf_attach_subregion_parallel', ['-ll:cpu', '4']],
]

if platform.system() != 'Darwin':
    legion_hdf_cxx_tests += [
        # FIXME: Fails non-deterministically on Mac OS: https://github.com/StanfordLegion/legion/issues/213
        ['examples/attach_file/attach_file', ['-h', 'data.h5', '-d', '/path/to/data']],
    ]

def get_legion_cxx_perf_tests(nodes, cores_per_node):
    return [
        # Circuit: Heavy Compute
        ['examples/circuit/circuit',
         ['-l', '10', '-p', str(cores_per_node * nodes), '-npp', '2500', '-wpp', '10000',
          '-ll:csize', '8192', '-ll:cpu', str(cores_per_node)]],

        # Circuit: Light Compute
        ['examples/circuit/circuit',
         ['-l', '10', '-p', '100', '-npp', '2', '-wpp', '4', '-ll:cpu', '2']],
    ]

def get_regent_perf_tests(nodes, cores_per_node):
    return [
        # Circuit: Heavy Compute
        ['language/examples/circuit_sparse.rg',
         ['-l', '10', '-p', str(nodes * cores_per_node), '-npp', '2500', '-wpp', '10000',
          '-ll:csize', '8192', '-ll:cpu', str(cores_per_node), '-fflow-spmd-shardsize', str(cores_per_node)]],

        # Circuit: Light Compute
        ['language/examples/circuit_sparse.rg',
         ['-l', '10', '-p', '100', '-npp', '2', '-wpp', '4', '-ll:cpu', '2',
          '-fflow-spmd-shardsize', '2']],

        # PENNANT: Heavy Compute
        ['language/examples/pennant_fast.rg',
         ['pennant.tests/sedovbig3x30/sedovbig.pnt',
          '-seq_init', '0', '-par_init', '1', '-print_ts', '1', '-prune', '5',
          '-npieces', str(nodes * cores_per_node), '-numpcx', '1', '-numpcy', str(nodes * cores_per_node),
          '-ll:csize', '8192', '-ll:cpu', str(cores_per_node), '-fflow-spmd-shardsize', str(cores_per_node),
          '-fvectorize-unsafe', '1']],
    ]

class TestTimeoutException(Exception):
    pass

def sigalrm_handler(signum, frame):
    raise TestTimeoutException

def cmd(command, env=None, cwd=None, timelimit=None):
    print(' '.join(command))
    sys.stdout.flush()  # python 2 doesn't have flush option in print
    if timelimit:
        child = subprocess.Popen(command, env=env, cwd=cwd)
        signal.signal(signal.SIGALRM, sigalrm_handler)
        signal.alarm(timelimit)
        try:
            ret = child.wait()
            signal.alarm(0)  # disable alarm
            signal.signal(signal.SIGALRM, signal.SIG_DFL)
            if ret:
                raise subprocess.CalledProcessError(ret, command)
            return ret
        except TestTimeoutException:
            child.kill()
            raise  # re-raise
    else:
        return subprocess.check_call(command, env=env, cwd=cwd)

def clone_github(namespace, repository, output_dir, tmp_dir, branch='master'):
    # GitHub clones fail regularly, so we need to replace "git clone"
    # with fetching the repository tarball.

    # cmd(['git', 'clone', '-b', branch, 'https://github.com/%s/%s.git' % (namespace, repository), output_dir])

    url = 'https://github.com/%s/%s/archive/refs/heads/%s.tar.gz' % (namespace, repository, branch)
    wget = subprocess.Popen(['wget', '-nv', '-O', '-', url], stdout=subprocess.PIPE)
    subprocess.check_call(['tar', '-zxf', '-'], stdin=wget.stdout, cwd=tmp_dir)
    if wget.wait() != 0:
        raise Exception('Fetching URL failed: %s' % url)
    os.rename(os.path.join(tmp_dir, '%s-%s' % (repository, branch)), output_dir)

def run_test_regent(launcher, root_dir, tmp_dir, bin_dir, env, thread_count):
    cmd([sys.executable, os.path.join(root_dir, 'language/travis.py')], env=env)

def run_cxx(tests, flags, launcher, root_dir, bin_dir, env, thread_count, timelimit):
    prev_built_dir = None
    for test_file, test_flags in tests:
        test_dir = os.path.dirname(os.path.join(root_dir, test_file))
        if bin_dir:
            test_path = os.path.join(bin_dir, os.path.basename(test_file))
        else:
            test_path = os.path.join(root_dir, test_file)
            # build if this is in a new directory
            if test_dir != prev_built_dir:
                # and clean up the previous directory to keep disk usage down
                if prev_built_dir:
                    cmd(['find', prev_built_dir , '-type', 'f', '(', '-name', '*.a', '-o', '-name', os.path.basename(test_file), ')', '-exec', 'rm', '-v', '{}', ';'])
                cmd([make_exe, '-C', test_dir, '-j', str(thread_count)], env=env)
                prev_built_dir = test_dir
        cmd(launcher + [test_path] + flags + test_flags, env=env, cwd=test_dir, timelimit=timelimit)
    if prev_built_dir:
        cmd(['find', prev_built_dir , '-type', 'f', '(', '-name', '*.a', '-o', '-name', os.path.basename(test_file), ')', '-exec', 'rm', '-v', '{}', ';'])

def run_regent(tests, flags, launcher, root_dir, env, thread_count, timelimit):
    for test_file, test_flags in tests:
        test_dir = os.path.dirname(os.path.join(root_dir, test_file))
        test_path = os.path.join(root_dir, test_file)
        cmd(launcher + [test_path] + flags + test_flags, env=env, cwd=test_dir, timelimit=timelimit)

def precompile_regent(tests, flags, launcher, root_dir, env, thread_count):
    exe_tests = []
    for test_file, test_flags in tests:
        test_dir = os.path.dirname(os.path.join(root_dir, test_file))
        test_path = os.path.join(root_dir, test_file)

        exe = os.path.splitext(test_path)[0] + '.exe'
        env = dict(list(env.items()) + [('OBJNAME', exe)])

        cmd(launcher + [test_path] + flags + test_flags, env=env, cwd=test_dir)

        exe_tests.append([exe, test_flags])
    return exe_tests

def get_default_args(env):
    flags = []
    if env['USE_CUDA'] == '1' or env['USE_HIP'] == '1':
        flags.extend(['-ll:gpu', '1', '-ll:fsize', '1024', '-ll:msize', '64'])
    if env['USE_KOKKOS'] == '1' and env['USE_OPENMP'] == '1':
        flags.extend(['-ll:ocpu', '1', '-ll:onuma', '0' ])
    return flags

def run_test_legion_cxx(launcher, root_dir, tmp_dir, bin_dir, env, thread_count, timelimit):
    flags = ['-logfile', 'out_%.log']
    flags.extend(get_default_args(env))
    run_cxx(legion_cxx_tests, flags, launcher, root_dir, bin_dir, env, thread_count, timelimit)

def run_test_legion_network_cxx(launcher, root_dir, tmp_dir, bin_dir, env, thread_count, timelimit):
    flags = ['-logfile', 'out_%.log']
    flags.extend(get_default_args(env))
    run_cxx(legion_network_cxx_tests, flags, launcher, root_dir, bin_dir, env, thread_count, timelimit)

def run_test_legion_openmp_cxx(launcher, root_dir, tmp_dir, bin_dir, env, thread_count, timelimit):
    flags = ['-logfile', 'out_%.log']
    flags.extend(get_default_args(env))
    run_cxx(legion_openmp_cxx_tests, flags, launcher, root_dir, bin_dir, env, thread_count, timelimit)

def run_test_legion_kokkos_cxx(launcher, root_dir, tmp_dir, bin_dir, env, thread_count, timelimit):
    flags = ['-logfile', 'out_%.log']
    flags.extend(get_default_args(env))
    run_cxx(legion_kokkos_cxx_tests, flags, launcher, root_dir, bin_dir, env, thread_count, timelimit)

def run_test_legion_python_cxx(launcher, root_dir, tmp_dir, bin_dir, env, thread_count, timelimit):
    # Hack: legion_python currently requires the module name to come first
    flags = [] # ['-logfile', 'out_%.log']
    flags.extend(get_default_args(env))
    python_dir = os.path.join(root_dir, 'bindings', 'python')
    # Hack: Fix up the environment so that Python can find all the examples.
    env = dict(list(env.items()) + [
        # In Make, this is where all Python files lives.
        # In CMake, we still need this, but only for tests.
        ('PYTHONPATH', ':'.join([env.get('PYTHONPATH'), python_dir])),
    ])
    if bin_dir is None:
        env['LD_LIBRARY_PATH'] = python_dir
    # If we're not already using shared libraries, clean up because
    # we're going to force them
    if bin_dir is None and env['SHARED_OBJECTS'] != '1':
        cmd([make_exe, '-C', python_dir, 'clean'], env=env)
    run_cxx(legion_python_cxx_tests, flags, launcher, root_dir, bin_dir, env, thread_count, timelimit)
    if bin_dir is None and env['SHARED_OBJECTS'] != '1':
        cmd([make_exe, '-C', python_dir, 'clean'], env=env)

def run_test_legion_jupyter_cxx(launcher, root_dir, tmp_dir, bin_dir, env, thread_count, timelimit):
    # Hack: legion_python currently requires the module name to come first
    flags = [] # ['-logfile', 'out_%.log']
    flags.extend(get_default_args(env))
    python_dir = os.path.join(root_dir, 'bindings', 'python')
    env = env.copy()
    if bin_dir is None:
        env['LD_LIBRARY_PATH'] = python_dir
    # If we're not already using shared libraries, clean up because
    # we're going to force them
    if bin_dir is None and env['SHARED_OBJECTS'] != '1':
        cmd([make_exe, '-C', python_dir, 'clean'], env=env)
    if bin_dir is None:
        cmd([make_exe, '-C', python_dir, '-j', str(thread_count)], env=env)
    jupyter_dir = os.path.join(root_dir, 'jupyter_notebook')
    jupyter_install_cmd = [sys.executable, './install_jupyter.py', '--legion-prefix', python_dir, '--verbose']
    cmd(jupyter_install_cmd, env=env, cwd=jupyter_dir)
    jupyter_test_file = os.path.join(root_dir, 'jupyter_notebook', 'ci_test.py')
    jupyter_test_cmd = ['jupyter', 'run', '--kernel', 'legion_kernel_nocr', jupyter_test_file]
    cmd(jupyter_test_cmd, env=env)
    canonical_jupyter_test_file = os.path.join(root_dir, 'jupyter_notebook', 'test_canonical.py')
    canonical_jupyter_test_cmd = ['jupyter', 'run', '--kernel', 'python3', canonical_jupyter_test_file]
    cmd(canonical_jupyter_test_cmd, env=env)
    canonical_python_test_cmd = [sys.executable, canonical_jupyter_test_file]
    cmd(canonical_python_test_cmd, env=env)
    if bin_dir is None and env['SHARED_OBJECTS'] != '1':
        cmd([make_exe, '-C', python_dir, 'clean'], env=env)

def run_test_legion_prof_cxx(launcher, root_dir, tmp_dir, bin_dir, env, thread_count, timelimit):
    flags = ['-lg:prof','1', '-lg:prof_logfile', 'prof_%.gz']
    flags.extend(get_default_args(env))
    from tools.test_prof import run_prof_test
    for test_file, test_flags in legion_cxx_prof_tests:
        prof_test = [[test_file, test_flags],]
        run_cxx(prof_test, flags, launcher, root_dir, bin_dir, env, thread_count, timelimit)
        test_file_path = Path(os.path.join(root_dir, test_file))
        test_dir = test_file_path.parent.absolute()
        run_prof_test(root_dir, test_dir, tmp_dir)

def run_test_legion_hdf_cxx(launcher, root_dir, tmp_dir, bin_dir, env, thread_count, timelimit):
    flags = ['-logfile', 'out_%.log']
    flags.extend(get_default_args(env))
    run_cxx(legion_hdf_cxx_tests, flags, launcher, root_dir, bin_dir, env, thread_count, timelimit)

def run_test_legion_fortran(launcher, root_dir, tmp_dir, bin_dir, env, thread_count, timelimit):
    flags = ['-logfile', 'out_%.log']
    flags.extend(get_default_args(env))
    run_cxx(legion_fortran_tests, flags, launcher, root_dir, bin_dir, env, thread_count, timelimit)

def run_test_fuzzer(launcher, root_dir, tmp_dir, bin_dir, env, thread_count):
    env = dict(list(env.items()) + [('WARN_AS_ERROR', '0')])
    fuzz_dir = os.path.join(tmp_dir, 'fuzz-tester')
    clone_github('StanfordLegion', 'fuzz-tester', fuzz_dir, tmp_dir)
    # TODO; Merge deppart branch into master after this makes it to stable Legion branch
    cmd(['git', 'checkout', 'deppart'], cwd=fuzz_dir)
    cmd(['python3', 'main.py'], env=env, cwd=fuzz_dir)

def run_test_realm(launcher, root_dir, tmp_dir, bin_dir, env, thread_count, timelimit):
    test_dir = os.path.join(root_dir, 'test/realm')
    cmd([make_exe, '-C', test_dir, 'DEBUG=0', 'clean'], env=env)
    cmd([make_exe, '-C', test_dir, 'DEBUG=0', '-j', str(thread_count), 'build'], env=env)
    cmd([make_exe, '-C', test_dir, 'DEBUG=0', 'run_all'], env=env, timelimit=timelimit)

def run_test_external1(launcher, root_dir, tmp_dir, bin_dir, env, thread_count, timelimit):
    flags = ['-logfile', 'out_%.log']

    # Realm perf test (move back to perf test when integrated with perf.py)
    perf_dir = os.path.join(root_dir, 'test/performance/realm')
    cmd([make_exe, '-C', perf_dir, 'DEBUG=0', 'clean_all'], env=env)
    cmd([make_exe, '-C', perf_dir, 'DEBUG=0', 'build_all'], env=env)
    cmd([make_exe, '-C', perf_dir, 'DEBUG=0', 'RUNMODE=short', 'run_all'], env=env, timelimit=timelimit)

    # Fast Direct Solver
    # Contact: Chao Chen <cchen10@stanford.edu>
    solver_dir = os.path.join(tmp_dir, 'fastSolver2')
    clone_github('Charles-Chao-Chen', 'fastSolver2', solver_dir, tmp_dir)
    solver = [[os.path.join(solver_dir, 'spmd_driver/solver'),
        ['-machine', '1', '-core', '8', '-mtxlvl', '6', '-ll:cpu', '8', '-ll:csize', '1024']]]
    run_cxx(solver, flags, launcher, root_dir, None, env, thread_count, timelimit)

    # Parallel Research Kernels: Stencil
    # Contact: Wonchan Lee <wonchan@cs.stanford.edu>
    prk_dir = os.path.join(tmp_dir, 'prk')
    clone_github('magnatelee', 'PRK', prk_dir, tmp_dir)
    # This uses a custom Makefile that requires additional
    # configuration. Rather than go to that trouble it's easier to
    # just use a copy of the standard Makefile template.
    stencil_dir = os.path.join(prk_dir, 'LEGION', 'Stencil')
    stencil_env = dict(list(env.items()) + [
        ('OUTFILE', 'stencil'),
        ('GEN_SRC', 'stencil.cc'),
        ('CXXFLAGS', (env['CXXFLAGS'] if 'CXXFLAGS' in env else '') +
         ' -DRADIUS=2 -DRESTRICT_KEYWORD -DDISABLE_BARRIER_MIGRATION'),
    ])
    makefile = os.path.join(root_dir, 'apps/Makefile.template')
    cmd([make_exe, '-f', makefile, '-C', stencil_dir, '-j', str(thread_count)], env=stencil_env)
    stencil = os.path.join(stencil_dir, 'stencil')
    # HACK: work around stencil mapper issue with -ll:ext_sysmem 0
    cmd([stencil, '4', '10', '1000', '-ll:ext_sysmem', '0'], env=env, timelimit=timelimit)

    # SNAP
    # Contact: Mike Bauer <mbauer@nvidia.com>
    snap_dir = os.path.join(tmp_dir, 'snap')
    # TODO: Merge deppart branch into master after this makes it to stable Legion branch
    clone_github('StanfordLegion', 'Legion-SNAP', snap_dir, tmp_dir, branch='ctrlrepl')
    # This can't handle flags before application arguments, so place
    # them after.
    snap = [[os.path.join(snap_dir, 'src/snap'),
             [os.path.join(snap_dir, 'input/mms.in')] + flags]]
    run_cxx(snap, [], launcher, root_dir, None, env, thread_count, timelimit) 

def run_test_external2(launcher, root_dir, tmp_dir, bin_dir, env, thread_count, timelimit, use_cuda):
    # HTR
    # Contact: Mario Di Renzo <direnzo.mario1@gmail.com>
    htr_dir = os.path.join(tmp_dir, 'htr')
    # clone_github('stanfordhpccenter', 'HTR-solver', htr_dir, tmp_dir)
    # NOTE: the legion-ci branch currently requires g++ (not clang) to build and
    #  is REALLY slow unless you set DEBUG=0
    cmd(['git', 'clone', '-b', 'legion-ci', 'git@gitlab.com:insieme1/htr/htr-solver.git', htr_dir])
    htr_env = dict(list(env.items()) + [
        ('LEGION_DIR', root_dir),
        ('LD_LIBRARY_PATH', '%s:%s' % (env.get('LD_LIBRARY_PATH', ''), os.path.join(root_dir, 'bindings', 'regent'))),
        ('HTR_DIR', htr_dir),
        ('CC', 'gcc'),
        ('CXX', 'g++'),
        ('DEBUG', '0'),
    ])

    # Try to auto-detect the runner's GPU_ARCH for the test
    if env['USE_CUDA'] == '1' and 'GPU_ARCH' not in env:
        try:
            device_query = subprocess.check_output(['nvidia-smi', '-i', '0', '-q']).decode('utf-8').splitlines()
            htr_env['GPU_ARCH'] = [line.split()[-1].lower() for line in device_query if line.strip().startswith('Product Architecture')][0]
            print("Auto-detected GPU_ARCH='%s'" % htr_env['GPU_ARCH'])
        except OSError:
            print('Command failed: %s' % cmd, file=sys.stderr, flush=True)
            raise

    cmd(['python3', os.path.join(htr_dir, 'unitTests', 'testAll.py')], env=htr_env)

    # TaskAMR
    # Contact: Jonathan Graham <jgraham@lanl.gov>
    task_amr_dir = os.path.join(tmp_dir, 'task_amr')
    clone_github('lanl', 'TaskAMR', task_amr_dir, tmp_dir)
    task_amr_env = dict(list(env.items()) + [
        ('LEGION_ROOT', root_dir),
    ])
    cmd([make_exe, '-C', os.path.join(task_amr_dir)], env=task_amr_env)

    # Barnes-Hut
    # Contact: Haithem Turki <turki.haithem@gmail.com>
    barnes_hut_dir = os.path.join(tmp_dir, 'barnes_hut')
    clone_github('StanfordLegion', 'barnes-hut', barnes_hut_dir, tmp_dir)
    regent_path = os.path.join(root_dir, 'language', 'regent.py')
    cmd([sys.executable, regent_path, 'hdf5_converter.rg',
         '-i', 'input/bodies-16384-blitz.csv',
         '-o', 'bodies-16384-blitz.h5',
         '-n', '16384'],
        cwd=barnes_hut_dir,
        env=env,
        timelimit=timelimit)
    cmd([sys.executable, regent_path, 'barnes_hut.rg',
         '-i', 'bodies-16384-blitz.h5',
         '-n', '16384'],
        cwd=barnes_hut_dir,
        env=env,
        timelimit=timelimit)

    # Since we're not actually testing this, don't both building on GPUs currently
    if not use_cuda:
        # Soleil-X
        # Contact: Manolis Papadakis <mpapadak@stanford.edu>
        soleil_dir = os.path.join(tmp_dir, 'soleil-x')
        clone_github('stanfordhpccenter', 'soleil-x', soleil_dir, tmp_dir)
        soleil_env = dict(list(env.items()) + [
            ('LEGION_DIR', root_dir),
            ('SOLEIL_DIR', soleil_dir),
            ('CC', 'gcc'),
        ])
        cmd([make_exe, '-C', os.path.join(soleil_dir, 'src')], env=soleil_env)
        # FIXME: Actually run it

def run_test_private(launcher, root_dir, tmp_dir, bin_dir, env, thread_count, timelimit):
    flags = ['-logfile', 'out_%.log']

    # PENNANT
    # Contact: Galen Shipman <gshipman@lanl.gov>
    pennant_dir = os.path.join(tmp_dir, 'pennant')
    clone_github('StanfordLegion', 'pennant-legion', pennant_dir, tmp_dir)
    # This uses a custom Makefile that requires additional
    # configuration. Rather than go to that trouble it's easier to
    # just use a copy of the standard Makefile template.
    pennant_env = dict(list(env.items()) + [
        ('OUTFILE', 'pennant'),
        ('GEN_SRC', ' '.join(glob.glob(os.path.join(pennant_dir, 'src/*.cc')))),
        ('CXXFLAGS', (env['CXXFLAGS'] if 'CXXFLAGS' in env else '') +
         ' -Wno-sign-compare -Wno-unknown-pragmas -Wno-unused-variable' +
         ' -D__STDC_FORMAT_MACROS -DDISABLE_BARRIER_MIGRATION'),
        ('WARN_AS_ERROR', '0'),
    ])
    makefile = os.path.join(root_dir, 'apps/Makefile.template')
    # Previous build uses -DASSUME_UNALLOCABLE. Clean first to get a fresh environment.
    cmd([make_exe, '-f', makefile, '-C', pennant_dir, 'clean'], env=pennant_env)
    cmd([make_exe, '-f', makefile, '-C', pennant_dir, '-j', str(thread_count)], env=pennant_env)
    pennant = os.path.join(pennant_dir, 'pennant')
    cmd([pennant, '-n', str(app_cores), '-f', 'test/sedov/sedov.pnt', '-ll:cpu', str(app_cores)],
        env=env,
        cwd=pennant_dir,
        timelimit=timelimit)


def run_test_ctest(launcher, root_dir, tmp_dir, bin_dir, env, thread_count, timelimit):
    build_dir = os.path.join(tmp_dir, 'build')
    args = ['ctest', '--output-on-failure']
    # do not run tests in parallel if they use GPUs - might not all fit
    if env['USE_CUDA'] != '1' and env['USE_HIP'] != '1':
        args.extend(['-j', str(thread_count)])
    if timelimit:
        args.extend(['--timeout', str(timelimit)])
    cmd(args,
        env=env,
        cwd=build_dir)

def run_test_legion_prof_mypy(root_dir):
    mypy_cmd = [
        "mypy",
        "--disallow-any-unimported",
        "--disallow-any-explicit",
        "--disallow-untyped-defs",
        "--disallow-incomplete-defs",
        "--warn-redundant-casts",
        "--warn-unused-ignores",
        os.path.join(root_dir, 'tools', 'legion_prof.py'),
    ]
    print('Running mypy test:', cmd)
    cmd(mypy_cmd)

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

def run_test_perf_one_configuration(launcher, root_dir, tmp_dir, bin_dir, env, thread_count, nodes, timelimit):
    flags = ['-logfile', 'out_%.log']

    # for backward-compatibility, use app_cores if PERF_CORES_PER_NODE is not specified
    cores_per_node = int(os.environ.get('PERF_CORES_PER_NODE', app_cores))

    legion_cxx_perf_tests = get_legion_cxx_perf_tests(nodes, cores_per_node)
    regent_perf_tests = get_regent_perf_tests(nodes, cores_per_node)

    # Regent needs special flags when in precompile mode
    precompile = os.environ.get('PERF_PRECOMPILE_REGENT') == '1'

    # Performance test configuration:
    metadata = {
        'host': (os.environ['CI_RUNNER_DESCRIPTION']
                 if 'CI_RUNNER_DESCRIPTION' in os.environ else hostname()),
        'nodes': nodes,
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
        ('PERF_LAUNCHER', ' '.join(launcher) if precompile else ''),
        ('LAUNCHER', '' if precompile else ' '.join(launcher)),
    ])

    # Build Regent first to avoid recompiling later.
    cmd([sys.executable, os.path.join(root_dir, 'language/travis.py'), '--install-only'], env=env)

    # Run Legion C++ performance tests.
    runner = os.path.join(root_dir, 'perf.py')
    run_cxx(legion_cxx_perf_tests, flags, [runner], root_dir, bin_dir, cxx_env, thread_count, timelimit)

    # Run Regent performance tests.
    regent_path = os.path.join(root_dir, 'language/regent.py')
    if precompile:
        # Precompile executables.
        build_env = dict(list(env.items()) + [
            ('SAVEOBJ', '1'),
            ('STANDALONE', '1'),
            ('LAUNCHER', ''),
        ])
        exe_tests = precompile_regent(regent_perf_tests, [], [regent_path], root_dir, build_env, thread_count)

        # FIXME: PENNANT can't handle the -logfile flag coming first, so just skip it.
        run_regent(exe_tests, [], [runner], root_dir, regent_env, thread_count)
    else:
        # FIXME: PENNANT can't handle the -logfile flag coming first, so just skip it.
        run_regent(regent_perf_tests, [], [runner, regent_path], root_dir, regent_env, thread_count)

    # Render the final charts.
    subprocess.check_call(
        [sys.executable,
         os.path.join(root_dir, 'tools', 'perf_chart.py'),
         'git@github.com:StanfordLegion/perf-data.git',
         'git@github.com:StanfordLegion/perf-data.git'],
        env=env)

def run_test_perf(launcher, root_dir, tmp_dir, bin_dir, env, thread_count, min_nodes, max_nodes, timelimit):
    nodes = min_nodes
    while nodes <= max_nodes:
        launcher = [w.format(**{'NODES': nodes}) for w in launcher]
        run_test_perf_one_configuration(launcher, root_dir, tmp_dir, bin_dir, env, thread_count, nodes, timelimit)
        nodes *= 2

def check_test_legion_cxx(root_dir):
    print('Checking that tests that SHOULD be tested are ACTUALLY tested...')
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
    tests = legion_cxx_tests + legion_network_cxx_tests + \
            legion_openmp_cxx_tests + legion_python_cxx_tests + \
            legion_hdf_cxx_tests + legion_kokkos_cxx_tests + \
            legion_fortran_tests
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

def build_cmake(root_dir, tmp_dir, env, thread_count,
                test_regent, test_legion_cxx,
                test_external1, test_external2, test_perf, test_ctest, test_realm_unit_ctest):
    build_dir = os.path.join(tmp_dir, 'build')
    install_dir = os.path.join(tmp_dir, 'install')
    os.mkdir(build_dir)
    os.mkdir(install_dir)
    cmake_cmd = ['cmake', '-DCMAKE_INSTALL_PREFIX=%s' % install_dir ]
    cmake_cmd.append('-DCMAKE_BUILD_TYPE=%s' % ('Debug' if env['DEBUG'] == '1' else
                                              'Release'))
    cmake_cmd.append('-DLegion_BUILD_WARN_AS_ERROR=%s' % ('ON' if env.get('WARN_AS_ERROR','0') == '1' else 'OFF'))
    cmake_cmd.append('-DLegion_MAX_DIM=%s' % env['MAX_DIM'])
    cmake_cmd.append('-DLegion_NETWORKS=%s' % env['REALM_NETWORKS'])
    if 'EMBED_GASNET' in env:
        cmake_cmd.append('-DLegion_EMBED_GASNet=ON')
        cmake_cmd.append('-DGASNet_CONDUIT=' + env['CONDUIT'])
        if 'EMBED_GASNET_SRC' in env:
            cmake_cmd.append('-DLegion_EMBED_GASNet_LOCALSRC=' + env['EMBED_GASNET_SRC'])
    cmake_cmd.append('-DLegion_USE_CUDA=%s' % ('ON' if env['USE_CUDA'] == '1' else 'OFF'))
    if 'GPU_ARCH' in env:
        cmake_cmd.append('-DLegion_CUDA_ARCH=%s' % env['GPU_ARCH'])
    cmake_cmd.append('-DLegion_USE_HIP=%s' % ('ON' if env['USE_HIP'] == '1' else 'OFF'))
    if 'HIP_ARCH' in env:
        cmake_cmd.append('-DLegion_HIP_ARCH=%s' % env['HIP_ARCH'])
    if 'THRUST_PATH' in env and env['USE_COMPLEX'] == '1':
        cmake_cmd.append('-DHIP_THRUST_ROOT_DIR=%s' % env['THRUST_PATH'])
    cmake_cmd.append('-DLegion_USE_NVTX=%s' % ('ON' if env['USE_NVTX'] == '1' else 'OFF'))
    cmake_cmd.append('-DLegion_USE_OpenMP=%s' % ('ON' if env['USE_OPENMP'] == '1' else 'OFF'))
    cmake_cmd.append('-DLegion_USE_Kokkos=%s' % ('ON' if env['USE_KOKKOS'] == '1' else 'OFF'))
    cmake_cmd.append('-DLegion_USE_Python=%s' % ('ON' if env['USE_PYTHON'] == '1' else 'OFF'))
    cmake_cmd.append('-DLegion_USE_LLVM=%s' % ('ON' if env['USE_LLVM'] == '1' else 'OFF'))
    cmake_cmd.append('-DLegion_USE_HDF5=%s' % ('ON' if env['USE_HDF'] == '1' else 'OFF'))
    cmake_cmd.append('-DLegion_USE_Fortran=%s' % ('ON' if env['LEGION_USE_FORTRAN'] == '1' else 'OFF'))
    cmake_cmd.append('-DLegion_SPY=%s' % ('ON' if env['USE_SPY'] == '1' else 'OFF'))
    cmake_cmd.append('-DLegion_BOUNDS_CHECKS=%s' % ('ON' if env['BOUNDS_CHECKS'] == '1' else 'OFF'))
    cmake_cmd.append('-DLegion_PRIVILEGE_CHECKS=%s' % ('ON' if env['PRIVILEGE_CHECKS'] == '1' else 'OFF'))
    cmake_cmd.append('-DLegion_REDOP_COMPLEX=%s' % ('ON' if env['USE_COMPLEX'] == '1' else 'OFF'))
    cmake_cmd.append('-DLegion_BACKTRACE_USE_LIBDW=%s' % ('ON' if env['REALM_BACKTRACE_USE_LIBDW'] == '1' else 'OFF'))
    if 'LEGION_WARNINGS_FATAL' in env:
        cmake_cmd.append('-DLegion_WARNINGS_FATAL=%s' % ('ON' if env['LEGION_WARNINGS_FATAL'] == '1' else 'OFF'))
    if test_ctest:
        cmake_cmd.append('-DLegion_ENABLE_TESTING=ON')
        if 'LAUNCHER' in env:
            cmake_cmd.append('-DLegion_TEST_LAUNCHER=%s' % env['LAUNCHER'])
        cmake_cmd.append('-DLegion_TEST_ARGS=%s' % ' '.join(get_default_args(env)))
        if test_realm_unit_ctest:
            cmake_cmd.append('-DLegion_BUILD_REALM_UNIT_TESTS=ON')
    else:
        cmake_cmd.append('-DLegion_ENABLE_TESTING=OFF')
    if test_regent or (test_legion_cxx and (env['USE_PYTHON'] == '1')):
        cmake_cmd.append('-DLegion_BUILD_BINDINGS=ON')
    if test_legion_cxx or test_ctest:
        cmake_cmd.extend(['-DLegion_BUILD_APPS=ON',
                        '-DLegion_BUILD_EXAMPLES=ON',
                        '-DLegion_BUILD_TUTORIAL=ON',
                        '-DLegion_BUILD_TESTS=ON',
                        ])
    # several different conditions force the use of shared libraries
    if test_regent or test_external1 or test_external2 or (env['USE_PYTHON'] == '1') or (env['SHARED_OBJECTS'] == '1'):
        cmake_cmd.append('-DBUILD_SHARED_LIBS=ON')
    else:
        cmake_cmd.append('-DBUILD_SHARED_LIBS=OFF')
    # if MARCH is set in the environment, give that to cmake as BUILD_MARCH
    if 'MARCH' in env:
        cmake_cmd.append('-DBUILD_MARCH=' + env['MARCH'])
    # add cxx standard
    if 'CXX_STANDARD' in env:
        cmake_cmd.append('-DCMAKE_CXX_STANDARD=' + env['CXX_STANDARD'])
    # cmake before 3.16 doesn't know how to look for CUDAHOSTCXX
    if 'CUDAHOSTCXX' in env:
        cmake_cmd.append('-DCMAKE_CUDA_HOST_COMPILER=' + env['CUDAHOSTCXX'])
    # add any extra cmake args requested in the environment
    if 'EXTRA_CMAKE_ARGS' in env:
        cmake_cmd.extend(shlex.split(env['EXTRA_CMAKE_ARGS']))
    # last argument to cmake is the root of the tree
    cmake_cmd.append(root_dir)

    cmd(cmake_cmd, env=env, cwd=build_dir)
    cmd([make_exe, '-C', build_dir, 'install', '-j', str(thread_count)], env=env)

    bin_dir = os.path.join(build_dir, 'bin')
    python_dir = None
    if env['USE_PYTHON'] == '1':
        for dir_entry in os.scandir(os.path.join(install_dir, 'lib')):
            if dir_entry.name.startswith('python'):
                site_packages = os.path.join(dir_entry.path, 'site-packages')
                if os.path.exists(site_packages):
                    python_dir = site_packages
        if python_dir is None:
            raise Exception('Unable to find Python site-packages in installation directory')
    return bin_dir, python_dir

def build_legion_prof_rs(root_dir, tmp_dir, env):
    legion_prof_dir = os.path.join(root_dir, 'tools', 'legion_prof_rs')
    cmd(['cargo', 'install',
         '--all-features',
         '--locked',
         '--debug', # Enables debug checks. Still optimizes like -O2.
         '--path', legion_prof_dir,
         '--root', tmp_dir],
        env=env)
    cmd(['cargo', 'test', '--all-features'], env=env, cwd=legion_prof_dir)
    cmd(['cargo', 'fmt', '--all', '--', '--check'], env=env, cwd=legion_prof_dir)

def build_regent(root_dir, env):
    cmd([os.path.join(root_dir, 'language/travis.py'), '--install-only'], env=env)

def clean_cxx(tests, root_dir, env, thread_count):
    env = dict(list(env.items()) + [
        ('MAKEFLAGS', 's'), # Always silence initial clean.
    ])
    for test_file, test_flags in tests:
        test_dir = os.path.dirname(os.path.join(root_dir, test_file))
        cmd([make_exe, '-C', test_dir, 'clean'], env=env)

def build_make_clean(root_dir, env, thread_count, test_legion_cxx, test_perf,
                     test_external1, test_external2, test_private):
    # External and private also require cleaning, even though they get
    # built separately.
    if test_legion_cxx or test_perf or test_external1 or test_external2 or test_private:
        clean_cxx(legion_cxx_tests, root_dir, env, thread_count)
    if test_legion_cxx and env['LEGION_USE_FORTRAN'] == '1':
        clean_cxx(legion_fortran_tests, root_dir, env, thread_count)

def build_make(root_dir, tmp_dir, env, thread_count, networks):
    build_dir = os.path.join(tmp_dir, 'build')
    install_dir = os.path.join(tmp_dir, 'install')
    os.mkdir(build_dir)
    os.mkdir(install_dir)
    makefile = os.path.join(build_dir, 'Makefile')
    shutil.copyfile(os.path.join(root_dir, 'apps', 'Makefile.template'), makefile)
    # We'll just always for shared objects here for performance
    env['SHARED_OBJECTS'] = '1'
    # If we have networks always turn on the use of the network
    if networks:
        env['USE_NETWORK'] = '1'
    local_env = dict(list(env.items()) + [
        ('PREFIX', install_dir),
    ])
    cmd([make_exe, '-C', build_dir, 'install', '-j', str(thread_count)], env=local_env)
    # Setup the LEGION_DIR for the Makefile to use that instead of building everything from source
    env['LG_INSTALL_DIR'] = install_dir
    if platform.system() == 'Darwin':
        # Be aware this doesn't really work on subprocessses on MacOS systems that have their system
        # integrity protections enabled which will prevent this from having any effect
        # https://stackoverflow.com/questions/35568122/why-isnt-dyld-library-path-being-propagated-here
        ld_path = env.get('DYLD_LIBRARY_PATH', '')
        env['DYLD_LIBRARY_PATH'] = ld_path+':'+os.path.join(install_dir,'lib')
    else:
        ld_path = env.get('LD_LIBRARY_PATH', '')
        env['LD_LIBRARY_PATH'] = ld_path+':'+os.path.join(install_dir,'lib')

def option_enabled(option, options, default, envprefix='', envname=None):
    if options is not None: return option in options
    if envname is not None:
        option_var = envname
    else:
        option_var = '%s%s' % (envprefix, option.upper())
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

def report_mode(debug, max_dim, launcher,
                test_regent, test_legion_cxx, test_fuzzer, test_realm,
                test_external1, test_external2, test_private,
                test_perf, test_ctest, test_realm_unit_ctest, test_jupyter, networks,
                use_cuda, use_hip, use_openmp, use_kokkos, use_python, use_llvm,
                use_hdf, use_fortran, use_spy, use_prof,
                use_bounds_checks, use_privilege_checks, use_complex,
                use_shared_objects,
                use_gcov, use_cmake, use_nvtx, use_libdw, cxx_standard):
    print()
    print('#'*60)
    print('### Test Suite Configuration')
    print('###')
    print('### Python:')
    print('\n'.join(['###   ' + line for line in  sys.version.split('\n')]))
    print('###')
    print('### Debug:          %s' % debug)
    print('### Launcher:       %s' % launcher)
    print('###')
    print('### Running Tests:')
    print('###   * Regent:     %s' % test_regent)
    print('###   * Legion C++: %s' % test_legion_cxx)
    print('###   * Fuzzer:     %s' % test_fuzzer)
    print('###   * Realm:      %s' % test_realm)
    print('###   * External1:  %s' % test_external1)
    print('###   * External2:  %s' % test_external2)
    print('###   * Private:    %s' % test_private)
    print('###   * Perf:       %s' % test_perf)
    print('###   * CTest:      %s' % test_ctest)
    print('###   * Realm Unit CTest:      %s' % test_realm_unit_ctest)
    print('###   * Jupyter:    %s' % test_jupyter)
    print('###')
    print('### Build Flags:')
    print('###   * Networks:   %s' % networks)
    print('###   * CUDA:       %s' % use_cuda)
    print('###   * HIP:        %s' % use_hip)
    print('###   * OpenMP:     %s' % use_openmp)
    print('###   * Kokkos:     %s' % use_kokkos)
    print('###   * Python:     %s' % use_python)
    print('###   * LLVM:       %s' % use_llvm)
    print('###   * HDF5:       %s' % use_hdf)
    print('###   * Fortran:    %s' % use_fortran)
    print('###   * Spy:        %s' % use_spy)
    print('###   * Prof:       %s' % use_prof)
    print('###   * Bounds:     %s' % use_bounds_checks)
    print('###   * Privilege:  %s' % use_privilege_checks)
    print('###   * Cplx Redop: %s' % use_complex)
    print('###   * Shared Obj: %s' % use_shared_objects)
    print('###   * Gcov:       %s' % use_gcov)
    print('###   * CMake:      %s' % use_cmake)
    print('###   * NVTX:       %s' % use_nvtx)
    print('###   * LIBDW:      %s' % use_libdw)
    print('###   * Max DIM:    %s' % max_dim)
    print('###   * C++ STD:    %s' % cxx_standard)
    print('#'*60)
    print()
    sys.stdout.flush()

def run_tests(test_modules=None,
              debug=True,
              max_dim=3,
              use_features=None,
              networks='',
              launcher=None,
              thread_count=None,
              root_dir=None,
              tmp_dir=None,
              check_ownership=False,
              keep_tmp_dir=False,
              timelimit=None,
              verbose=False):
    if thread_count is None:
        try:
            # this correctly considers the current affinity mask
            thread_count = len(os.sched_getaffinity(0))
        except AttributeError:
            # this works on macos
            thread_count = multiprocessing.cpu_count()

    if root_dir is None:
        root_dir = os.path.dirname(os.path.realpath(__file__))

    if timelimit is None:
        if 'TIMELIMIT' in os.environ:
            timelimit = int(os.environ['TIMELIMIT'])

    # Determine which test modules to run.
    def module_enabled(module, default=True, prefix='TEST_', **kwargs):
        return option_enabled(module, test_modules, default,
                              envprefix=prefix, **kwargs)
    test_regent = module_enabled('regent')
    test_legion_cxx = module_enabled('legion_cxx')
    test_fuzzer = module_enabled('fuzzer', False)
    test_realm = module_enabled('realm', not debug)
    test_external1 = module_enabled('external1', False)
    test_external2 = module_enabled('external2', False)
    test_private = module_enabled('private', False)
    test_perf = module_enabled('perf', False)
    test_ctest = module_enabled('ctest', False)
    test_realm_unit_ctest = module_enabled('realm_unit_ctest', False)
    test_jupyter = module_enabled('jupyter', False)

    # Determine which features to build with.
    def feature_enabled(feature, default=True, prefix='USE_', **kwargs):
        return option_enabled(feature, use_features, default,
                              envprefix=prefix, **kwargs)
    use_cuda = feature_enabled('cuda', False)
    use_hip = feature_enabled('hip', False)
    use_openmp = feature_enabled('openmp', False)
    use_kokkos = feature_enabled('kokkos', False)
    use_python = feature_enabled('python', False)
    use_llvm = feature_enabled('llvm', False)
    use_hdf = feature_enabled('hdf', False)
    use_fortran = feature_enabled('fortran', False, prefix='LEGION_USE_')
    use_spy = feature_enabled('spy', False)
    use_prof = feature_enabled('prof', False)
    use_bounds_checks = feature_enabled('bounds', False,
                                        envname='BOUNDS_CHECKS')
    use_privilege_checks = feature_enabled('privilege', False,
                                           envname='PRIVILEGE_CHECKS')
    use_complex = feature_enabled('complex', True)
    use_gcov = feature_enabled('gcov', False)
    use_cmake = feature_enabled('cmake', False)
    use_nvtx = feature_enabled('nvtx', False)
    use_libdw = feature_enabled('libdw', False, prefix='REALM_BACKTRACE_USE_')
    use_shared_objects = feature_enabled('shared', False,
                                         envname='SHARED_OBJECTS')

    if use_kokkos and not use_cmake:
        raise Exception('Kokkos support requires use of CMake')

    # Determine parameters for performance tests.
    if test_perf:
        if 'PERF_MIN_NODES' not in os.environ:
            raise Exception('Performance tests requested but PERF_MIN_NODES is not set')
        min_nodes = int(os.environ['PERF_MIN_NODES'])
        if 'PERF_MAX_NODES' not in os.environ:
            raise Exception('Performance tests requested but PERF_MAX_NODES is not set')
        max_nodes = int(os.environ['PERF_MAX_NODES'])

    if test_perf and debug:
        raise Exception('Performance tests requested but DEBUG is enabled')

    if test_ctest and not use_cmake:
        raise Exception('CTest cannot be used without CMake')

    if test_jupyter and not use_python:
        raise Exception('Jupyter requires Python')

    if networks and launcher is None:
        raise Exception('Network(s) is enabled but launcher is not set (use --launcher or LAUNCHER)')
    launcher = launcher.split() if launcher is not None else []

    # CXX Standard
    cxx_standard = os.environ.get('CXX_STANDARD', '')
    # if not use cmake, let's add -std=c++NN to CXXFLAGS
    if not use_cmake and cxx_standard != '':
        cxx_std_flag = " -std=c++" + cxx_standard
        os.environ['CXXFLAGS'] = os.environ.get('CXXFLAGS', '') + cxx_std_flag
        if use_cuda:
            os.environ['NVCC_FLAGS'] = os.environ.get('NVCC_FLAGS', '') + cxx_std_flag
        if use_hip:
            os.environ['HIPCC_FLAGS'] = os.environ.get('HIPCC_FLAGS', '') + cxx_std_flag

    gcov_flags = ' -ftest-coverage -fprofile-arcs'

    if check_ownership:
        check_test_legion_cxx(root_dir)
        return

    report_mode(debug, max_dim, launcher,
                test_regent, test_legion_cxx, test_fuzzer, test_realm,
                test_external1, test_external2, test_private,
                test_perf, test_ctest, test_realm_unit_ctest, test_jupyter,
                networks,
                use_cuda, use_hip, use_openmp, use_kokkos, use_python, use_llvm,
                use_hdf, use_fortran, use_spy, use_prof,
                use_bounds_checks, use_privilege_checks, use_complex,
                use_shared_objects,
                use_gcov, use_cmake, use_nvtx, use_libdw, cxx_standard)

    if not tmp_dir:
        tmp_dir = tempfile.mkdtemp(dir=root_dir)
    else:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.mkdir(tmp_dir)

    if verbose:
        print('Using build directory: %s' % tmp_dir)
        print()

    # Normalize the test environment.
    env = dict(list(os.environ.items()) + [
        ('DEBUG', '1' if debug else '0'),
        ('LAUNCHER', ' '.join(launcher)),
        ('REALM_NETWORKS', networks),
        ('USE_CUDA', '1' if use_cuda else '0'),
        ('TEST_CUDA', '1' if use_cuda else '0'),
        ('USE_HIP', '1' if use_hip else '0'),
        ('TEST_HIP', '1' if use_hip else '0'),
        ('USE_OPENMP', '1' if use_openmp else '0'),
        ('TEST_OPENMP', '1' if use_openmp else '0'),
        ('USE_KOKKOS', '1' if use_kokkos else '0'),
        ('TEST_KOKKOS', '1' if use_kokkos else '0'),
        ('USE_PYTHON', '1' if use_python else '0'),
        ('TEST_PYTHON', '1' if use_python else '0'),
        ('USE_LLVM', '1' if use_llvm else '0'),
        ('USE_HDF', '1' if use_hdf else '0'),
        ('TEST_HDF', '1' if use_hdf else '0'),
        ('LEGION_USE_FORTRAN', '1' if use_fortran else '0'),
        ('TEST_FORTRAN', '1' if use_fortran else '0'),
        ('USE_SPY', '1' if use_spy else '0'),
        ('TEST_SPY', '1' if use_spy else '0'),
        ('USE_PROF', '1' if use_prof else '0'),
        ('TEST_PROF', '1' if use_prof else '0'),
        ('BOUNDS_CHECKS', '1' if use_bounds_checks else '0'),
        ('PRIVILEGE_CHECKS', '1' if use_privilege_checks else '0'),
        ('USE_COMPLEX', '1' if use_complex else '0'),
        ('SHARED_OBJECTS', '1' if use_shared_objects else '0'),
        ('TEST_GCOV', '1' if use_gcov else '0'),
        ('USE_NVTX', '1' if use_nvtx else '0'),
        ('REALM_BACKTRACE_USE_LIBDW', '1' if use_libdw else '0'),
        ('MAX_DIM', str(max_dim)),
        ('LG_RT_DIR', os.path.join(root_dir, 'runtime')),
        ('DEFINE_HEADERS_DIR', os.path.join(root_dir, 'runtime')),
        ('CMAKE_BUILD_DIR', os.path.join(tmp_dir, 'build')),
        ('TMP_BIN_DIR', os.path.join(tmp_dir, 'bin'))] + (

        # Gcov doesn't get a USE_GCOV flag, but instead stuff the GCC
        # options for Gcov on to the compile and link flags.
        [('CXXFLAGS', (os.environ['CXXFLAGS'] + gcov_flags
                       if 'CXXFLAGS' in os.environ else gcov_flags)),
         ('LDFLAGS', (os.environ['LDFLAGS'] + gcov_flags
                       if 'LDFLAGS' in os.environ else gcov_flags)),
        ] if use_gcov else []))

    try:
        # Build tests.
        with Stage('build'):
            if use_prof or use_spy:
                build_legion_prof_rs(root_dir, tmp_dir, env)
            if use_prof:
                run_test_legion_prof_mypy(root_dir)
            if use_cmake:
                # We should always be using ctest if we're building with
                # cmake, except for some unusual cases with Regent
                # (ask @eslaught for details about Regent cases)
                assert test_ctest or test_regent
                bin_dir, python_dir = build_cmake(
                    root_dir, tmp_dir, env, thread_count,
                    test_regent, test_legion_cxx, test_external1,
                    test_external2,
                    test_perf, test_ctest, test_realm_unit_ctest)
            else:
                # With GNU Make, builds happen inline. But clean here.
                build_make_clean(
                    root_dir, env, thread_count, test_legion_cxx, test_perf,
                    # These configurations also need to be cleaned first.
                    test_external1, test_external2, test_private)
                # Build just one copy of the runtime unless we're running regent tests
                if not test_regent and not test_external2:
                    build_make(root_dir, tmp_dir, env, thread_count, networks)
                bin_dir = None
                python_dir = None
                if use_python:
                    python_dir = os.path.join(root_dir, 'bindings', 'python')

        # Set PYTHONPATH for Python tests.
        if use_python:
            env['PYTHONPATH'] = python_dir

        # Run tests.
        if test_regent:
            with Stage('regent'):
                run_test_regent(launcher, root_dir, tmp_dir, bin_dir, env, thread_count)
        if test_legion_cxx:
            with Stage('legion_cxx'):
                run_test_legion_cxx(launcher, root_dir, tmp_dir, bin_dir, env, thread_count, timelimit)
                if use_prof:
                    run_test_legion_prof_cxx(launcher, root_dir, tmp_dir, bin_dir, env, thread_count, timelimit)
                if networks:
                    run_test_legion_network_cxx(launcher, root_dir, tmp_dir, bin_dir, env, thread_count, timelimit)
                if use_openmp:
                    run_test_legion_openmp_cxx(launcher, root_dir, tmp_dir, bin_dir, env, thread_count, timelimit)
                if use_kokkos:
                    run_test_legion_kokkos_cxx(launcher, root_dir, tmp_dir, bin_dir, env, thread_count, timelimit)
                if use_python:
                    run_test_legion_python_cxx(launcher, root_dir, tmp_dir, bin_dir, env, thread_count, timelimit)
                if use_hdf:
                    run_test_legion_hdf_cxx(launcher, root_dir, tmp_dir, bin_dir, env, thread_count, timelimit)
                if use_fortran:
                    run_test_legion_fortran(launcher, root_dir, tmp_dir, bin_dir, env, thread_count, timelimit)
        if test_fuzzer:
            with Stage('fuzzer'):
                run_test_fuzzer(launcher, root_dir, tmp_dir, bin_dir, env, thread_count)
        if test_realm and not test_ctest:
            with Stage('realm'):
                run_test_realm(launcher, root_dir, tmp_dir, bin_dir, env, thread_count, timelimit)
        if test_external1:
            with Stage('external1'):
                run_test_external1(launcher, root_dir, tmp_dir, bin_dir, env, thread_count, timelimit)
        if test_external2:
            with Stage('external2'):
                if not test_regent:
                    build_regent(root_dir, env)
                run_test_external2(launcher, root_dir, tmp_dir, bin_dir, env, thread_count, timelimit, use_cuda)
        if test_private:
            with Stage('private'):
                run_test_private(launcher, root_dir, tmp_dir, bin_dir, env, thread_count, timelimit)
        if test_perf:
            with Stage('perf'):
                run_test_perf(launcher, root_dir, tmp_dir, bin_dir, env, thread_count, min_nodes, max_nodes, timelimit)
        if test_ctest:
            with Stage('ctest'):
                run_test_ctest(launcher, root_dir, tmp_dir, bin_dir, env, thread_count, timelimit)
        if test_jupyter:
            with Stage('jupyter'):
                run_test_legion_jupyter_cxx(launcher, root_dir, tmp_dir, bin_dir, env, thread_count, timelimit)
    finally:
        if keep_tmp_dir:
            print('Leaving build directory:')
            print('  %s' % tmp_dir)
        else:
            if verbose:
                print('Removing build directory:')
                print('  %s' % tmp_dir)
            shutil.rmtree(tmp_dir)

# behaves enough like a normal list for ArgumentParser's needs, except for
#  the __contains__ method, which accepts a list of values and checks each
#  one for membership
class MultipleChoiceList(object):
    def __init__(self, *args):
        self.list = list(args)

    def __contains__(self, x):
        if type(x) is list:
            for v in x:
                if v not in self.list:
                    return False
            return True
        else:
            return x in self.list

    def __iter__(self):
        return self.list.__iter__()

class ExtendAction(argparse.Action):
    def __init__(self, **kwargs):
        super(ExtendAction, self).__init__(**kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest, None)
        items = items[:] if items else []
        if type(values) is list:
            items.extend(values)
        else:
            items.append(values)
        setattr(namespace, self.dest, items)

def driver():
    parser = argparse.ArgumentParser(
        description = 'Legion test suite')

    # What tests to run:
    parser.add_argument(
        '--test', dest='test_modules', action=ExtendAction,
        choices=MultipleChoiceList('regent', 'legion_cxx', 'fuzzer',
                                   'realm', 'external1', 'external2',
                                   'private', 'perf', 'ctest', 'jupyter'),
        type=lambda s: s.split(','),
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
        '--max-dim', dest='max_dim', type=int,
        default=int(os.environ['MAX_DIM']) if 'MAX_DIM' in os.environ else 3,
        help='Maximum number of dimensions (also via MAX_DIM).')
    parser.add_argument(
        '--use', dest='use_features', action=ExtendAction,
        choices=MultipleChoiceList('gasnet', 'cuda', 'hip', 'openmp', 'kokkos',
                                   'python', 'llvm', 'hdf', 'fortran', 'spy', 'prof',
                                   'bounds', 'privilege', 'complex',
                                   'gcov', 'cmake', 'nvtx'),
        type=lambda s: s.split(','),
        default=None,
        help='Build Legion with features (also via USE_*).')
    parser.add_argument(
        '--network', dest='networks', action='store',
        default=os.environ.get('REALM_NETWORKS', 'gasnet1' if os.environ.get('USE_GASNET', '0') == '1' else ''),
        help='Network backend(s) to build with')
    parser.add_argument(
        '--launcher', dest='launcher', action='store',
        default=os.environ['LAUNCHER'] if 'LAUNCHER' in os.environ else None,
        help='Launcher for Legion tests (also via LAUNCHER).')

    parser.add_argument(
        '-C', '--directory', dest='root_dir', metavar='DIR', action='store', required=False,
        help='Legion root directory.')

    parser.add_argument(
        '--tmp-dir', dest='tmp_dir', metavar='DIR', action='store', required=False,
        help='Temporary directory path for out-of-source builds')

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
        '--timelimit', dest='timelimit', type=int,
        help='Maximum time (in seconds) allowed for individual test execution')

    parser.add_argument(
        '-v', '--verbose', dest='verbose', action='store_true',
        help='Print more debugging information.')

    args = parser.parse_args()

    run_tests(**vars(args))

if __name__ == '__main__':
    driver()
