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
import argparse, codecs, glob, itertools, json, multiprocessing, os, optparse, queue, re, shutil, signal, subprocess, sys, tempfile, traceback, time
from collections import OrderedDict
import regent

_version = sys.version_info.major

if _version == 2: # Python 2.x:
    def _glob(path):
        def visit(result, dirname, filenames):
            for filename in filenames:
                result.append(os.path.join(dirname, filename))
        result = []
        os.path.walk(path, visit, result)
        return result
elif _version == 3: # Python 3.x:
    def _glob(path):
        return [os.path.join(dirname, filename)
                for dirname, _, filenames in os.walk(path)
                for filename in filenames]
else:
    raise Exception('Incompatible Python version')

class TestFailure(Exception):
    def __init__(self, command, retcode, output):
        Exception.__init__(self, command, output)
        self.command = command
        self.retcode = retcode
        self.output = output

def detect_python_interpreter():
    try:
        if _version == 2 and subprocess.call(["pypy", "-h"], stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0:
            return "pypy"
        if _version == 3 and subprocess.call(["pypy3", "-h"], stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0:
            return "pypy3"
    except:
        pass
    return sys.executable

def run(filename, debug, verbose, timelimit, flags, env):
    args = ((['-g'] if debug else []) +
            [os.path.basename(filename)] + flags +
            ([] if verbose else ['-level', 'runtime=5,threads=5']))
    if verbose: print('Running', ' '.join(args))

    def set_timeout():
        def timeout_handler(signum, frame):
            raise TimeoutError("Timed-out for "+filename)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, timelimit)

    proc = regent.regent(
        args,
        stdout=None if verbose else subprocess.PIPE,
        stderr=None if verbose else subprocess.STDOUT,
        env=env,
        cwd=os.path.dirname(os.path.abspath(filename)),
        preexec_fn=set_timeout)
    try:
        output, _ = proc.communicate()
        retcode = proc.wait()
    except KeyboardInterrupt:
        if verbose: print('terminating child process...')
        proc.terminate()
        maxtime = 15
        while True:
            retcode = proc.poll()
            if retcode is not None:
                break
            if maxtime > 0:
                time.sleep(0.1)
                maxtime = maxtime - 0.1
            else:
                print('Child process failed to terminate - sending SIGKILL')
                proc.kill()
                break
        raise
    if retcode != 0:
        raise TestFailure(' '.join(args), retcode, output.decode('utf-8') if output is not None else None)
    return ' '.join(args)

def run_spy(logfiles, verbose, py_exe_path):
    cmd = [py_exe_path, os.path.join(regent.root_dir(), 'tools', 'legion_spy.py'),
           '--logical',
           '--physical',
           '--cycle',
           # '--sanity', # FIXME: This breaks on several test cases.
           '--leaks',
           # '--geometry', # FIXME: This is *very* slow.
           '--assert-error',
           '--assert-warning'] + logfiles
    if verbose: print('Running', ' '.join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=None if verbose else subprocess.PIPE,
        stderr=None if verbose else subprocess.STDOUT)
    output, _ = proc.communicate()
    retcode = proc.wait()
    if retcode != 0:
        raise TestFailure(' '.join(cmd), retcode, output.decode('utf-8') if output is not None else None)

def run_gc(logfiles, verbose, py_exe_path):
    cmd = [py_exe_path, os.path.join(regent.root_dir(), 'tools', 'legion_gc.py'),
           '--leaks',
           '--cycles',
           '--assert'] + logfiles
    if verbose: print('Running', ' '.join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=None if verbose else subprocess.PIPE,
        stderr=None if verbose else subprocess.STDOUT)
    output, _ = proc.communicate()
    retcode = proc.wait()
    if retcode != 0:
        raise TestFailure(' '.join(cmd), retcode, output.decode('utf-8') if output is not None else None)

def run_prof(out_dir, logfiles, verbose, py_exe_path):
    result_dir = os.path.join(out_dir, 'legion_prof')
    cmd = [
        py_exe_path,
        os.path.join(regent.root_dir(), 'tools', 'legion_prof.py'),
        '-o', result_dir,
    ] + logfiles
    if verbose: print('Running', ' '.join(cmd))
    cmd_env = dict(os.environ.items())
    cmd_env["USE_TYPE_CHECK"] = "1"
    proc = subprocess.Popen(
        cmd,
        stdout=None if verbose else subprocess.PIPE,
        stderr=None if verbose else subprocess.STDOUT,
        env=cmd_env)
    output, _ = proc.communicate()
    retcode = proc.wait()
    if retcode != 0:
        raise TestFailure(' '.join(cmd), retcode, output.decode('utf-8') if output is not None else None)
    return result_dir

def run_prof_rs(out_dir, logfiles, verbose, legion_prof_rs):
    result_dir = os.path.join(out_dir, 'legion_prof_rs')
    cmd = [legion_prof_rs, '-o', result_dir,] + logfiles
    if verbose: print('Running', ' '.join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=None if verbose else subprocess.PIPE,
        stderr=None if verbose else subprocess.STDOUT)
    output, _ = proc.communicate()
    retcode = proc.wait()
    if retcode != 0:
        raise TestFailure(' '.join(cmd), retcode, output.decode('utf-8') if output is not None else None)
    return result_dir

def run_prof_rs_archive(out_dir, logfiles, verbose, legion_prof_rs):
    result_dir = os.path.join(out_dir, 'legion_prof_rs_archive')
    cmd = [legion_prof_rs, '--archive', '--levels', '3', '--zstd-compression', '1', '-o', result_dir,] + logfiles
    if verbose: print('Running', ' '.join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=None if verbose else subprocess.PIPE,
        stderr=None if verbose else subprocess.STDOUT)
    output, _ = proc.communicate()
    retcode = proc.wait()
    if retcode != 0:
        raise TestFailure(' '.join(cmd), retcode, output.decode('utf-8') if output is not None else None)
    return result_dir

def run_prof_subnode(out_dir, logfiles, verbose, subnodes, py_exe_path):
    result_dir = os.path.join(out_dir, 'legion_prof_filter_input')
    cmd = [
        py_exe_path,
        os.path.join(regent.root_dir(), 'tools', 'legion_prof_verify_subnodes.py'),
        '--outdir', out_dir,
        '--nodes', str(subnodes),
    ] + logfiles
    if verbose: 
        print('Running', ' '.join(cmd))
    cmd_env = dict(os.environ.items())
    cmd_env["USE_TYPE_CHECK"] = "1"
    proc = subprocess.Popen(
        cmd,
        stdout=None if verbose else subprocess.PIPE,
        stderr=None if verbose else subprocess.STDOUT,
        env=cmd_env)
    output, _ = proc.communicate()
    retcode = proc.wait()
    if retcode != 0:
        raise TestFailure(' '.join(cmd), retcode, output.decode('utf-8') if output is not None else None)
    return result_dir

def run_prof_rs_subnode(out_dir, logfiles, verbose, subnodes, py_exe_path, legion_prof_rs):
    result_dir = os.path.join(out_dir, 'legion_prof_filter_input_rs')
    cmd = [
        py_exe_path,
        os.path.join(regent.root_dir(), 'tools', 'legion_prof_verify_subnodes.py'),
        '--outdir', out_dir,
        '--nodes', str(subnodes),
        '--rust',
        '--rustexe', legion_prof_rs,
    ] +  logfiles
    if verbose:
        print('Running rs', ' '.join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=None if verbose else subprocess.PIPE,
        stderr=None if verbose else subprocess.STDOUT)
    output, _ = proc.communicate()
    retcode = proc.wait()
    if retcode != 0:
        raise TestFailure(' '.join(cmd), retcode, output.decode('utf-8') if output is not None else None)
    return result_dir

def compare_prof_results(verbose, py_exe_path, profile_dirs):
    cmd = ['diff', '-r', '-u',
           '--exclude', 'critical_path.json',
           ] + profile_dirs
    if verbose: print('Running', ' '.join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=None if verbose else subprocess.PIPE,
        stderr=None if verbose else subprocess.STDOUT)
    output, _ = proc.communicate()
    retcode = proc.wait()
    if retcode != 0:
        raise TestFailure(' '.join(cmd), retcode, output.decode('utf-8') if output is not None else None)

_re_label = r'^[ \t\r]*--[ \t]+{label}:[ \t\r]*$\n((^[ \t\r]*--.*$\n)+)'
def find_labeled_text(filename, label):
    re_label = re.compile(_re_label.format(label=label), re.MULTILINE)
    with codecs.open(filename, 'r', encoding='utf-8') as f:
        program_text = f.read()
    match = re.search(re_label, program_text)
    if match is None:
        return None
    match_lines = match.group(1).strip().split('\n')
    match_text = '\n'.join([line.strip()[2:].strip() for line in match_lines])
    return match_text

def find_labeled_flags(filename, prefix, short):
    flags = [[]]
    flags_text = find_labeled_text(filename, prefix)
    if flags_text is not None:
        flags = json.loads(flags_text)
        assert isinstance(flags, list), "%s declaration must be a json-formatted nested list" % prefix
        for flag in flags:
            assert isinstance(flag, list), "%s declaration must be a json-formatted nested list" % prefix
    if short:
        return flags[:1]
    return flags

def test_compile_fail(filename, debug, verbose, out_dir, short, timelimit, py_exe_path, legion_prof_rs, flags, env):
    expected_failure = find_labeled_text(filename, 'fails-with')
    if expected_failure is None:
        raise Exception('No fails-with declaration in compile_fail test')

    runs_with = find_labeled_flags(filename, 'runs-with', short)
    try:
        for params in runs_with:
            last_cmd = run(filename, debug, False, timelimit, params + flags, env)
    except TestFailure as e:
        failure = e.output
        lines = set(line.strip() for line in failure.strip().split('\n')
                    if len(line.strip()) > 0)
        expected_lines = expected_failure.split('\n')
        for expected_line in expected_lines:
            if expected_line not in lines:
                raise TestFailure(e.command, 1, 'Output mismatch - expected failure:\n%s\n\nInstead got:\n%s' % (expected_failure, failure))
    else:
        raise TestFailure(last_cmd, 1, 'Expected failure, but test ran successfully!')

def test_run_pass(filename, debug, verbose, out_dir, short, timelimit, py_exe_path, legion_prof_rs, flags, env):
    runs_with = find_labeled_flags(filename, 'runs-with', short)
    for params in runs_with:
        run(filename, debug, verbose, timelimit, params + flags, env)

def test_spy(filename, debug, verbose, out_dir, short, timelimit, py_exe_path, legion_prof_rs, flags, env):
    spy_dir = tempfile.mkdtemp(dir=out_dir, prefix='test_spy_%s_' % (os.path.splitext(os.path.basename(filename))[0]))
    spy_log = os.path.join(spy_dir, 'spy_%.log')
    spy_flags = ['-lg:spy', '-level', 'legion_spy=2', '-logfile', spy_log]

    runs_with = find_labeled_flags(filename, 'runs-with', short)
    try:
        for params in runs_with:
            run(filename, debug, verbose, timelimit, params + flags + spy_flags, env)

            spy_logs = glob.glob(os.path.join(spy_dir, 'spy_*.log'))
            assert len(spy_logs) > 0
            run_spy(spy_logs, verbose, py_exe_path)
            # Run legion_prof_rs too so that we can be sure it's at least parsing all the logs
            if legion_prof_rs is not None:
                run_prof_rs(spy_dir, spy_logs, verbose, legion_prof_rs)
    except:
        raise
    else:
        shutil.rmtree(spy_dir)

def test_gc(filename, debug, verbose, out_dir, short, timelimit, py_exe_path, legion_prof_rs, flags, env):
    gc_dir = tempfile.mkdtemp(dir=out_dir, prefix='test_gc_%s_' % (os.path.splitext(os.path.basename(filename))[0]))
    gc_log = os.path.join(gc_dir, 'gc_%.log')
    gc_flags = ['-level', 'legion_gc=2', '-logfile', gc_log]

    runs_with = find_labeled_flags(filename, 'runs-with', short)
    try:
        for params in runs_with:
            run(filename, debug, verbose, timelimit, params + flags + gc_flags, env)

            gc_logs = glob.glob(os.path.join(gc_dir, 'gc_*.log'))
            assert len(gc_logs) > 0
            run_gc(gc_logs, verbose, py_exe_path)
    except:
        raise
    else:
        shutil.rmtree(gc_dir)

def test_prof(filename, debug, verbose, out_dir, short, timelimit, py_exe_path, legion_prof_rs, flags, env):
    if legion_prof_rs is None:
        raise Exception('Need to specify the path to legion_prof_rs via --legion-prof-rs')

    prof_dir = tempfile.mkdtemp(dir=out_dir, prefix='test_prof_%s_' % (os.path.splitext(os.path.basename(filename))[0]))
    prof_log = os.path.join(prof_dir, 'prof_%.gz')
    prof_flags = ['-hl:prof', '1024', '-hl:prof_logfile', prof_log]

    runs_with = find_labeled_flags(filename, 'runs-with', short)
    try:
        for params in runs_with:
            run(filename, debug, verbose, timelimit, params + flags + prof_flags, env)

            prof_logs = glob.glob(os.path.join(prof_dir, 'prof_*.gz'))
            assert len(prof_logs) > 0
            result_py = run_prof(prof_dir, prof_logs, verbose, py_exe_path)
            result_rs = run_prof_rs(prof_dir, prof_logs, verbose, legion_prof_rs)
            compare_prof_results(verbose, py_exe_path, [result_py, result_rs])
            run_prof_rs_archive(prof_dir, prof_logs, verbose, legion_prof_rs)
            # we only test subnodes when running on multi-node
            if os.environ.get('LAUNCHER'):
                result_subnodes_py = run_prof_subnode(prof_dir, prof_logs, verbose, 1, py_exe_path)
                result_subnodes_rs = run_prof_rs_subnode(prof_dir, prof_logs, verbose, 1, py_exe_path, legion_prof_rs)
                compare_prof_results(verbose, py_exe_path, [result_subnodes_py, result_subnodes_rs])
    except:
        raise
    else:
        shutil.rmtree(prof_dir)

red = "\033[1;31m"
green = "\033[1;32m"
clear = "\033[0m"

PASS = 'pass'
FAIL = 'fail'
INTERRUPT = 'interrupt'
INTERNALERROR = 'internalerror'

def test_runner(test_name, test_closure, debug, verbose, out_dir, filename, timelimit, short, py_exe_path, legion_prof_rs):
    test_fn, test_args = test_closure
    try:
        test_fn(filename, debug, verbose, out_dir, short, timelimit, py_exe_path, legion_prof_rs, *test_args)
    except KeyboardInterrupt:
        return test_name, filename, None, INTERRUPT, 0, None
    except TestFailure as e:
        return test_name, filename, e.command, FAIL, e.retcode, e.output
    except Exception as e:
        if verbose:
            return test_name, filename, None, INTERNALERROR, 0, ''.join(traceback.format_exception(*sys.exc_info()))
        return test_name, filename, None, INTERNALERROR, 0, ''.join(traceback.format_exception_only(*sys.exc_info()[:2]))
    else:
        return test_name, filename, None, PASS, 0, None

class Counter:
    def __init__(self):
        self.passed = 0
        self.failed = 0

def get_test_specs(legion_dir, use_run, use_spy, use_gc, use_prof, use_hdf5, use_openmp, use_gpu, use_python, max_dim, no_pretty, extra_flags):
    base_env = {
    }
    run_env = {
        'REALM_BACKTRACE': '1',
    }

    unit_tests = (
        os.path.join('tests', 'regent', 'unit_test'),
    )
    run_pass_tests = (
        os.path.join('tests', 'regent', 'run_pass'),
        os.path.join('tests', 'regent', 'perf'),
        os.path.join('tests', 'regent', 'bugs'),
        os.path.join('tests', 'regent', 'layout'),
        os.path.join('tests', 'bishop', 'run_pass'),
        os.path.join('examples'),
        os.path.join('..', 'tutorial'),
        os.path.join('tests', 'runtime', 'bugs'),
        os.path.join('tests', 'runtime', 'features'),
    )

    base = [
        # FIXME: Move this flag into a per-test parameter so we don't use it everywhere.
        # Don't include backtraces on those expected to fail
        ('compile_fail', (test_compile_fail, (['-fbounds-checks', '1'] + extra_flags, base_env)),
         (os.path.join('tests', 'regent', 'compile_fail'),
          os.path.join('tests', 'bishop', 'compile_fail'),
         )),
    ]
    pretty = [
        ('pretty', (test_run_pass, (['-fpretty', '1'] + extra_flags, base_env)),
         unit_tests + run_pass_tests),
    ]
    run = [
        ('run_pass', (test_run_pass, ([] + extra_flags, run_env)),
         unit_tests + run_pass_tests),
    ]
    spy = [
        ('spy', (test_spy, ([] + extra_flags, base_env)),
         run_pass_tests),
    ]
    gc = [
        ('gc', (test_gc, ([] + extra_flags, base_env)),
         run_pass_tests),
    ]
    prof = [
        ('prof', (test_prof, ([] + extra_flags, base_env)),
         run_pass_tests),
    ]
    hdf5 = [
        ('run_pass', (test_run_pass, ([] + extra_flags, run_env)),
         (os.path.join('tests', 'hdf5', 'run_pass'),
         )),
    ]
    openmp = [
        ('compile_fail', (test_compile_fail, (['-fbounds-checks', '1'] + extra_flags, base_env)),
         (os.path.join('tests', 'openmp', 'compile_fail'),
         )),
        ('run_pass', (test_run_pass, ([] + extra_flags, run_env)),
         (os.path.join('tests', 'openmp', 'run_pass'),
         )),
    ]
    gpu = [
        ('compile_fail', (test_compile_fail, (['-fbounds-checks', '1'] + extra_flags, base_env)),
         (os.path.join('tests', 'cuda', 'compile_fail'),
         )),
        ('run_pass', (test_run_pass, (['-fgpu', use_gpu] + extra_flags, run_env)),
         (os.path.join('tests', 'cuda', 'run_pass'),
          os.path.join('tests', 'cuda', 'examples'),
         )),
    ]
    python = [
        ('run_pass', (test_run_pass, ([] + extra_flags, run_env)),
         (os.path.join('tests', 'python', 'run_pass'),
          os.path.join('..', 'apps'),
         )),
    ]
    def max_dim_tests(dim):
        return [
            ('run_pass', (test_run_pass, ([] + extra_flags, run_env)),
             (os.path.join('tests', 'maxdim%s' % dim, 'run_pass'),
             )),
        ]

    result = []
    if not (use_run or use_spy or use_gc or use_prof or use_hdf5 or use_gpu):
        result.extend(base)
        if not no_pretty:
            result.extend(pretty)
        result.extend(run)
    if use_run:
        result.extend(run)
    if use_spy:
        result.extend(spy)
    if use_gc:
        result.extend(gc)
    if use_prof:
        result.extend(prof)
    if use_hdf5:
        result.extend(hdf5)
    if use_openmp:
        result.extend(openmp)
    if use_gpu:
        result.extend(gpu)
    if use_python:
        result.extend(python)
    for dim in range(4, min(max_dim, 8) + 1):
        result.extend(max_dim_tests(dim))
    return result

def run_all_tests(thread_count, debug, out_dir, max_dim, run, spy, gc, prof, hdf5,
                  openmp, gpu, python, extra_flags, verbose, quiet,
                  only_patterns, skip_patterns, timelimit, poll_interval,
                  short, no_pretty, legion_prof_rs):
    # run only one test at a time if '-j' isn't set
    if not thread_count:
        thread_count = 1
    thread_pool = multiprocessing.Pool(thread_count)

    result_queue = queue.Queue()
    num_queued = 0

    legion_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    py_exe_path = detect_python_interpreter()

    if spy or gc or prof:
        if not out_dir:
            out_dir = tempfile.mkdtemp(dir=os.getcwd(), prefix='test_output_')
        else:
            out_dir = os.abspath(out_dir)
            # The user specifically asked for this directory so we
            # just fail if we can't create it
            os.mkdir(out_dir)
    else:
        if out_dir is not None:
            print('Ignoring --output flag because no tests that use output are enabled')
        out_dir = None

    # Run tests asynchronously.
    tests = get_test_specs(legion_dir, run, spy, gc, prof, hdf5, openmp, gpu, python, max_dim, no_pretty, extra_flags)
    for test_name, test_fn, test_dirs in tests:
        test_paths = []
        for test_dir in test_dirs:
            if os.path.isfile(test_dir):
                test_paths.append(test_dir)
            else:
                test_paths.extend(
                    path for path in sorted(_glob(test_dir))
                    if os.path.isfile(path) and os.path.splitext(path)[1] in ('.rg',))

        for test_path in test_paths:
            if only_patterns and not(any(re.search(p,test_path) for p in only_patterns)):
                continue
            if skip_patterns and any(re.search(p,test_path) for p in skip_patterns):
                continue
            num_queued += 1
            def callback(r):
                result_queue.put(r)
            def error_callback(e):
                print('ERROR CALLBACK', e)
                raise e
            thread_pool.apply_async(test_runner,
                                    (test_name, test_fn, debug, verbose, out_dir,
                                     test_path, timelimit, short, py_exe_path,
                                     legion_prof_rs),
                                    callback=callback,
                                    error_callback=error_callback)

    thread_pool.close()

    test_counters = OrderedDict()
    for test_name, test_fn, test_dirs in tests:
        test_counter = Counter()
        test_counters[test_name] = test_counter

    num_passed = 0
    num_failed = 0
    num_remaining = num_queued
    t_start = time.time()
    t_last = t_start
    try:
        while num_remaining:
            # wait for up to 'interval' seconds for something to finish
            try:
                test_name, filename, command, outcome, retcode, output = result_queue.get(timeout=poll_interval)
                num_remaining -= 1
                if outcome == PASS:
                    if not quiet:
                        print('[%sPASS%s] (%s) %s' % (green, clear, test_name, filename))
                        if verbose and command is not None:
                            print(command)
                    if output is not None: print(output)
                    test_counters[test_name].passed += 1
                    num_passed += 1
                elif outcome == FAIL:
                    if retcode < 0:
                        try:
                            errmsg = 'signal ' + signal.Signals(-retcode).name
                        except ValueError:
                            errmsg = 'signal %d' % (-retcode)
                    else:
                        errmsg = 'return code %d' % retcode
                    print('[%sFAIL%s] (%s) (%s) %s' % (red, clear, test_name, errmsg, filename))
                    if command is not None:
                        print(command)
                    if output is not None: print(output)
                    test_counters[test_name].failed += 1
                    num_failed += 1
                elif outcome == INTERNALERROR:
                    print('[%sERROR%s] (%s) %s' % (red, clear, test_name, filename))
                    if command is not None:
                        print(command)
                    if output is not None: print(output)
                    test_counters[test_name].failed += 1
                    num_failed += 1
                else:
                    raise Exception('Unexpected test outcome %s' % outcome)
            except queue.Empty:
                pass

            # if requested, print out updates periodically
            if poll_interval:
                t_now = time.time()
                if (t_now >= t_last + poll_interval) or (num_remaining == 0):
                    print('Time elapsed: %4d s, tests passed: %4d, failed: %4d, remaining: %4d' % (int(t_now - t_start), num_passed, num_failed, num_remaining), flush=True)
                    t_last = t_now

        thread_pool.join()
    except KeyboardInterrupt:
        # try to terminate all child tests before propagating interrupt
        print('Keyboard interrupt received - terminating %d tests' % num_remaining)
        thread_pool.terminate()
        num_failed += num_remaining
        raise

    if num_queued > 0:
        print()
        print('Summary of test results by category:')
        for test_name, test_counter in test_counters.items():
            test_total = test_counter.passed + test_counter.failed
            if test_total > 0:
                print('%24s: Passed %4d of %4d tests (%5.1f%%)' % (
                    '%s' % test_name, test_counter.passed, test_total,
                    float(100*test_counter.passed)/test_total))
        print('    ' + '~'*54)
        print('%24s: Passed %4d of %4d tests (%5.1f%%)' % (
            'total', num_passed, num_queued,
            (float(100*num_passed)/num_queued)))

    if num_failed > 0:
        if out_dir:
            print()
            print('Output (e.g., logs) from failed tests has been kept in:')
            print('    %s' % out_dir)
        if not verbose:
            print()
            print('For detailed information on test failures, run:')
            print('    ./test.py -j1 -v')
        sys.exit(1)
    else:
        if out_dir:
            shutil.rmtree(out_dir)

def test_driver(argv):
    parser = argparse.ArgumentParser(description='Regent compiler test suite')
    parser.add_argument('-j',
                        nargs='?',
                        type=int,
                        help='number threads used to compile',
                        dest='thread_count')
    parser.add_argument('--debug', '-g',
                        action='store_true',
                        help='enable debug mode')
    parser.add_argument('--output', '-o',
                        dest='out_dir',
                        help='directory to be used for test output')
    parser.add_argument('--max-dim',
                        type=int,
                        default=3,
                        help='maximum number of dimensions',
                        dest='max_dim')
    parser.add_argument('--run_pass',
                        action='store_true',
                        help='limit to run_pass tests')
    parser.add_argument('--spy', '-s',
                        action='store_true',
                        help='run Legion Spy tests')
    parser.add_argument('--gc',
                        action='store_true',
                        help='run Legion GC tests')
    parser.add_argument('--prof',
                        action='store_true',
                        help='run Legion Prof tests')
    parser.add_argument('--hdf5',
                        action='store_true',
                        help='run HDF5 tests')
    parser.add_argument('--openmp',
                        action='store_true',
                        help='run OpenMP tests')
    parser.add_argument('--gpu',
                        choices=['cuda', 'hip'],
                        help='run GPU tests')
    parser.add_argument('--python',
                        action='store_true',
                        help='run Python tests')
    parser.add_argument('--extra',
                        action='append',
                        default=[],
                        help='extra flags to use for each test',
                        dest='extra_flags')
    parser.add_argument('-v',
                        action='store_true',
                        help='display verbose output',
                        dest='verbose')
    parser.add_argument('-q',
                        action='store_true',
                        help='suppress passing test results',
                        dest='quiet')
    parser.add_argument('--only',
                        action='append',
                        default=[],
                        help='only run tests matching pattern',
                        dest='only_patterns')
    parser.add_argument('--skip',
                        action='append',
                        default=[],
                        help='skip tests matching pattern',
                        dest='skip_patterns')
    parser.add_argument('--limit',
                        default=900, # 15 minutes
                        type=int,
                        help='max run time for each test (in seconds)',
                        dest='timelimit')
    parser.add_argument('--poll',
                        default=60, # 1 minute
                        type=int,
                        help='test completion polling interval (in seconds)',
                        dest='poll_interval')
    parser.add_argument('--short',
                        action='store_true',
                        help='truncate runs-with list of each test to one item',
                        dest='short')
    parser.add_argument('--no-pretty',
                        action='store_true',
                        help='disable pretty-printing tests',
                        dest='no_pretty')
    parser.add_argument('--legion-prof-rs',
                        help='location of Legion Prof Rust binary')
    args = parser.parse_args(argv[1:])

    run_all_tests(
        args.thread_count,
        args.debug,
        args.out_dir,
        args.max_dim,
        args.run_pass,
        args.spy,
        args.gc,
        args.prof,
        args.hdf5,
        args.openmp,
        args.gpu,
        args.python,
        args.extra_flags,
        args.verbose,
        args.quiet,
        args.only_patterns,
        args.skip_patterns,
        args.timelimit,
        args.poll_interval,
        args.short,
        args.no_pretty,
        args.legion_prof_rs)

if __name__ == '__main__':
    test_driver(sys.argv)
