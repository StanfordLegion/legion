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
import argparse, codecs, glob, itertools, json, multiprocessing, os, optparse, re, shutil, subprocess, sys, tempfile, traceback, signal, time
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
    def __init__(self, command, output):
        Exception.__init__(self, command, output)
        self.command = command
        self.output = output

def run(filename, debug, verbose, flags, env):
    args = ((['-mg'] if debug else []) +
            [os.path.basename(filename)] + flags +
            ([] if verbose else ['-level', '5']))
    if verbose: print('Running', ' '.join(args))
    proc = regent.regent(
        args,
        stdout=None if verbose else subprocess.PIPE,
        stderr=None if verbose else subprocess.STDOUT,
        env=env,
        cwd=os.path.dirname(os.path.abspath(filename)))
    try:
        output, _ = proc.communicate()
        retcode = proc.wait()
    except (KeyboardInterrupt, TestTimeoutException):
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
        raise TestFailure(' '.join(args), output.decode('utf-8') if output is not None else None)
    return ' '.join(args)

def run_spy(logfiles, verbose):
    cmd = ['pypy', os.path.join(regent.root_dir(), 'tools', 'legion_spy.py'),
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
        raise TestFailure(' '.join(cmd), output.decode('utf-8') if output is not None else None)

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

def test_compile_fail(filename, debug, verbose, short, flags, env):
    expected_failure = find_labeled_text(filename, 'fails-with')
    if expected_failure is None:
        raise Exception('No fails-with declaration in compile_fail test')

    runs_with = find_labeled_flags(filename, 'runs-with', short)
    try:
        for params in runs_with:
            run(filename, debug, False, params + flags, env)
    except TestFailure as e:
        failure = e.output
        lines = set(line.strip() for line in failure.strip().split('\n')
                    if len(line.strip()) > 0)
        expected_lines = expected_failure.split('\n')
        for expected_line in expected_lines:
            if expected_line not in lines:
                raise Exception('Command failed:\n%s\n\nExpected failure:\n%s\n\nInstead got:\n%s' % (e.command, expected_failure, failure))
    else:
        raise Exception('Expected failure, but test passed')

def test_run_pass(filename, debug, verbose, short, flags, env):
    runs_with = find_labeled_flags(filename, 'runs-with', short)
    try:
        for params in runs_with:
            run(filename, debug, verbose, params + flags, env)
    except TestFailure as e:
        raise Exception('Command failed:\n%s\n\nOutput:\n%s' % (e.command, e.output))

def test_spy(filename, debug, verbose, short, flags, env):
    spy_dir = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(filename)))
    spy_log = os.path.join(spy_dir, 'spy_%.log')
    spy_flags = ['-level', 'legion_spy=2', '-logfile', spy_log]

    runs_with = find_labeled_flags(filename, 'runs-with', short)
    try:
        for params in runs_with:
            try:
                cmd = run(filename, debug, verbose, params + flags + spy_flags, env)
            except TestFailure as e:
                raise Exception('Command failed:\n%s\n\nOutput:\n%s' % (e.command, e.output))

            try:
                spy_logs = glob.glob(os.path.join(spy_dir, 'spy_*.log'))
                assert len(spy_logs) > 0
                run_spy(spy_logs, verbose)
            except TestFailure as e:
                raise Exception('Command failed:\n%s\n%s\n\nOutput:\n%s' % (cmd, e.command, e.output))
    finally:
        shutil.rmtree(spy_dir)

red = "\033[1;31m"
green = "\033[1;32m"
clear = "\033[0m"

PASS = 'pass'
FAIL = 'fail'
INTERRUPT = 'interrupt'
TIMEOUT = 'timeout'

class TestTimeoutException(Exception):
    pass

def sigalrm_handler(signum, frame):
    raise TestTimeoutException

def test_runner(test_name, test_closure, debug, verbose, filename, timelimit, short):
    test_fn, test_args = test_closure
    saved_temps = []
    if timelimit:
        signal.signal(signal.SIGALRM, sigalrm_handler)
        signal.alarm(timelimit)
    try:
        test_fn(filename, debug, verbose, short, *test_args)
        signal.alarm(0)
    except KeyboardInterrupt:
        return test_name, filename, [], INTERRUPT, None
    except TestTimeoutException:
        return test_name, filename, [], TIMEOUT, None
    # except driver.CompilerException as e:
    #     if verbose:
    #         return test_name, filename, e.saved_temps, FAIL, ''.join(traceback.format_exception(*sys.exc_info()))
    #     return test_name, filename, e.saved_temps, FAIL, None
    except Exception as e:
        if verbose:
            return test_name, filename, [], FAIL, ''.join(traceback.format_exception(*sys.exc_info()))
        return test_name, filename, [], FAIL, ''.join(traceback.format_exception_only(*sys.exc_info()[:2]))
    else:
        return test_name, filename, [], PASS, None

class Counter:
    def __init__(self):
        self.passed = 0
        self.failed = 0

def get_test_specs(use_run, use_spy, use_hdf5, use_openmp, short, extra_flags):
    base = [
        # FIXME: Move this flag into a per-test parameter so we don't use it everywhere.
        # Don't include backtraces on those expected to fail
        ('compile_fail', (test_compile_fail, (['-fbounds-checks', '1'] + extra_flags, {})),
         (os.path.join('tests', 'regent', 'compile_fail'),
          os.path.join('tests', 'bishop', 'compile_fail'),
         )),
    ]
    pretty = [
        ('pretty', (test_run_pass, (['-fpretty', '1'] + extra_flags, {})),
         (os.path.join('tests', 'regent', 'run_pass'),
          os.path.join('tests', 'regent', 'perf'),
          os.path.join('tests', 'regent', 'bugs'),
          os.path.join('tests', 'bishop', 'run_pass'),
          os.path.join('examples'),
          os.path.join('..', 'tutorial'),
         )),
    ]
    run = [
        ('run_pass', (test_run_pass, ([] + extra_flags, {'REALM_BACKTRACE': '1'})),
         (os.path.join('tests', 'regent', 'run_pass'),
          os.path.join('tests', 'regent', 'perf'),
          os.path.join('tests', 'regent', 'bugs'),
          os.path.join('tests', 'bishop', 'run_pass'),
          os.path.join('examples'),
          os.path.join('..', 'tutorial'),
          os.path.join('tests', 'runtime', 'bugs'),
         )),
    ]
    spy = [
        ('spy', (test_spy, ([] + extra_flags, {})),
         (os.path.join('tests', 'regent', 'run_pass'),
          os.path.join('tests', 'regent', 'perf'),
          os.path.join('tests', 'regent', 'bugs'),
          os.path.join('tests', 'bishop', 'run_pass'),
          os.path.join('examples'),
          os.path.join('..', 'tutorial'),
         )),
    ]
    hdf5 = [
        ('run_pass', (test_run_pass, ([] + extra_flags, {})),
         (os.path.join('tests', 'hdf5', 'run_pass'),
         )),
    ]
    openmp = [
        ('run_pass', (test_run_pass, ([] + extra_flags, {})),
         (os.path.join('tests', 'openmp', 'run_pass'),
         )),
    ]

    result = []
    if not use_run and not use_spy and not use_hdf5:
        result.extend(base)
        if not short:
            result.extend(pretty)
        result.extend(run)
    if use_run:
        result.extend(run)
    if use_spy:
        result.extend(spy)
    if use_hdf5:
        result.extend(hdf5)
    if use_openmp:
        result.extend(openmp)
    return result

def run_all_tests(thread_count, debug, run, spy, hdf5, openmp, extra_flags, verbose, quiet,
                  only_patterns, skip_patterns, timelimit, short):
    thread_pool = multiprocessing.Pool(thread_count)
    results = []

    # Run tests asynchronously.
    tests = get_test_specs(run, spy, hdf5, openmp, short, extra_flags)
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
            results.append((test_name, test_path, thread_pool.apply_async(test_runner, (test_name, test_fn, debug, verbose, test_path, timelimit, short))))

    thread_pool.close()

    test_counters = OrderedDict()
    for test_name, test_fn, test_dirs in tests:
        test_counter = Counter()
        test_counters[test_name] = test_counter

    all_saved_temps = []
    try:
        for test_name, filename, result in results:
            while True:
                try:
                    _test_name, _filename, saved_temps, outcome, output = result.get(timeout=60)
                    assert _test_name == test_name and _filename == filename
                except multiprocessing.TimeoutError:
                    print('Potential Hang: (%s) %s' % (test_name, filename))
                    sys.stdout.flush()
                else:
                    break
            if len(saved_temps) > 0:
                all_saved_temps.append((test_name, filename, saved_temps))
            if outcome == PASS:
                if quiet:
                    print('.', end='')
                    sys.stdout.flush()
                else:
                    print('[%sPASS%s] (%s) %s' % (green, clear, test_name, filename))
                if output is not None: print(output)
                test_counters[test_name].passed += 1
            elif outcome == FAIL:
                if quiet: print()
                print('[%sFAIL%s] (%s) %s' % (red, clear, test_name, filename))
                if output is not None: print(output)
                test_counters[test_name].failed += 1
            elif outcome == TIMEOUT:
                if quiet: print()
                print('[%sTIMEOUT%s] (%s) %s' % (red, clear, test_name, filename))
                if output is not None: print(output)
                test_counters[test_name].failed += 1
            else:
                raise Exception('Unexpected test outcome %s' % outcome)
    except KeyboardInterrupt:
        raise

    thread_pool.join()

    global_counter = Counter()
    for test_counter in test_counters.values():
        global_counter.passed += test_counter.passed
        global_counter.failed += test_counter.failed
    global_total = global_counter.passed + global_counter.failed

    if len(all_saved_temps) > 0:
        print()
        print('The following temporary files have been saved:')
        print()
        for test_name, filename, saved_temps in all_saved_temps:
            print('[%sFAIL%s] (%s) %s' % (red, clear, test_name, filename))
            for saved_temp in saved_temps:
                print('  %s' % saved_temp)

    if global_total > 0:
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
            'total', global_counter.passed, global_total,
            (float(100*global_counter.passed)/global_total)))

    if not verbose and global_counter.failed > 0:
        print()
        print('For detailed information on test failures, run:')
        print('    ./test.py -j1 -v')
        sys.exit(1)

def test_driver(argv):
    parser = argparse.ArgumentParser(description='Regent compiler test suite')
    parser.add_argument('-j',
                        nargs='?',
                        type=int,
                        help='number threads used to compile',
                        dest='thread_count')
    parser.add_argument('--debug', '-g',
                        action='store_true',
                        help='enable debug mode',
                        dest='debug')
    parser.add_argument('--run_pass',
                        action='store_true',
                        help='limit to run_pass tests',
                        dest='run_pass')
    parser.add_argument('--spy', '-s',
                        action='store_true',
                        help='run Legion Spy tests',
                        dest='spy')
    parser.add_argument('--hdf', '--hdf5',
                        action='store_true',
                        help='run HDF5 tests',
                        dest='hdf5')
    parser.add_argument('--openmp',
                        action='store_true',
                        help='run OpenMP tests',
                        dest='openmp')
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
                        default=600, # 10 minutes
                        type=int,
                        help='max run time for each test (in seconds)',
                        dest='timelimit')
    parser.add_argument('--short',
                        action='store_true',
                        help='truncate runs-with list of each test to one item',
                        dest='short')
    args = parser.parse_args(argv[1:])

    run_all_tests(
        args.thread_count,
        args.debug,
        args.run_pass,
        args.spy,
        args.hdf5,
        args.openmp,
        args.extra_flags,
        args.verbose,
        args.quiet,
        args.only_patterns,
        args.skip_patterns,
        args.timelimit,
        args.short)

if __name__ == '__main__':
    test_driver(sys.argv)
