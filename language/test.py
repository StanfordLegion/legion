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
import argparse, codecs, itertools, json, multiprocessing, os, optparse, re, subprocess, sys, tempfile, traceback
from collections import OrderedDict
import regent

_version = sys.version_info.major

if _version == 2: # Python 2.x:
    def glob(path):
        def visit(result, dirname, filenames):
            for filename in filenames:
                result.append(os.path.join(dirname, filename))
        result = []
        os.path.walk(path, visit, result)
        return result
elif _version == 3: # Python 3.x:
    def glob(path):
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
    output, _ = proc.communicate()
    retcode = proc.wait()
    if retcode != 0:
        raise TestFailure(' '.join(args), output.decode('utf-8') if output is not None else None)

def run_spy(logfile, verbose):
    cmd = ['pypy', os.path.join(regent.root_dir(), 'tools', 'legion_spy.py'),
           '--logical',
           '--physical',
           '--cycle',
           # '--sanity', # FIXME: This breaks on several test cases.
           '--leaks',
           # '--geometry', # FIXME: This is *very* slow.
           '--assert-error',
           '--assert-warning',
           logfile]
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

def find_labeled_flags(filename, prefix):
    flags = [[]]
    flags_text = find_labeled_text(filename, prefix)
    if flags_text is not None:
        flags = json.loads(flags_text)
        assert isinstance(flags, list), "%s declaration must be a json-formatted nested list" % prefix
        for flag in flags:
            assert isinstance(flag, list), "%s declaration must be a json-formatted nested list" % prefix
    return flags

def test_compile_fail(filename, debug, verbose, flags, env):
    expected_failure = find_labeled_text(filename, 'fails-with')
    if expected_failure is None:
        raise Exception('No fails-with declaration in compile_fail test')

    runs_with = find_labeled_flags(filename, 'runs-with')
    try:
        for params in runs_with:
            run(filename, debug, False, flags + params, env)
    except TestFailure as e:
        failure = e.output
        lines = set(line.strip() for line in failure.strip().split('\n')
                    if len(line.strip()) > 0)
        expected_lines = expected_failure.split('\n')
        for expected_line in expected_lines:
            if expected_line not in lines:
                raise Exception('Expected failure:\n%s\n\nInstead got:\n%s' % (expected_failure, failure))
    else:
        raise Exception('Expected failure, but test passed')

def test_run_pass(filename, debug, verbose, flags, env):
    runs_with = find_labeled_flags(filename, 'runs-with')
    try:
        for params in runs_with:
            run(filename, debug, verbose, flags + params, env)
    except TestFailure as e:
        raise Exception('Command failed:\n%s\n\nOutput:\n%s' % (e.command, e.output))

def test_spy(filename, debug, verbose, flags, env):
    spy_fd, spy_log = tempfile.mkstemp()
    os.close(spy_fd)
    spy_flags = ['-level', 'legion_spy=2', '-logfile', spy_log]

    runs_with = find_labeled_flags(filename, 'runs-with')
    try:
        for params in runs_with:
            run(filename, debug, verbose, flags + params + spy_flags, env)
            run_spy(spy_log, verbose)
    except TestFailure as e:
        raise Exception('Command failed:\n%s\n\nOutput:\n%s' % (e.command, e.output))
    finally:
        os.remove(spy_log)

red = "\033[1;31m"
green = "\033[1;32m"
clear = "\033[0m"

PASS = 'pass'
FAIL = 'fail'
INTERRUPT = 'interrupt'

def test_runner(test_name, test_closure, debug, verbose, filename):
    test_fn, test_args = test_closure
    saved_temps = []
    try:
        test_fn(filename, debug, verbose, *test_args)
    except KeyboardInterrupt:
        return test_name, filename, [], INTERRUPT, None
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


def get_test_specs(include_spy, extra_flags):
    base = [
        # FIXME: Move this flag into a per-test parameter so we don't use it everywhere.
        # Don't include backtraces on those expected to fail
        ('compile_fail', (test_compile_fail, (['-fbounds-checks', '1'] + extra_flags, {})),
         (os.path.join('tests', 'regent', 'compile_fail'),
          os.path.join('tests', 'bishop', 'compile_fail'),
         )),
        ('pretty', (test_run_pass, (['-fpretty', '1'] + extra_flags, {})),
         (os.path.join('tests', 'regent', 'run_pass'),
          os.path.join('tests', 'bishop', 'run_pass'),
          os.path.join('examples'),
          os.path.join('..', 'tutorial'),
         )),
        ('run_pass', (test_run_pass, ([] + extra_flags, {'REALM_BACKTRACE': '1'})),
         (os.path.join('tests', 'regent', 'run_pass'),
          os.path.join('tests', 'bishop', 'run_pass'),
          os.path.join('examples'),
          os.path.join('..', 'tutorial'),
          os.path.join('tests', 'runtime', 'bugs'),
         )),
    ]
    spy = [
        ('spy', (test_spy, ([] + extra_flags, {})),
         (os.path.join('tests', 'regent', 'run_pass'),
          os.path.join('tests', 'bishop', 'run_pass'),
          os.path.join('examples'),
          os.path.join('..', 'tutorial'),
         )),
    ]
    if include_spy:
        return spy
    else:
        return base

def run_all_tests(thread_count, debug, spy, extra_flags, verbose, quiet,
                  only_patterns, skip_patterns):
    thread_pool = multiprocessing.Pool(thread_count)
    results = []

    # Run tests asynchronously.
    tests = get_test_specs(spy, extra_flags)
    for test_name, test_fn, test_dirs in tests:
        test_paths = []
        for test_dir in test_dirs:
            if os.path.isfile(test_dir):
                test_paths.append(test_dir)
            else:
                test_paths.extend(
                    path for path in sorted(glob(test_dir))
                    if os.path.isfile(path) and os.path.splitext(path)[1] in ('.rg', '.md'))

        for test_path in test_paths:
            if only_patterns and not(any(re.search(p,test_path) for p in only_patterns)):
                continue
            if skip_patterns and any(re.search(p,test_path) for p in skip_patterns):
                continue
            results.append(thread_pool.apply_async(test_runner, (test_name, test_fn, debug, verbose, test_path)))

    thread_pool.close()

    test_counters = OrderedDict()
    for test_name, test_fn, test_dirs in tests:
        test_counter = Counter()
        test_counters[test_name] = test_counter

    all_saved_temps = []
    try:
        for result in results:
            test_name, filename, saved_temps, outcome, output = result.get()
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
                print('%24s: Passed %3d of %3d tests (%5.1f%%)' % (
                    '%s' % test_name, test_counter.passed, test_total,
                    float(100*test_counter.passed)/test_total))
        print('    ' + '~'*54)
        print('%24s: Passed %3d of %3d tests (%5.1f%%)' % (
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
    parser.add_argument('--spy', '-s',
                        action='store_true',
                        help='enable Legion Spy mode',
                        dest='spy')
    parser.add_argument('--extra',
                        action='append',
                        required=False,
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
    args = parser.parse_args(argv[1:])

    run_all_tests(
        args.thread_count,
        args.debug,
        args.spy,
        args.extra_flags,
        args.verbose,
        args.quiet,
        args.only_patterns,
        args.skip_patterns)

if __name__ == '__main__':
    test_driver(sys.argv)
