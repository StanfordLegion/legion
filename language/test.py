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

import argparse, itertools, json, multiprocessing, os, optparse, re, subprocess, sys, traceback
from collections import OrderedDict
import legion

class TestFailure(Exception):
    def __init__(self, command, output):
        Exception.__init__(self, command, output)
        self.command = command
        self.output = output

def run(filename, verbose, flags):
    args = [os.path.basename(filename)] + flags + (
        [] if verbose else ['-level', '5'])
    proc = legion.legion(
        args,
        stdout = None if verbose else subprocess.PIPE,
        stderr = None if verbose else subprocess.STDOUT,
        cwd = os.path.dirname(os.path.abspath(filename)))
    output, _ = proc.communicate()
    retcode = proc.wait()
    if retcode != 0:
        raise TestFailure(' '.join(args), str(output))

_re_label = r'^[ \t\r]*--[ \t]+{label}:[ \t\r]*$\n((^[ \t\r]*--.*$\n)+)'
def find_labeled_prefix(filename, label):
    re_label = re.compile(_re_label.format(label = label), re.MULTILINE)
    with open(filename, 'rb') as f:
        program_text = f.read()
    match = re.search(re_label, program_text)
    if match is None:
        return None
    match_lines = match.group(1).strip().split('\n')
    match_text = '\n'.join([line.strip()[2:].strip() for line in match_lines])
    return match_text

def test_compile_fail(filename, verbose, flags):
    expected_failure = find_labeled_prefix(filename, 'fails-with')
    if expected_failure is None:
        raise Exception('No fails-with declaration in compile_fail test')

    try:
        run(filename, False, flags)
    except TestFailure as e:
        failure = '\n'.join(
            itertools.takewhile(
                (lambda line: line != 'stack traceback:'),
                itertools.dropwhile(
                    (lambda line: 'Errors reported during' in line),
                    (line.strip() for line in e.output.strip().split('\n')
                     if len(line.strip()) > 0))))
        if failure != expected_failure:
            raise Exception('Expected failure:\n%s\n\nInstead got:\n%s' % (expected_failure, failure))
    else:
        raise Exception('Expected failure, but test passed')

def test_run_pass(filename, verbose, flags):
    runs_with = [[]]
    runs_with_text = find_labeled_prefix(filename, 'runs-with')
    if runs_with_text is not None:
        runs_with = json.loads(runs_with_text)

    try:
        for params in runs_with:
            run(filename, verbose, flags + params)
    except TestFailure as e:
        raise Exception('Command failed:\n%s\n\nOutput:\n%s' % (e.command, e.output))

red = "\033[1;31m"
green = "\033[1;32m"
clear = "\033[0m"

PASS = 'pass'
FAIL = 'fail'
INTERRUPT = 'interrupt'

def test_runner(test_name, test_closure, verbose, filename):
    test_fn, test_args = test_closure
    saved_temps = []
    try:
        test_fn(filename, verbose, *test_args)
    except KeyboardInterrupt:
        return test_name, filename, [], INTERRUPT, None
    # except driver.CompilerException as e:
    #     if verbose:
    #         return test_name, filename, e.saved_temps, FAIL, ''.join(traceback.format_exception(*sys.exc_info()))
    #     return test_name, filename, e.saved_temps, FAIL, None
    except Exception as e:
        if verbose:
            return test_name, filename, [], FAIL, ''.join(traceback.format_exception_only(*sys.exc_info()[:2]))
        return test_name, filename, [], FAIL, ''.join(traceback.format_exception_only(*sys.exc_info()[:2]))
    else:
        return test_name, filename, [], PASS, None

class Counter:
    def __init__(self):
        self.passed = 0
        self.failed = 0

tests = [
    # FIXME: Move this flag into a per-test parameter so we don't use it everywhere.
    ('compile_fail', (test_compile_fail, (['-fbounds-checks', '1'],)),
     (os.path.join('tests', 'compile_fail'),)),
    ('run_pass', (test_run_pass, (['-fbounds-checks', '1'],)),
     (os.path.join('tests', 'run_pass'),
      os.path.join('examples'),
     )),
]

def run_all_tests(thread_count, verbose):
    thread_pool = multiprocessing.Pool(thread_count)
    results = []

    # Run tests asynchronously.
    for test_name, test_fn, test_dirs in tests:
        test_paths = []
        for test_dir in test_dirs:
            if os.path.isfile(test_dir):
                test_paths.append(test_dir)
            else:
                os.path.walk(
                    test_dir,
                    lambda args, dirname, names: test_paths.extend(
                        path
                        for name in sorted(names)
                        for path in [os.path.join(dirname, name)]
                        if os.path.isfile(path) and os.path.splitext(path)[1] in ('.lg', '.md')),
                    ())

        for test_path in test_paths:
            results.append(thread_pool.apply_async(test_runner, (test_name, test_fn, verbose, test_path)))

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
                print '[%sPASS%s] (%s) %s' % (green, clear, test_name, filename)
                if output is not None: print output
                test_counters[test_name].passed += 1
            elif outcome == FAIL:
                print '[%sFAIL%s] (%s) %s' % (red, clear, test_name, filename)
                if output is not None: print output
                test_counters[test_name].failed += 1
            else:
                raise Exception('Unexpected test outcome %s' % outcome)
    except KeyboardInterrupt:
        raise

    thread_pool.join()

    global_counter = Counter()
    for test_counter in test_counters.itervalues():
        global_counter.passed += test_counter.passed
        global_counter.failed += test_counter.failed
    global_total = global_counter.passed + global_counter.failed

    if len(all_saved_temps) > 0:
        print
        print 'The following temporary files have been saved:'
        print
        for test_name, filename, saved_temps in all_saved_temps:
            print '[%sFAIL%s] (%s) %s' % (red, clear, test_name, filename)
            for saved_temp in saved_temps:
                print '  %s' % saved_temp

    if global_total > 0:
        print
        print 'Summary of test results by category:'
        for test_name, test_counter in test_counters.iteritems():
            test_total = test_counter.passed + test_counter.failed
            if test_total > 0:
                print '%24s: Passed %3d of %3d tests (%5.1f%%)' % (
                    '%s' % test_name, test_counter.passed, test_total,
                    float(100*test_counter.passed)/test_total)
        print '    ' + '~'*54
        print '%24s: Passed %3d of %3d tests (%5.1f%%)' % (
            'total', global_counter.passed, global_total,
            (float(100*global_counter.passed)/global_total))

    if not verbose and global_counter.failed > 0:
        print
        print 'For detailed information on test failures, run:'
        print '    ./test.py -j1 -v'
        sys.exit(1)

def test_driver(argv):
    parser = argparse.ArgumentParser(description = 'Legion compiler test suite')
    parser.add_argument('-j',
                        nargs = '?',
                        type = int,
                        help = 'number threads used to compile',
                        dest = 'thread_count')
    parser.add_argument('-v',
                        action = 'store_true',
                        help = 'display verbose output',
                        dest = 'verbose')
    args = parser.parse_args(argv[1:])

    run_all_tests(
        args.thread_count,
        args.verbose)

if __name__ == '__main__':
    test_driver(sys.argv)
