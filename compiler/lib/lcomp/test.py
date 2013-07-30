#!/usr/bin/env python

# Copyright 2013 Stanford University and Los Alamos National Security, LLC
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

###
### Test suite
###

import json, multiprocessing, optparse, os, re, subprocess, sys, tempfile, traceback
from cStringIO import StringIO
# Work around for OrderedDict missing in Python 2.6.
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
from . import ast, doc_check, driver, passes, pretty

red = "\033[1;31m"
green = "\033[1;32m"
clear = "\033[0m"

def precompile_runtime(thread_count, clean_first, verbose):
    if 'LG_RT_DIR' not in os.environ:
        return

    (fd_source, temp_source) = tempfile.mkstemp(suffix = '.lg')
    (fd_binary, temp_binary) = tempfile.mkstemp()
    try:
        os.write(fd_source, 'task main() {}')
        os.close(fd_source)
        os.close(fd_binary)
        driver.driver(
            ['./driver.py', temp_source, '-o', temp_binary] +
            (['-j', str(thread_count)] if thread_count is not None else []) +
            (['--clean'] if clean_first else []),
            verbose = verbose)
    finally:
        os.remove(temp_source)
        os.remove(temp_binary)

def test_pretty(filename, verbose):
    with open(filename, 'rb') as f:
        original = f.read()
    with open(filename, 'rb') as f:
        program1 = passes.parse(f)
        pretty1 = pretty.pretty_print(program1)
        program2 = passes.parse(StringIO(pretty1))
        pretty2 = pretty.pretty_print(program2)
        if pretty1 != pretty2:
            raise Exception('Programs differ:\n\n%s\n\n%s' % (pretty1, pretty2))
        if not ast.equivalent(program1, program2):
            raise Exception('Programs differ:\n\n%s\n\n%s' % (original, pretty1))

def test_compile_pass(filename, verbose):
    env_needs_cleanup = False
    if 'LG_RT_DIR' not in os.environ:
        os.environ['LG_RT_DIR'] = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            '..', '..', 'runtime')
        env_needs_cleanup = True

    if verbose:
        stdout, stderr = None, None
    else:
        stdout, stderr = subprocess.PIPE, subprocess.STDOUT

    (fd, temp_obj) = tempfile.mkstemp(suffix = '.o')
    try:
        os.close(fd)
        driver.driver(
            ['./driver.py', '-c', filename, '-o', temp_obj],
            verbose = verbose)
    finally:
        os.remove(temp_obj)
        if env_needs_cleanup:
            del os.environ['LG_RT_DIR']

_re_expected_failure = re.compile(r'^//\s+fails-with:$\s*((^//.*$\s*)+)', re.MULTILINE)
def test_compile_fail(filename, verbose):
    # Search for the expected failure in the program
    with open(filename, 'rb') as f:
        program_text = f.read()
    expected_failure_match = re.search(_re_expected_failure, program_text)
    if expected_failure_match is None:
        raise Exception('No fails-with declaration in compile_fail testcase')
    expected_failure_lines = expected_failure_match.group(1).strip().split('\n')
    expected_failure = '\n'.join([line[2:].strip() for line in expected_failure_lines])

    # Check that the program fails to compile
    try:
        with open(filename, 'rb') as f:
            program = passes.parse(f)
        search_path = (os.path.abspath(os.path.dirname(filename)),)
        type_map, constraints = passes.check(program, search_path)
        program_type = type_map[program]
    except KeyboardInterrupt:
        raise
    except:
        failure_lines = traceback.format_exception_only(*sys.exc_info()[:2])
        if len(failure_lines) != 1:
            raise Exception('Expecting failure with one line, got:\n%s' % ''.join(failure_lines))
        failure = '\n'.join(line.strip() for line in failure_lines[0].strip().split('\n'))
        if failure != expected_failure:
            raise Exception('Expecting failure:\n%s\nInstead got:\n%s' % (expected_failure, failure))
    else:
        raise Exception('Expecting failure, type checker returned successfully:\n%s' % program_type)

def test_run_pass(filename, verbose):
    if 'LG_RT_DIR' not in os.environ:
        raise Exception('LG_RT_DIR variable is not defined')

    if verbose:
        stdout, stderr = None, None
    else:
        stdout, stderr = subprocess.PIPE, subprocess.STDOUT

    input_filenames = [filename]
    cpp_filename = ''.join([os.path.splitext(filename)[0], '.cc'])
    if os.path.exists(cpp_filename):
        input_filenames.append(cpp_filename)

    (fd, temp_binary) = tempfile.mkstemp()
    try:
        os.close(fd)
        driver.driver(
            ['./driver.py'] + input_filenames + ['-o', temp_binary, '-j', '1'],
            verbose = verbose)
        proc = subprocess.Popen(
            [temp_binary],
            stdout = stdout, stderr = stderr)
        output, errors = proc.communicate()
        if proc.returncode != 0:
            raise Exception(output, errors)
    finally:
        os.remove(temp_binary)

def test_doc(filename, verbose):
    with open(filename) as source:
        search_path = (os.path.abspath(os.path.dirname(filename)),)
        doc_check.doc_check(source, search_path)

PASS = 'pass'
FAIL = 'fail'
INTERRUPT = 'interrupt'

def test_runner(test_name, test_fn, verbose, filename):
    try:
        test_fn(filename, verbose)
    except KeyboardInterrupt:
        return test_name, filename, INTERRUPT, None
    except Exception as e:
        if verbose:
            return test_name, filename, FAIL, ''.join(traceback.format_exception(*sys.exc_info()))
        return test_name, filename, FAIL, None
    else:
        return test_name, filename, PASS, None

class Counter:
    def __init__(self):
        self.passed = 0
        self.failed = 0

tests = [
    ('compile_fail', test_compile_fail, (os.path.join('tests', 'compile_fail'),)),
    ('compile_pass', test_compile_pass, (os.path.join('tests', 'compile_pass'),
                                         os.path.join('tests', 'run_pass'),
                                         os.path.join('examples'))),
    ('pretty',       test_pretty,       (os.path.join('tests', 'compile_pass'),
                                         os.path.join('tests', 'run_pass'),
                                         os.path.join('examples'))),
    ('doc',          test_doc,          ('README.md', 'doc')),
    ('run_pass',     test_run_pass,     (os.path.join('tests', 'run_pass'),
                                         os.path.join('examples'))),
    ]

def run_all_tests(thread_count, clean_first, verbose):
    thread_pool = multiprocessing.Pool(thread_count)
    results = []

    # Precompile the runtime so threads don't thrash the filesystem.
    print 'Compiling Legion runtime'
    precompile_runtime(thread_count, clean_first, verbose)

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
                        for name in names
                        for path in [os.path.join(dirname, name)]
                        if os.path.isfile(path) and os.path.splitext(path)[1] in ('.lg', '.md')),
                    ())

        for test_path in test_paths:
            results.append(thread_pool.apply_async(test_runner, (test_name, test_fn, verbose, test_path)))

    test_counters = OrderedDict()
    for test_name, test_fn, test_dirs in tests:
        test_counter = Counter()
        test_counters[test_name] = test_counter

    try:
        for result in results:
            test_name, filename, outcome, output = result.get()
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

    global_counter = Counter()
    for test_counter in test_counters.itervalues():
        global_counter.passed += test_counter.passed
        global_counter.failed += test_counter.failed

    global_total = global_counter.passed + global_counter.failed
    if global_total > 0:
        print
        print 'Summary of test results by category:'
        for test_name, test_counter in test_counters.iteritems():
            test_total = test_counter.passed + test_counter.failed
            if test_total > 0:
                print '%16s: Passed %3d of %3d tests (%5.1f%%)' % (
                    '%s' % test_name, test_counter.passed, test_total,
                    float(100*test_counter.passed)/test_total)
        print '    ' + '~'*46
        print '%16s: Passed %3d of %3d tests (%5.1f%%)' % (
            'total', global_counter.passed, global_total,
            (float(100*global_counter.passed)/global_total))

        if not verbose and global_counter.failed > 0:
            print
            print 'For detailed information on test failures, run:'
            print '    ./test.py -j1 -v'

def test_driver(argv):
    parser = optparse.OptionParser(description = 'Compiles a Legion source file.')
    parser.add_option('-v', dest = 'verbose', action = 'store_true', default = False)
    parser.add_option('-j', dest = 'thread_count', default = None, type = int)
    parser.add_option('--clean', dest = 'clean_first', action = 'store_true', default = False)
    (options, args) = parser.parse_args(argv[1:])
    if len(args) != 0:
        parser.error('Incorrect number of arguments')

    run_all_tests(
        options.thread_count,
        options.clean_first,
        options.verbose)
