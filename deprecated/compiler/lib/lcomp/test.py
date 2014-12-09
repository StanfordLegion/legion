#!/usr/bin/env python

# Copyright 2014 Stanford University and Los Alamos National Security, LLC
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

import json, multiprocessing, optparse, os, re, shutil, subprocess, sys, tempfile, traceback
from cStringIO import StringIO
# Work around for OrderedDict missing in Python 2.6.
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
from . import ast, doc_check, driver, options, passes, pretty, style_check

red = "\033[1;31m"
green = "\033[1;32m"
clear = "\033[0m"

class environment_variable:
    def __init__(self, variable, value, optional):
        self.variable = variable
        self.value = value
        self.optional = optional
        self.cleanup = None
    def __enter__(self):
        if self.variable in os.environ:
            if not self.optional:
                raise Exception('Environment variable %s already defined' % (
                    self.variable))
            return self
        self.cleanup = True
        os.environ[self.variable] = self.value
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cleanup:
            del os.environ[self.variable]
        return False

default_runtime_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '..', '..', 'runtime')

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
            ['./driver.py', '-g', temp_source, '-o', temp_binary] +
            (['-j', str(thread_count)] if thread_count is not None else []) +
            (['--clean'] if clean_first else []) +
            (['-v'] if verbose else ['-q']))
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

def test_style_pass(filename, verbose):
    with open(filename, 'rb') as f:
        program = passes.parse(f)
        style_check.style_check(program)

_re_expected_failure = re.compile(r'^[ \t\r]*//[ \t]+fails-with:[ \t\r]*$\n((^[ \t\r]*//.*$\n)+)', re.MULTILINE)
def test_style_fail(filename, verbose):
    # Search for the expected failure in the program
    with open(filename, 'rb') as f:
        program_text = f.read()
    expected_failure_match = re.search(_re_expected_failure, program_text)
    if expected_failure_match is None:
        raise Exception('No fails-with declaration in style_fail testcase')
    expected_failure_lines = expected_failure_match.group(1).strip().split('\n')
    expected_failure = '\n'.join([line.strip()[2:].strip() for line in expected_failure_lines])

    # Check that the program fails to compile
    try:
        with open(filename, 'rb') as f:
            program = passes.parse(f)
        style_check.style_check(program)
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
        raise Exception('Expecting failure, style checker returned successfully')

def test_compile_pass(filename, verbose, flags):
    with environment_variable('LG_RT_DIR', default_runtime_dir, True):

        if verbose:
            stdout, stderr = None, None
        else:
            stdout, stderr = subprocess.PIPE, subprocess.STDOUT

        (fd, temp_obj) = tempfile.mkstemp(suffix = '.lg.o')
        try:
            os.close(fd)
            saved_temps = driver.driver(
                ['./driver.py', '-g', '-c', filename, '-o', temp_obj, '--save-temps'] +
                (['-v'] if verbose else ['-q']) +
                flags)
        except:
            # On failure, keep saved temporaries so that the user can
            # inspect the failure.
            raise
        else:
            # On success, clean up the saved temporaries.
            for saved_temp in saved_temps:
                shutil.rmtree(saved_temp)
        finally:
            # Always remove the output file, as it will not be valid on
            # failure anyway.
            os.remove(options.with_ext(temp_obj, '.h'))
            os.remove(temp_obj)

def test_compile_fail(filename, verbose):
    # Search for the expected failure in the program
    with open(filename, 'rb') as f:
        program_text = f.read()
    expected_failure_match = re.search(_re_expected_failure, program_text)
    if expected_failure_match is None:
        raise Exception('No fails-with declaration in compile_fail testcase')
    expected_failure_lines = expected_failure_match.group(1).strip().split('\n')
    expected_failure = '\n'.join([line.strip()[2:].strip() for line in expected_failure_lines])

    # Check that the program fails to compile
    try:
        with open(filename, 'rb') as f:
            program = passes.parse(f)
        opts = options.build_fake_options(filename, verbose)
        type_map, constraints = passes.check(program, opts)
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

_re_run_params = re.compile(r'^[ \t\r]*//[ \t]+run-with-params:[ \t\r]*$\n((^[ \t\r]*//.*$\n)+)', re.MULTILINE)
def test_run_pass(filename, verbose, compile_flags):
    # Search for the command-line parameters specified in the program
    with open(filename, 'rb') as f:
        program_text = f.read()
    run_params_match = re.search(_re_run_params, program_text)
    run_configurations = [[]]
    if run_params_match is not None:
        run_params_lines = run_params_match.group(1).strip().split('\n')
        run_params = '\n'.join([line.strip()[2:].strip() for line in run_params_lines])
        run_configurations = json.loads(run_params)

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
        saved_temps = driver.driver(
            ['./driver.py', '-g'] +
            input_filenames +
            ['-o', temp_binary, '-j', '1', '--save-temps'] +
            (['-v'] if verbose else ['-q']) +
            compile_flags)
        for run_configuration in run_configurations:
            run_command = [temp_binary] + run_configuration
            if verbose:
                print ' '.join(run_command)
            proc = subprocess.Popen(
                run_command,
                cwd = os.path.dirname(os.path.realpath(filename)),
                stdout = stdout, stderr = stderr)
            output, errors = proc.communicate()
            if proc.returncode != 0:
                raise driver.CompilerException(output, errors, saved_temps)
    except:
        # On failure, keep saved temporaries so that the user can
        # inspect the failure.
        raise
    else:
        # On success, clean up the saved temporaries.
        for saved_temp in saved_temps:
            shutil.rmtree(saved_temp)
    finally:
        # Always remove the output file, as it will not be valid on
        # failure anyway.
        os.remove(temp_binary)

def test_run_fail(filename, verbose, compile_flags):
    # Search for the expected failure in the program
    with open(filename, 'rb') as f:
        program_text = f.read()
    expected_failure_match = re.search(_re_expected_failure, program_text)
    if expected_failure_match is None:
        raise Exception('No fails-with declaration in run_fail testcase')
    expected_failure_lines = expected_failure_match.group(1).strip().split('\n')
    expected_failure = '\n'.join([line.strip()[2:].strip() for line in expected_failure_lines])

    if 'LG_RT_DIR' not in os.environ:
        raise Exception('LG_RT_DIR variable is not defined')

    if verbose:
        stdout, stderr = None, subprocess.PIPE
    else:
        stdout, stderr = subprocess.PIPE, subprocess.PIPE

    input_filenames = [filename]
    cpp_filename = ''.join([os.path.splitext(filename)[0], '.cc'])
    if os.path.exists(cpp_filename):
        input_filenames.append(cpp_filename)

    (fd, temp_binary) = tempfile.mkstemp()
    try:
        os.close(fd)
        saved_temps = driver.driver(
            ['./driver.py', '-g'] +
            input_filenames +
            ['-o', temp_binary, '-j', '1', '--save-temps'] +
            (['-v'] if verbose else ['-q']) +
            compile_flags)
        proc = subprocess.Popen(
            [temp_binary],
            cwd = os.path.dirname(os.path.realpath(filename)),
            stdout = stdout, stderr = stderr)
        output, errors = proc.communicate()
        if proc.returncode == 0:
            raise Exception('Expecting failure, program returned successfully')
        else:
            if errors.find(expected_failure) < 0:
                raise Exception('Expected failure not found in program stderr')
    except KeyboardInterrupt:
        raise
    except driver.CompilerException:
        # Clean up the saved temporaries.
        for saved_temp in saved_temps:
            shutil.rmtree(saved_temp)
        raise
    else:
        # Clean up the saved temporaries.
        for saved_temp in saved_temps:
            shutil.rmtree(saved_temp)
    finally:
        # Always remove the output file.
        os.remove(temp_binary)

def test_compile_fail(filename, verbose):
    # Search for the expected failure in the program
    with open(filename, 'rb') as f:
        program_text = f.read()
    expected_failure_match = re.search(_re_expected_failure, program_text)
    if expected_failure_match is None:
        raise Exception('No fails-with declaration in compile_fail testcase')
    expected_failure_lines = expected_failure_match.group(1).strip().split('\n')
    expected_failure = '\n'.join([line.strip()[2:].strip() for line in expected_failure_lines])

    # Check that the program fails to compile
    try:
        with open(filename, 'rb') as f:
            program = passes.parse(f)
        opts = options.build_fake_options(filename, verbose)
        type_map, constraints = passes.check(program, opts)
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

def test_doc(filename, verbose):
    with open(filename) as source:
        opts = options.build_fake_options(filename, verbose)
        doc_check.doc_check(opts, source)

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
    except driver.CompilerException as e:
        if verbose:
            return test_name, filename, e.saved_temps, FAIL, ''.join(traceback.format_exception(*sys.exc_info()))
        return test_name, filename, e.saved_temps, FAIL, None
    except Exception as e:
        if verbose:
            return test_name, filename, [], FAIL, ''.join(traceback.format_exception(*sys.exc_info()))
        return test_name, filename, [], FAIL, None
    else:
        return test_name, filename, [], PASS, None

class Counter:
    def __init__(self):
        self.passed = 0
        self.failed = 0

tests = [
    ('pretty', (test_pretty, ()),
     (os.path.join('tests', 'compile_pass'),
      os.path.join('tests', 'run_pass'),
      os.path.join('examples'))),
    ('style_fail', (test_style_fail, ()),
     (os.path.join('tests', 'style_fail'),)),
    ('style_pass', (test_style_pass, ()),
     (os.path.join('tests', 'compile_pass'),
      os.path.join('tests', 'run_pass'),
      os.path.join('examples'))),
    ('compile_fail', (test_compile_fail, ()),
     (os.path.join('tests', 'compile_fail'),)),
    ('compile_pass', (test_compile_pass, ([],)),
     (os.path.join('tests', 'style_fail'),
      os.path.join('tests', 'compile_pass'),
      os.path.join('tests', 'run_pass'),
      os.path.join('examples'))),
    ('doc', (test_doc, ()),
     ('README.md', 'doc')),
    ('run_pass', (test_run_pass, ([],)),
     (os.path.join('tests', 'run_pass'),
      os.path.join('examples'))),
    ('run_pass_dcheck', (test_run_pass, (['--pointer-checks'],)),
     (os.path.join('tests', 'run_pass'),)),
    ('run_fail_dcheck', (test_run_fail, (['--pointer-checks'],)),
     (os.path.join('tests', 'run_fail'),)),
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
                        for name in sorted(names)
                        for path in [os.path.join(dirname, name)]
                        if os.path.isfile(path) and os.path.splitext(path)[1] in ('.lg', '.md')),
                    ())

        for test_path in test_paths:
            results.append(thread_pool.apply_async(test_runner, (test_name, test_fn, verbose, test_path)))

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

def test_driver(argv = None):
    if argv is None:
        argv = sys.argv

    parser = optparse.OptionParser(description = 'Legion compiler test suite')
    parser.add_option('-v', dest = 'verbose', action = 'store_true', default = False, help = 'display verbose output')
    parser.add_option('-j', dest = 'thread_count', default = None, type = int, help = 'number of threads used to run test suite')
    parser.add_option('--clean', dest = 'clean_first', action = 'store_true', default = False, help = 'recompile the Legion runtime')
    (options, args) = parser.parse_args(argv[1:])
    if len(args) != 0:
        parser.error('Incorrect number of arguments')

    run_all_tests(
        options.thread_count,
        options.clean_first,
        options.verbose)
