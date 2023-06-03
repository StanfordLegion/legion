#!/usr/bin/env python3

# Copyright 2023 Stanford University, NVIDIA Corporation
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

from __future__ import absolute_import, division, print_function, unicode_literals

import gc
import os
import sys
import code
import types
import struct
import readline
import threading
import importlib
import traceback
import warnings

from legion_cffi import ffi, lib as c

from legion_info import __version__

#################################################################
### Shared between Legion builtin Python and canonical Python ###
#################################################################

try:
    unicode # Python 2
except NameError:
    unicode = str # Python 3

try:
    FileNotFoundError # Python 3
except:
    FileNotFoundError = IOError # Python 2


# Storage for variables that apply to the top-level task.
# IMPORTANT: They are valid ONLY in the top-level task.
# or in global import tasks.
top_level = threading.local()
# Fields:
#     top_level.runtime
#     top_level.context
#     top_level.task

# This variable tracks all objects that need to be cleaned
# up in any python process created by legion python
_cleanup_items = list()

# This variable stores the set of modules that were already loaded at entry to
# legion mode. At shutdown we will only remove any modules that were loaded
# since then.
_starting_modules = None


def remove_added_modules():
    for mod in set(sys.modules.keys()) - _starting_modules:
        del sys.modules[mod]


def add_cleanup_item(item):
    _cleanup_items.append(item)


# Helper class for deduplicating output streams with control replication
class LegionOutputStream(object):
    def __init__(self, stream):
        # This is the original stream
        self.stream = stream

    def close(self):
        self.stream.close()

    def fileno(self):
        return self.stream.fileno()

    def flush(self):
        self.stream.flush()

    def write(self, string):
        if self.print_local_shard():
            self.stream.write(string)

    def writelines(self, sequence):
        if self.print_local_shard():
            self.stream.writelines(sequence)

    def isatty(self):
        return self.stream.isatty()

    def print_local_shard(self):
        return c.legion_runtime_local_shard_without_context() == 0
        
    def set_parent(self, parent):
        self.stream.set_parent(parent)

# Replace the output stream with one that will deduplicate
# printing for any control-replicated tasks
sys.stdout = LegionOutputStream(sys.stdout)


# Also direct output from the `warnings` module to this deduplicated
# stdout. This avoids flooding of output in control replicated
# applications that use `warnings`. This method of doing so
# was taken from https://stackoverflow.com/a/858928.
def customwarn(message, category, filename, lineno, file=None, line=None):
    sys.stdout.write(warnings.formatwarning(message, category, filename, lineno))
warnings.showwarning = customwarn


def input_args(filter_runtime_options=False):
    raw_args = c.legion_runtime_get_input_args()

    args = []
    for i in range(raw_args.argc):
        args.append(ffi.string(raw_args.argv[i]).decode('utf-8'))

    if filter_runtime_options:
        i = 1 # Skip program name

        # Legion & Realm will remove any flag that they consume, but we need to
        # do some extra filtering on top of that in some cases, to make sure
        # runtime arguments don't make it to the user code.
        prefixes = [
            # These flags are consumed by optional modules. If that module
            # is not loaded, the flag will not be cleared.
            '-cuda:', '-numa:', '-ucx:', '-gex:',
            # These flags are read by other parts of the stack, which cannot
            # remove the arguments they parse.
            '-dm:', '-bishop:'
        ]
        while i < len(args):
            match = False
            for prefix in prefixes:
                if args[i].startswith(prefix):
                    match = True
                    break
            if args[i] == '-level':
                match = True
            if args[i] == '-logfile':
                match = True
            if match:
                args.pop(i)
                # Assume that every option has an argument, as long as
                # the subsequent value does **NOT** start with a dash.
                if i < len(args) and not args[i].startswith('-'):
                    args.pop(i)
                continue
            i += 1
    return args


# In general we discourage the use of this function, but some libraries are
# not safe to use with control replication so this will give them a way
# to check whether they are running in a safe context or not
def is_control_replicated():
    try:
        # We should only be doing something for this if we're the top-level task
        return c.legion_context_get_num_shards(top_level.runtime[0],
                top_level.context[0], True) > 1
    except AttributeError:
        raise RuntimeError('"is_control_replicated" must be called in a legion_python task')


###########################################################
################ Legion Python starts here ################
###########################################################

# This has to match the unique name in main.cc
_unique_name = 'legion_python'


# This code is borrowed from the Python docs:
# https://docs.python.org/3/library/readline.html
class LegionConsole(code.InteractiveConsole):
    def __init__(self, locals=None, filename='<console>',
                 histfile = os.path.expanduser('~/.python-history')):
        code.InteractiveConsole.__init__(self, locals, filename)
        self.init_history(histfile)
        self.histfile = histfile

    def init_history(self, histfile):
        readline.parse_and_bind('tab: complete')
        if hasattr(readline, 'read_history_file'):
            try:
                readline.read_history_file(histfile)
            except FileNotFoundError:
                pass

    def save_history(self):
        readline.set_history_length(10000)
        readline.write_history_file(self.histfile)


def run_repl():
    try:
        shell = LegionConsole()
        shell.interact(banner='Welcome to Legion Python interactive console')
    except (SystemExit, KeyboardInterrupt):
        pass
    finally:
        # Save the history
        shell.save_history()
        # Wait for execution to finish here before removing the module
        # because executing tasks might still need to refer to it
        future = c.legion_runtime_issue_execution_fence(
                top_level.runtime[0], top_level.context[0])
        # block waiting on the future
        c.legion_future_wait(future, True, ffi.NULL)
        c.legion_future_destroy(future)
        del shell


def remove_all_aliases(to_delete):
    aliases = []
    for name, module in sys.modules.items():
        if module is to_delete:
            aliases.append(name)

    for name in aliases:
        del sys.modules[name]


def run_cmd(cmd, run_name):
    import imp
    module = imp.new_module(run_name)
    setattr(module, '__name__', run_name)
    setattr(module, '__package__', None)

    # Hide the current module if it exists.
    old_module = sys.modules.get(run_name)
    sys.modules[run_name] = module
    try:
        code = compile(cmd, '<string>', 'exec')
        exec(code, module.__dict__, module.__dict__)
    except SyntaxError as ex:
        traceback.print_exception(SyntaxError,ex,sys.exc_info()[2],0)
        c.legion_runtime_set_return_code(1)
    except SystemExit as ex:
        if ex.code is not None:
            if isinstance(ex.code,int):
                c.legion_runtime_set_return_code(ex.code)
            else:
                traceback.print_exception(SyntaxError,ex,sys.exc_info()[2],0)
                c.legion_runtime_set_return_code(1)
    # Wait for execution to finish here before removing the module
    # because executing tasks might still need to refer to it
    future = c.legion_runtime_issue_execution_fence(
            top_level.runtime[0], top_level.context[0])
    # block waiting on the future
    c.legion_future_wait(future, True, ffi.NULL)
    c.legion_future_destroy(future)
    # Make sure our module gets deleted to clean up any references
    # to variables the user might have made
    remove_all_aliases(module)
    if old_module is not None:
        sys.modules[run_name] = old_module
    del module


# We can't use runpy for this since runpy is aggressive about
# cleaning up after itself and removes the module before execution
# has completed.
def run_path(filename, run_name):
    import imp
    module = imp.new_module(run_name)
    setattr(module, '__name__', run_name)
    setattr(module, '__file__', filename)
    setattr(module, '__loader__', None)
    setattr(module, '__package__', run_name.rpartition('.')[0])

    # Hide the current module if it exists.
    old_module = sys.modules.get(run_name)
    sys.modules[run_name] = module

    sys.path.append(os.path.dirname(filename))

    try:
        with open(filename) as f:
            code = compile(f.read(), filename, 'exec')
            exec(code, module.__dict__, module.__dict__)
    except FileNotFoundError as ex:
        print("legion_python: can't open file "+str(filename)+": "+str(ex))
        c.legion_runtime_set_return_code(1)
    except SyntaxError as ex:
        traceback.print_exception(SyntaxError,ex,sys.exc_info()[2],0)
        c.legion_runtime_set_return_code(1)
    except SystemExit as ex:
        if ex.code is not None:
            if isinstance(ex.code,int):
                c.legion_runtime_set_return_code(ex.code)
            else:
                traceback.print_exception(SyntaxError,ex,sys.exc_info()[2],0)
                c.legion_runtime_set_return_code(1)
    # Wait for execution to finish here before removing the module
    # because executing tasks might still need to refer to it
    future = c.legion_runtime_issue_execution_fence(
            top_level.runtime[0], top_level.context[0])
    # block waiting on the future
    c.legion_future_wait(future, True, ffi.NULL)
    c.legion_future_destroy(future)
    # Make sure our module gets deleted to clean up any references
    # to variables the user might have made
    remove_all_aliases(module)
    if old_module is not None:
        sys.modules[run_name] = old_module
    del module


# This method will ensure that a module is globally imported across all
# Python processors in a Legion job before returning. It cannot be called
# within an import statement though without creating a deadlock with
# Python's import locks. Alternatively, the user can set the 'block'
# parameter to 'False' which will return a future for when the global
# import is complete and the function will return a handle to a future
# that the caller can use for checking when the global import is complete
# The type of the future is an integer that will report the number of
# failed imports. It is up to the caller to destroy the handle when done.
def import_global(module, check_depth=True, block=True):
    try:
        # We should only be doing something for this if we're the top-level task
        if c.legion_task_get_depth(top_level.task[0]) > 0 and check_depth:
            return None
    except AttributeError:
        raise RuntimeError('"import_global" must be called in a legion_python task')
    if isinstance(module,str):
        name = module
    elif isinstance(module,unicode):
        name = module
    elif isinstance(module,types.ModuleType):
        name = module.__name__
    else:
        raise TypeError('"module" arg to "import_global" must be a ModuleType or str type')
    mapper = c.legion_runtime_generate_library_mapper_ids(
            top_level.runtime[0], _unique_name.encode('utf-8'), 1)
    future = c.legion_runtime_select_tunable_value(
            top_level.runtime[0], top_level.context[0], 0, mapper, 0)
    num_python_procs = struct.unpack_from('i',
            ffi.buffer(c.legion_future_get_untyped_pointer(future),4))[0]
    c.legion_future_destroy(future)
    assert num_python_procs > 0
    # Launch an index space task across all the python
    # processors to import the module in every interpreter
    task_id = c.legion_runtime_generate_library_task_ids(
            top_level.runtime[0], _unique_name.encode('utf-8'), 3) + 1
    rect = ffi.new('legion_rect_1d_t *')
    rect[0].lo.x[0] = 0
    rect[0].hi.x[0] = num_python_procs - 1
    domain = c.legion_domain_from_rect_1d(rect[0])
    packed = name.encode('utf-8')
    arglen = len(packed)
    array = ffi.new('char[]', arglen)
    ffi.buffer(array, arglen)[:] = packed
    args = ffi.new('legion_task_argument_t *')
    args[0].args = array
    args[0].arglen = arglen
    argmap = c.legion_argument_map_create()
    launcher = c.legion_index_launcher_create(task_id, domain,
            args[0], argmap, c.legion_predicate_true(), False, mapper, 0)
    future = c.legion_index_launcher_execute_reduction(top_level.runtime[0],
            top_level.context[0], launcher, c.LEGION_REDOP_SUM_INT32)
    c.legion_index_launcher_destroy(launcher)
    c.legion_argument_map_destroy(argmap)
    if block:
        result = struct.unpack_from('i',
                ffi.buffer(c.legion_future_get_untyped_pointer(future),4))[0]
        c.legion_future_destroy(future)
        if result > 0:
            raise ImportError('failed to globally import '+name+' on '+str(result)+' nodes')
        return None
    else:
        return future


def legion_python_main(raw_args, user_data, proc):
    # record modules loaded at the beginning of the run
    global _starting_modules
    _starting_modules = set(sys.modules.keys())

    raw_arg_ptr = ffi.new('char[]', bytes(raw_args))
    raw_arg_size = len(raw_args)

    # Execute preamble to obtain Legion API context.
    task = ffi.new('legion_task_t *')
    raw_regions = ffi.new('legion_physical_region_t **')
    num_regions = ffi.new('unsigned *')
    context = ffi.new('legion_context_t *')
    runtime = ffi.new('legion_runtime_t *')
    c.legion_task_preamble(
        raw_arg_ptr, raw_arg_size, proc,
        task, raw_regions, num_regions, context, runtime)

    top_level.runtime, top_level.context, top_level.task = runtime, context, task

    # Run user's script.
    args = input_args(True)
    start = 1
    if len(args) > 1 and args[1] == '--nocr':
        start += 1
        local_cleanup = False
    else:
        local_cleanup = True
    if len(args) < (start+1):
        sys.argv = ['']
        run_repl()
    elif args[start] == '-':
        sys.argv = args[start:]
        run_repl()
    elif args[start] == '-h':
        print('usage: legion_python [--nocr] [-c cmd | -m mod | file | -] [arg] ...')
        print('--nocr : disable control replication for multi-node execution')
        print('-c cmd : program passed in as string (terminates option list)')
        print('-h     : print this help message and exit')
        print('-m mod : run library module as a script (terminates option list)')
        print('file   : program read from script file')
        print('-      : program read from stdin (default; interactive mode if a tty)')
        print('arg ...: arguments passed to program in sys.argv[1:]')
    elif args[start] == '-c':
        if len(args) > (start+1):
            sys.argv = ['-c'] + list(args[start+2:])
            run_cmd(args[start+1], '__main__')
        else:
            print('Argument expected for the -c option')
            c.legion_runtime_set_return_code(1)
    elif args[start] == '-m':
        if len(args) > (start+1):
            mod_name = args[start+1]
            mod_dir = mod_name.split(".")
            if len(mod_dir) > 1:
                prefix_path = os.path.join(*mod_dir[:-1])
                mod_name = mod_dir[-1]
            else:
                prefix_path = ""
            filename = mod_name + '.py'
            found = False
            for path in sys.path:
                path = os.path.join(path, prefix_path)
                main_py_full_path = os.path.join(path, mod_name, '__main__.py')
                filename_full_path = os.path.join(path, filename)
                # first, check mod_name/__main__.py
                if os.path.exists(main_py_full_path):
                    module = main_py_full_path
                # second, check mod_name.py
                elif os.path.exists(filename_full_path):
                    module = filename_full_path
                else:
                    continue
                sys.argv = [module] + list(args[start+2:])
                run_path(module, '__main__')
                found = True
                break
            if not found:
                print('No module named ' + mod_name)
                c.legion_runtime_set_return_code(1)
        else:
            print('Argument expected for the -m option')
            c.legion_runtime_set_return_code(1)
    else:
        assert start < len(args)
        sys.argv = list(args[start:])
        run_path(args[start], '__main__')

    if local_cleanup:
        # If we were control replicated then we just need to do our cleanup
        # Do it in reverse order so modules get FILO properties
        for cleanup in reversed(_cleanup_items):
            cleanup()
        remove_added_modules()
    else:
        # Otherwise, run a task on every node to perform the cleanup
        mapper = c.legion_runtime_generate_library_mapper_ids(
                top_level.runtime[0], _unique_name.encode('utf-8'), 1)
        future = c.legion_runtime_select_tunable_value(
                top_level.runtime[0], top_level.context[0], 0, mapper, 0)
        num_python_procs = struct.unpack_from('i',
                ffi.buffer(c.legion_future_get_untyped_pointer(future),4))[0]
        c.legion_future_destroy(future)
        assert num_python_procs > 0
        # Launch an index space task across all the python
        # processors to import the module in every interpreter
        task_id = c.legion_runtime_generate_library_task_ids(
                top_level.runtime[0], _unique_name.encode('utf-8'), 3) + 2
        rect = ffi.new('legion_rect_1d_t *')
        rect[0].lo.x[0] = 0
        rect[0].hi.x[0] = num_python_procs - 1
        domain = c.legion_domain_from_rect_1d(rect[0])
        args = ffi.new('legion_task_argument_t *')
        args[0].args = ffi.NULL
        args[0].arglen = 0
        argmap = c.legion_argument_map_create()
        launcher = c.legion_index_launcher_create(task_id, domain,
            args[0], argmap, c.legion_predicate_true(), False, mapper, 0)
        future_map = c.legion_index_launcher_execute(top_level.runtime[0],
                top_level.context[0], launcher)
        c.legion_index_launcher_destroy(launcher)
        c.legion_argument_map_destroy(argmap)
        # Wait for all the cleanup tasks to be done
        c.legion_future_map_wait_all_results(future_map)
        c.legion_future_map_destroy(future_map)

    del top_level.runtime
    del top_level.context
    del top_level.task

    # Force a garbage collection so that we know that all objects which can
    # be collected are actually collected before we exit the top-level task
    gc.collect()

    # Execute postamble.
    c.legion_task_postamble(runtime[0], context[0], ffi.NULL, 0)


# This is our cleanup task that is run on every python processor when we
# are not control replicated to ensure that everything is collected before
# we exit the top-level task on node 0
def legion_python_cleanup(raw_args, user_data, proc):
    raw_arg_ptr = ffi.new('char[]', bytes(raw_args))
    raw_arg_size = len(raw_args)

    # Execute preamble to obtain Legion API context.
    task = ffi.new('legion_task_t *')
    raw_regions = ffi.new('legion_physical_region_t **')
    num_regions = ffi.new('unsigned *')
    context = ffi.new('legion_context_t *')
    runtime = ffi.new('legion_runtime_t *')
    c.legion_task_preamble(
        raw_arg_ptr, raw_arg_size, proc,
        task, raw_regions, num_regions, context, runtime)

    top_level.runtime, top_level.context, top_level.task = runtime, context, task

    # Do it in reverse order so modules get FILO properties
    for cleanup in reversed(_cleanup_items):
        cleanup()
    remove_added_modules()

    del top_level.runtime
    del top_level.context
    del top_level.task

    # Force a garbage collection so that we know that all objects which can
    # be collected are actually collected before we exit this cleanup task
    gc.collect()

    c.legion_task_postamble(runtime[0], context[0], ffi.NULL, 0)


# This is our helper task for ensuring that python modules are imported
# globally on all python processors across the system
def legion_python_import_global(raw_args, user_data, proc):
    raw_arg_ptr = ffi.new('char[]', bytes(raw_args))
    raw_arg_size = len(raw_args)

    # Execute preamble to obtain Legion API context.
    task = ffi.new('legion_task_t *')
    raw_regions = ffi.new('legion_physical_region_t **')
    num_regions = ffi.new('unsigned *')
    context = ffi.new('legion_context_t *')
    runtime = ffi.new('legion_runtime_t *')
    c.legion_task_preamble(
        raw_arg_ptr, raw_arg_size, proc,
        task, raw_regions, num_regions, context, runtime)

    top_level.runtime, top_level.context, top_level.task = runtime, context, task

    # Get the name of the task
    module_name = ffi.unpack(ffi.cast('char*', c.legion_task_get_args(task[0])),
            c.legion_task_get_arglen(task[0])).decode('utf-8')
    try:
        globals()[module_name] = importlib.import_module(module_name)
        failures = 0
    except ImportError:
        failures = 1

    del top_level.runtime
    del top_level.context
    del top_level.task

    result = struct.pack('i',failures)

    c.legion_task_postamble(runtime[0], context[0], ffi.from_buffer(result), 4)


###########################################################
############## Canonical Python starts here ###############
###########################################################

# This variable is used to track how many times the legion_canonical_python_main
# has been called. We only initialize the top level task when top_level_counter == 0
_top_level_counter = 0

# Since we are not able to shut down and re-start Legion, we use this variable to
# track if Legion has been inited
_legion_inited = False


def legion_canonical_python_main(sys_argv=None):
    # record modules loaded at the beginning of the run
    global _starting_modules
    _starting_modules = set(sys.modules.keys())

    # Do not init top level task if it has been initialized
    global _top_level_counter
    _top_level_counter += 1
    if _top_level_counter > 1:
        return
    global _legion_inited
    assert _legion_inited == False

    if sys_argv is None:
        sys_argv = sys.argv
    argv = []
    for arg in sys_argv:
        argv.append(ffi.new("char[]", arg.encode('ascii')))
    c.legion_canonical_python_begin_top_level_task(len(argv), argv)
    context = c.legion_runtime_get_context()
    runtime = c.legion_runtime_get_runtime()
    top_level.runtime, top_level.context, top_level.task = [runtime], [context], [None]
    _legion_inited = True

def legion_canonical_python_cleanup():
    global _top_level_counter
    _top_level_counter -= 1
    if _top_level_counter > 0:
        return
    global _legion_inited
    assert _legion_inited == True
    # add an execution fence to make sure all previous tasks are done
    future = c.legion_runtime_issue_execution_fence(
               top_level.runtime[0], top_level.context[0])
    c.legion_future_wait(future, True, ffi.NULL)
    c.legion_future_destroy(future)
    # clean up modules
    for cleanup in reversed(_cleanup_items):
        cleanup()
    remove_added_modules()
    # clean up context
    c.legion_context_destroy(top_level.context[0])
    c.legion_canonical_python_end_top_level_task()
    del top_level.context
    del top_level.runtime
    gc.collect()
