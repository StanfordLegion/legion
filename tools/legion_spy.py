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

import subprocess
import sys, os, shutil
import string
from getopt import getopt
from spy_parser import parse_log_file
from spy_analysis import *
#from spy_state import *

temp_dir = ".cent/"

def usage():
    print "Usage: "+sys.argv[0]+" [-l -c -p -m -r -i -k -s -v -P -D] <file_name>"
    print "  -l : perform logical analyses"
    print "  -c : perform physical analyses"
    print "  -p : make task pictures"
    print "  -k : keep temporary files"
    print "  -s : generate simplified graphs"
    print "  -v : verbose"
    print "  -P : make partition graphs"
    print "  -D : make dataflow graphs"
    print "  -i : make instance graphs"
    sys.exit(1)

def main():
    if len(sys.argv) < 2:
        usage()

    opts, args = getopt(sys.argv[1:],'lipckrmvsPD')
    opts = dict(opts)
    if len(args) <> 1:
        usage()
    file_name = args[0]

    logical_checks = False
    physical_checks = False
    make_pictures = False
    make_dataflow_graphs = False
    keep_temp_files = False
    print_instances = False
    print_processor_graphs = False 
    print_memory_graphs = False 
    simplify_graphs = False 
    verbose = False
    make_partition_graphs = False
    for opt in opts:
        if opt == '-l':
            logical_checks = True
            continue
        if opt == '-c':
            physical_checks = True
            continue
        if opt == '-p':
            make_pictures = True
            continue
        if opt == '-k':
            keep_temp_files = True
            continue
        if opt == '-i':
            print_instances = True
            continue
        if opt == '-r':
            print_processor_graphs = True
            continue
        if opt == '-m':
            print_memory_graphs = True
            continue
        if opt == '-s':
            simplify_graphs = True
            continue
        if opt == '-v':
            verbose = True
            continue
        if opt == '-P':
            make_partition_graphs = True
            continue
        if opt == '-D':
            make_dataflow_graphs = True
            continue

    state = State(verbose)

    print 'Loading log file '+file_name+'...'
    total_matches = parse_log_file(file_name, state)
    print 'Matched '+str(total_matches)+' lines'
    if total_matches == 0:
        print 'No matches. Exiting...'
        return

    if logical_checks:
        print "Performing logical checks..."
        state.check_logical()
    if make_pictures:
        print "Printing event graphs..."
        state.print_pictures(temp_dir, simplify_graphs)
    if physical_checks:
        print "Performing physical checks..."
        state.check_data_flow()
        state.check_instance_dependences()
    if print_instances:
        print "Printing instance graphs..."
        state.print_instance_graphs(temp_dir)
    if print_processor_graphs:
        print "Making processor graphs..."
        state.print_processor_graphs(temp_dir)
    if print_memory_graphs:
        print "Making memory graphs..."
        state.print_memory_graphs(temp_dir)
    #if verbose:
    #    state.print_instances()
    if make_partition_graphs:
        print "Printing partition graphs..."
        state.print_partition_graphs(temp_dir, simplify_graphs)
    if make_dataflow_graphs:
        print "Printing dataflow graphs..."
        state.print_dataflow_graphs(temp_dir, simplify_graphs)

    print 'Legion Spy analysis complete.  Exiting...'
    if keep_temp_files:
        try:
            subprocess.check_call(['cp '+temp_dir+'* .'],shell=True)
        except:
            print 'WARNING: Unable to copy temporary files into current directory'

if __name__ == "__main__":
    try:
        os.mkdir(temp_dir)
        sys.setrecursionlimit(10000)
        main()
        shutil.rmtree(temp_dir)
    except:
        shutil.rmtree(temp_dir)
        raise

# EOF

