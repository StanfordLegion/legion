#!/usr/bin/python

# Copyright 2012 Stanford University
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

#!/usr/bin/python

import subprocess
import sys, os, shutil
import string
from getopt import getopt
from cent_parse import parse_log_file
from cent_state import *

temp_dir = ".cent/"

def usage():
    print "Usage: "+sys.argv[0]+" [-l -p -m -k] <file_name>"
    sys.exit(1)

def main():
    if len(sys.argv) < 2:
        usage()

    opts, args = getopt(sys.argv[1:],'lipckv')
    opts = dict(opts)
    if len(args) <> 1:
        usage()
    file_name = args[0]

    logical_checks = False
    physical_checks = False
    make_pictures = False
    keep_temp_files = False
    print_instances = False
    verbose = False
    if '-l' in opts:
        logical_checks = True
    if '-c' in opts:
        physical_checks = True
    if '-p' in opts:
        make_pictures = True
    if '-k' in opts:
        keep_temp_files = True
    if '-i' in opts:
        print_instances = True
    if '-v' in opts:
        verbose = True

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
        state.print_pictures(temp_dir)
    if physical_checks:
        print "Performing physical checks..."
        state.check_instance_dependences()
        state.check_data_flow()
    if print_instances:
        print "Printing instance graphs..."
        state.print_instance_graphs(temp_dir)

    print 'Legion Spy analysis complete.  Exiting...'
    if keep_temp_files:
        try:
            subprocess.check_call(['cp '+temp_dir+'* .'],shell=True)
        except:
            print 'WARNING: Unable to copy temporary files into current directory'

if __name__ == "__main__":
    try:
        os.mkdir(temp_dir)
        main()
        shutil.rmtree(temp_dir)
    except:
        shutil.rmtree(temp_dir)
        raise

# EOF

