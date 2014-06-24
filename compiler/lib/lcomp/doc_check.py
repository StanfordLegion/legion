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
### Type Checker
###

# Work around for OrderedDict missing in Python 2.6.
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
import json, os, subprocess, tempfile
from cStringIO import StringIO
from . import passes, style_check

def extract_code(doc, language):
    code = {}
    for elt in doc[1]:
        if 'CodeBlock' in elt:
            block = elt['CodeBlock']
            identifier = block[0][0]
            classes = block[0][1]

            if language in classes:
                if identifier not in code:
                    code[identifier] = []
                code[identifier].append(block[1])
    return [str('\n\n'.join(section)) for section in code.itervalues()]

def doc_check(opts, source):
    doc = json.loads(
        subprocess.check_output(['pandoc', '--to', 'json'],
                                stdin = source))
    code = extract_code(doc, 'legion')

    result = []
    for section in code:
        program = passes.parse(StringIO(section))
        style_check.style_check(program)
        type_map, constraints, foreign_types = passes.check(program, opts)
        result.append((type_map, constraints, foreign_types))
    return result
