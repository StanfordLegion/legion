#!/usr/bin/env python3

# Copyright 2023 Stanford University
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
### Tool to run a performance measurement and upload results
###

from __future__ import print_function

import datetime, json, os, re, sys, subprocess

import github3 # Requires: pip install github3.py

def cmd(command, env=None, cwd=None):
    print(' '.join(command))
    return subprocess.check_output(command, env=env, cwd=cwd)

def get_repository(owner, repository, token):
    session = github3.login(token=token)
    return session.repository(owner=owner, repository=repository)

def create_result_file(repo, path, result):
    content = json.dumps(result)
    repo.create_file(path, 'Add measurement %s.' % path, content)

class ArgvMeasurement(object):
    __slots__ = ['start', 'index', 'filter']
    def __init__(self, start=None, index=None, filter=None):
        if (start is None) == (index is None):
            raise Exception('ArgvMeasurement requires start or index, but not both')
        self.start = int(start) if start is not None else None
        self.index = int(index) if index is not None else None
        if filter is None:
            self.filter = lambda x: x
        elif filter == "basename":
            self.filter = os.path.basename
        else:
            raise Exception('Unrecognized filter "%s"' % filter)
    def measure(self, argv, output):
        if self.start is not None:
            return [self.filter(x) for x in argv[self.start:]]
        elif self.index is not None:
            return self.filter(argv[self.index])
        else:
            assert False

class RegexMeasurement(object):
    __slots__ = ['pattern']
    def __init__(self, pattern=None, multiline=None):
        self.pattern = re.compile(pattern, re.MULTILINE if multiline else None)
    def measure(self, argv, output):
        match = re.search(self.pattern, output)
        if match is None:
            raise Exception('Regex match failed')
        result = match.group(1).strip()
        if len(result) == 0:
            raise Exception('Regex produced empty match')
        return result

class CommandMeasurement(object):
    __slots__ = ['args']
    def __init__(self, args=None):
        self.args = args
    def measure(self, argv, output):
        proc = subprocess.Popen(
            self.args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        result, _ = proc.communicate(output)
        if proc.returncode != 0:
            raise Exception('Command failed')
        result = result.strip()
        if len(result) == 0:
            raise Exception('Command produced no output')
        return result

measurement_types = {
    'argv': ArgvMeasurement,
    'regex': RegexMeasurement,
    'command': CommandMeasurement,
}

def strip_type(type=None, **kwargs):
    return kwargs

def get_measurement(value, argv, output):
    if 'type' not in value:
        raise Exception('Malformed measurement: Needs field "type"')
    if value['type'] not in measurement_types:
        raise Exception(
            'Malformed measurement: Unrecognized type "%s"' % value['type'])
    measurement = measurement_types[value['type']]
    return measurement(**strip_type(**value)).measure(argv, output)

def get_variable(name, description):
    if name not in os.environ:
        raise Exception(
            'Please set environment variable %s to %s' % (name, description))
    return os.environ[name]

def driver():
    # Parse inputs.
    owner = get_variable('PERF_OWNER', 'Github respository owner')
    repository = get_variable('PERF_REPOSITORY', 'Github respository name')
    access_token = get_variable('PERF_ACCESS_TOKEN', 'Github access token')
    metadata = json.loads(
        get_variable('PERF_METADATA', 'JSON-encoded metadata'))
    measurements = json.loads(
        get_variable('PERF_MEASUREMENTS', 'JSON-encoded measurements'))
    launcher = get_variable('PERF_LAUNCHER', 'launcher command').split()

    # Validate inputs.
    if 'benchmark' not in measurements:
        raise Exception('Malformed measurements: Measurement "benchmark" is required')
    if 'argv' not in measurements:
        raise Exception('Malformed measurements: Measurement "argv" is required')

    # Run command.
    args = sys.argv[1:]
    command = launcher + args
    output = cmd(command)

    # Capture measurements.
    measurement_data = {}
    for key, value in measurements.items():
        measurement_data[key] = get_measurement(value, args, output)

    # Build result.
    # Move benchmark and argv into metadata from measurements.
    metadata['benchmark'] = measurement_data['benchmark']
    del measurement_data['benchmark']
    metadata['argv'] = measurement_data['argv']
    del measurement_data['argv']
    # Capture measurement time.
    metadata['date'] = datetime.datetime.now().isoformat()
    result = {
        'metadata': metadata,
        'measurements': measurement_data,
    }

    print()
    print('"measurements":', json.dumps(measurement_data, indent=4, sort_keys=True))

    # Insert result into target repository.
    repo = get_repository(owner, repository, access_token)
    path = os.path.join('measurements', metadata['benchmark'], '%s.json' % metadata['date'])
    create_result_file(repo, path, result)

if __name__ == '__main__':
    driver()

