#!/usr/bin/env python

# Copyright 2017 Stanford University
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

import datetime, json, os, re, sys, subprocess

import github3 # Requires: pip install github3.py

def cmd(command, env=None, cwd=None):
    print(' '.join(command))
    return subprocess.check_output(command, env=env, cwd=cwd)

def get_repository(owner, repository, token):
    session = github3.login(token=token)
    return session.repository(owner=owner, repository=repository)

def create_result_file(repo, filename, result):
    now = datetime.datetime.now()
    path = '%s/%s.json' % (filename, now)
    content = json.dumps(result)
    repo.create_file(path, path, content)

class ArgvMeasurement(object):
    __slots__ = ['start']
    def __init__(self, start=None):
        self.start = int(start)
    def measure(self, argv, output):
        return argv[self.start:]

class RegexMeasurement(object):
    __slots__ = ['pattern']
    def __init__(self, pattern=None, multiline=None):
        self.pattern = re.compile(pattern, re.MULTILINE if multiline else None)
    def measure(self, argv, output):
        match = re.search(self.pattern, output)
        assert match is not None
        return match.group(1)

measurement_types = {
    'argv': ArgvMeasurement,
    'regex': RegexMeasurement,
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

    # Run command.
    args = sys.argv[1:]
    command = launcher + args
    output = cmd(command)

    # Capture measurements.
    measurement_data = {}
    for key, value in measurements.items():
        measurement_data[key] = get_measurement(value, args, output)

    # Build result.
    result = {
        'metadata': metadata,
        'measurements': measurement_data,
    }

    # Insert result into target repository.
    repo = get_repository(owner, repository, access_token)
    create_result_file(repo, os.path.basename(sys.argv[1]), result)

if __name__ == '__main__':
    driver()

