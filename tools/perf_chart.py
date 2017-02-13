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

import argparse, collections, datetime, json, os, shutil, subprocess, sys, tempfile

import github3 # Requires: pip install github3.py

_version = sys.version_info.major

if _version == 2: # Python 2.x:
    def _glob(path):
        def visit(result, dirname, filenames):
            for filename in filenames:
                result.append(os.path.join(dirname, filename))
        result = []
        os.path.walk(path, visit, result)
        return result
elif _version == 3: # Python 3.x:
    def _glob(path):
        return [os.path.join(dirname, filename)
                for dirname, _, filenames in os.walk(path)
                for filename in filenames]
else:
    raise Exception('Incompatible Python version')

def get_measurements(repo_url):
    tmp_dir = tempfile.mkdtemp()
    try:
        print(tmp_dir)
        subprocess.check_call(
            ['git', 'clone', repo_url, 'measurements'],
            cwd=tmp_dir)
        measurements_dir = os.path.join(tmp_dir, 'measurements', 'measurements')
        print(measurements_dir)
        measurements_paths = [path for path in _glob(measurements_dir)
                              if os.path.splitext(path)[1] == '.json']
        measurements = []
        for path in measurements_paths:
            with open(path) as f:
                measurements.append((path, json.load(f)))
        return measurements
    finally:
        shutil.rmtree(tmp_dir)

def extract_measurements(measurements):
    branches = set()
    commit_date = {}
    commits_by_branch = collections.defaultdict(lambda: set())
    measurements_by_commit = collections.defaultdict(lambda: [])
    for path, measurement in measurements:
        commit = measurement['metadata']['commit']
        branch = measurement['metadata']['branch']
        argv = ' '.join(measurement['metadata']['argv'])

        # Reinsert the compact argv into the measurement.
        measurement['metadata']['argv'] = argv

        # Record the branch used.
        branches.add(branch)

        # Record the earliest measurement date for this commit.
        date = datetime.datetime.strptime(
            measurement['metadata']['date'],
            '%Y-%m-%dT%H:%M:%S.%f')
        if commit not in commit_date or date < commit_date[commit]:
            commit_date[commit] = date

        # Add the commit to this branch.
        commits_by_branch[branch].add(commit)

        # Record the measurement.
        measurements_by_commit[commit].append(measurement)

    # Sort commits by earliest measurement date.
    commits_by_branch_by_date = dict(
        (branch, sorted(commits, key=lambda x: commit_date[x]))
        for branch, commits in commits_by_branch.items())

    return branches, commits_by_branch_by_date, measurements_by_commit

def get_repository(owner, repository, token):
    session = github3.login(token=token)
    return session.repository(owner=owner, repository=repository)

def push_json_file(repo, path, value):
    # Try to produce JSON files will that will generate small diffs.
    content = json.dumps(value, indent=0, separators=(',', ':'), sort_keys=True)
    previous = repo.contents(path)
    if previous is not None:
        repo.update_file(path, 'Update rendered chart.', content, previous.sha)
    else:
        repo.create_file(path, 'Update rendered chart.', content)

def make_charts(owner, repository, access_token, measurement_url):
    measurements = get_measurements(measurement_url)
    print('Got %s measurements...' % len(measurements))
    branches, commits, measurements = extract_measurements(measurements)

    result = {
        'branches': list(branches),
        'commits': commits,
        'measurements': measurements,
    }

    repo = get_repository(owner, repository, access_token)
    push_json_file(repo, 'rendered/chart.json', result)

def get_variable(name, description):
    if name not in os.environ:
        raise Exception(
            'Please set environment variable %s to %s' % (name, description))
    return os.environ[name]

def driver():
    owner = get_variable('PERF_OWNER', 'Github respository owner')
    repository = get_variable('PERF_REPOSITORY', 'Github respository name')
    access_token = get_variable('PERF_ACCESS_TOKEN', 'Github access token')

    parser = argparse.ArgumentParser(
        description = 'Render Legion performance charts')
    parser.add_argument('measurement_url', help='measurement repository URL')

    args = parser.parse_args()

    make_charts(owner, repository, access_token, **vars(args))

if __name__ == '__main__':
    driver()
