#!/usr/bin/env python3

import os
import sys
import argparse
import requests
import json
import time
import datetime
import gzip

parser = argparse.ArgumentParser(description = 'GitLab-CI pipeline fetcher')
parser.add_argument('-v', '--verbose',
                    action='store_true',
                    default=False,
                    help='print lots of information')
parser.add_argument('-q', '--quiet',
                    action='store_true',
                    default=False,
                    help='suppress all normal output')
parser.add_argument('-T', '--tokenfile',
                    help='path to file containing gitlab token')
parser.add_argument('-t', '--token',
                    help='gitlab token (for authentication)')
parser.add_argument('--repo', default='StanfordLegion/legion',
                    help='gitlab repo (including group) to query')
parser.add_argument('-d', '--delay', type=float, default=0.1,
                    help='delay (in seconds) between page fetches')
parser.add_argument('-M', '--max_age', type=int,
                    help='age limit (in days) of pipelines to consider')
parser.add_argument('-m', '--max_pipelines', type=int,
                    help='maximum number of pipelines to fetch')
parser.add_argument('-s', '--scratch', action='store_true',
                    help='build pipeline data from scratch')
parser.add_argument('outfile', type=str,
                    help='name of json file to write pipeline data to')
args = parser.parse_args()

if args.tokenfile:
    gitlab_token = open(args.tokenfile, 'r').read().strip()
elif args.token:
    gitlab_token = args.token
else:
    try:
        gitlab_token = os.environ['GITLAB_TOKEN']
    except KeyError:
        print('error: gitlab token must be specified via --tokenfile (preferred), GITLAB_TOKEN environment variable or --token')
        exit(1)

def fetch_paged_json_data(url, params=None, headers=None,
                          max_count=None, stop_if=None):
    # get first page
    if args.verbose:
        print('fetching {}...'.format(url))
    r = requests.get(url, params=params, headers=headers)
    assert r.status_code == 200

    if args.verbose and ('RateLimit-Remaining' in r.headers):
        print('rate limit: {} (reset at {})'.format(r.headers['RateLimit-Remaining'],
                                                    r.headers['RateLimit-ResetTime']))

    data = r.json()

    while 'next' in r.links:
        if stop_if and stop_if(data):
            break

        if max_count and (len(data) >= max_count):
            break

        if args.delay:
            time.sleep(args.delay)

        # params are already encoded in next link
        if args.verbose:
            print('fetching {}...'.format(r.links['next']['url']))
        r = requests.get(r.links['next']['url'], headers=headers)
        assert r.status_code == 200

        data.extend(r.json())

    return data

by_id = {}
if not args.scratch:
    try:
        if args.outfile.endswith('.gz'):
            f = gzip.open(args.outfile, 'rt')
        else:
            f = open(args.outfile, 'rt')
        by_id = json.load(f)
        f.close()
        #for j in by_id.values():
        #    j.pop('jobs', None)
    except FileNotFoundError:
        pass  # no existing data - no big deal
    except:
        raise

existing_ids = set(int(i) for i in by_id.keys())

def already_seen(data):
    return (data[-1]['id'] in existing_ids)

project = requests.utils.quote(args.repo, safe='')
base_url = 'https://gitlab.com/api/v4/projects/{}/pipelines'.format(project)

pipelines = fetch_paged_json_data(base_url,
                                  params = { 'per_page': 100 },
                                  headers = { 'PRIVATE-TOKEN': gitlab_token },
                                  max_count = args.max_pipelines,
                                  stop_if = already_seen)
if not pipelines:
    print('error: no data received!?')
    exit(1)

if args.verbose:
    print('fetched {} pipelines'.format(len(pipelines)))

# add any new pipelines to our list
for p in pipelines:
    if str(p['id']) not in by_id:
        by_id[str(p['id'])] = p

# see which pipelines need jobs info
needs_jobs = [ i for i,p in by_id.items() if ('jobs' not in p) and (p.get('status', '?') in ('success', 'failed')) ]
if not args.quiet:
    print('{} total pipelines, {} require jobs info'.format(len(by_id),
                                                            len(needs_jobs)))
# work on newer pipelines first
needs_jobs.sort(reverse=True, key=int)
now = datetime.datetime.now(datetime.timezone.utc)

for i in needs_jobs:
    # chop off fractional seconds and time zone
    ctime = datetime.datetime.strptime(by_id[i]['created_at'][0:19],
                                       '%Y-%m-%dT%H:%M:%S').replace(tzinfo=datetime.timezone.utc)
                 
    days = (now - ctime).total_seconds() / 86400
    print(i, days)
    if args.max_age and (days > args.max_age):
        break  # we're sorted in descending order, so can stop here

    try:
        # enforce a delay between jobs' fetches too
        if args.delay:
            time.sleep(args.delay)

        url = 'https://gitlab.com/api/v4/projects/{}/pipelines/{}/jobs'.format(project, i)
        jdata = fetch_paged_json_data(url,
                                      params = { 'per_page': 100 },
                                      headers = { 'PRIVATE-TOKEN': gitlab_token })
        if not args.quiet:
            print('pipeline {} has {} jobs'.format(i, len(jdata)))
        jobs = []
        user = None
        commit = None
        for j in jdata:
            u = j.pop('user', None)
            if u and not user:
                user = u
            c = j.pop('commit', None)
            if c and not commit:
                commit = c
            # we don't need to remember pipeline data
            j.pop('pipeline', None)
            jobs.append(j)
        if user:
            by_id[i]['user'] = user
        if commit:
            by_id[i]['commit'] = commit
        by_id[i]['jobs'] = jobs
    except Exception as e:
        print('Exception during job fetch: {}'.format(repr(e)))
        break

try:
    if args.outfile.endswith('.gz'):
        f = gzip.open(args.outfile, 'wt')
    else:
        f = open(args.outfile, 'wt')
    json.dump(by_id, f, indent=4)
    f.close()
except:
    raise
