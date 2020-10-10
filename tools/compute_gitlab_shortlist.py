#!/usr/bin/env python3

import os
import sys
import argparse
import requests
import json
import math
import gzip
import datetime
from collections import defaultdict

parser = argparse.ArgumentParser(description = 'GitLab-CI shortlist generator')
parser.add_argument('-v', '--verbose',
                    action='store_true',
                    default=False,
                    help='print lots of information')
parser.add_argument('-q', '--quiet',
                    action='store_true',
                    default=False,
                    help='suppress all normal output')
parser.add_argument('-M', '--max_age', type=int,
                    help='age limit (in days) of pipelines to consider')
parser.add_argument('-d', '--decay', type=int,
                    help='half-life (in days) of weighting for pipelines')
parser.add_argument('infile', type=str,
                    help='name of json file to read pipeline data from')
args = parser.parse_args()

if args.infile.endswith('.gz'):
    f = gzip.open(args.infile, 'rt')
else:
    f = open(args.infile, 'rt')
by_id = json.load(f)
f.close()

ids = list(i for i,p in by_id.items() if ('jobs' in p))
ids.sort(reverse=True, key=int)

now = datetime.datetime.now(datetime.timezone.utc)

jobstats = dict()
total_pass_weight = 0
total_pass_count = 0
total_fail_weight = 0
total_fail_count = 0
fail_weights = dict()

print('{} pipelines with job data read'.format(len(ids)))
for i in ids:
    p = by_id[i]
    # chop off fractional seconds and time zone
    ctime = datetime.datetime.strptime(p['created_at'][0:19],
                                       '%Y-%m-%dT%H:%M:%S').replace(tzinfo=datetime.timezone.utc)
                                       
    days = (now - ctime).total_seconds() / 86400
    if args.max_age and (days > args.max_age):
        continue
    if args.decay:
        weight = 0.5 ** (days / args.decay)
    else:
        weight = 1.0
    #print(p['status'], p['created_at'], days, weight)

    if p['status'] == 'success':
        total_pass_weight += weight
        total_pass_count += 1
    if p['status'] == 'failed':
        total_fail_weight += weight
        total_fail_count += 1

    job_pass_count = defaultdict(int)
    for j in p['jobs']:
        #print('  ', j['status'], j['name'], j['duration'])
        try:
            js = jobstats[j['name']]
        except KeyError:
            js = { 'pass_times': [],
                   'fails': set() }
            jobstats[j['name']] = js

        if j['status'] == 'success':
            js['pass_times'].append((weight, j['duration']))
            job_pass_count[j['name']] += 1

        if j['status'] == 'failed':
            job_pass_count[j['name']] += 0

    # if the overall pipeline failed, any job that didn't pass once (or more)
    #  is considered to have detected the failure
    if p['status'] == 'failed':
        fail_weights[p['id']] = weight
        for k,v in job_pass_count.items():
            if not v:
                jobstats[k]['fails'].add(p['id'])

#print(total_pass_weight, total_pass_count, total_fail_weight, total_fail_count)

# compute weighted mean and variance for job execution time (when it passes)
for n,js in jobstats.items():
    if not js['pass_times']:
        continue
    total_weight = sum(w for w,v in js['pass_times'])
    weighted_mean = sum(w*v for w,v in js['pass_times']) / total_weight
    weighted_var = math.sqrt(sum(w*(v-weighted_mean)**2 for w,v in js['pass_times']) /
                             total_weight)
    js['pass_mean'] = weighted_mean
    js['pass_var'] = weighted_var
    #print("{:50.50s} {:6.1f}+{:5.1f}".format(n, weighted_mean, weighted_var))

# greedily build shortlist by choosing one test at a time that maximizes
#  incremental coverage
shortlist = []
shortlist_time = 0
covered = set()
covered_weight = 0
print('{} pipeline failures found'.format(len(fail_weights)))
print('Cum.  Cum.')
print('Cov%  cpu-hr   Job Name')
for _ in range(50):
    inc_cov = []
    for n,js in jobstats.items():
        # a test that's never failed is clearly not useful
        if not js['fails']:
            continue
        # a test that has never passed isn't useful either
        if not js['pass_times']:
            continue
        detected_weight = sum(fail_weights[i] for i in js['fails'] - covered)
        if detected_weight:
            # penalize high-variance tasks by using the +1 sigma time
            inc_cov.append((detected_weight, 
                            js['pass_mean'] + js['pass_var'],
                            n))
    if not inc_cov:
        break
    inc_cov.sort(key=lambda x: (-x[0], x[1]))
    #print(total_fail_weight, covered_weight, inc_cov[0:10])
    chosen = inc_cov[0][2]
    shortlist.append(chosen)
    js = jobstats[chosen]
    shortlist_time += inc_cov[0][1]
    covered.update(js['fails'])
    covered_weight += inc_cov[0][0]
    print('{:5.2f}  {:5.1f}   {}'.format(100.0 * covered_weight / total_fail_weight,
                                         shortlist_time / 3600.0,
                                         chosen))
#print(set(fail_weights.keys()) - covered)
