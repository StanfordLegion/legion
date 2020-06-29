#!/usr/bin/env python3

import json
import argparse
import csv
import sys

parser = argparse.ArgumentParser()

parser.add_argument('-s', '--state',
                    default = 'open', choices = ['open' ,'closed', 'all'],
                    help = 'state of issues to fetch')
parser.add_argument('-l', '--label',
                    help = 'comma-separated labels to filter on')
parser.add_argument('-m', '--milestone',
                    help = 'comma-separated milestones to filter on')
parser.add_argument('-c', '--columns', default='nalmt',
                    help = 'columns to include in csv output')
parser.add_argument('-o', '--output', type=str,
                    help = 'output file location')
parser.add_argument('input', type=str,
                    help = 'location of input json data')

args = parser.parse_args()

with open(args.input, 'r') as f:
    data = json.load(f)

outfile = open(args.output, 'w') if args.output else sys.stdout
cw = csv.writer(outfile)

header_names = { 'n': 'Issue', #number
                 'a': 'Assignees',
                 'l': 'Labels',
                 'm': 'Milestone',
                 't': 'Title',
                 }

cw.writerow(header_names.get(c, c) for c in args.columns)

# helpers to pull things out of issues in list-of-strings form
def assignees(i):
    return list(a.get('login', '??') for a in i.get('assignees', []))

def labels(i):
    return list(l.get('name', '??') for l in i.get('labels', []))

def milestones(i):
    return [ i['milestone'].get('title', '??') ] if i.get('milestone') else []

col_values = { 'n': lambda i: str(i['number']),
               'a': lambda i: ','.join(assignees(i)),
               'l': lambda i: ','.join(labels(i)),
               'm': lambda i: ','.join(milestones(i)),
               't': lambda i: i['title'],
               }

def list_match(mlist, vlist):
    first = True
    accept = True
    for mterm in mlist.split(','):
        if mterm.startswith('-'):
            if mterm[1:] in vlist:
                accept = False
        else:
            if first:
                accept = False  # default is reject
            if mterm in vlist:
                accept = True
        first = False
    return accept

for num in sorted(data['issues']):
    i = data['issues'][num]

    if args.state and (i['state'] != args.state):
        continue

    if args.label and not list_match(args.label, labels(i)):
        continue

    if args.milestone and not list_match(args.milestone, milestones(i)):
        continue

    cw.writerow(col_values[c](i) for c in args.columns)

if args.output:
    outfile.close()
