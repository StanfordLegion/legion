#!/usr/bin/env python3

import argparse
import json
from collections import defaultdict

parser = argparse.ArgumentParser()

parser.add_argument('-a', '--assignee', type=str,
                    help = 'restrict checks to issues assigned to specified username')
parser.add_argument('-v', '--verbose', action='store_true',
                    help = 'verbose output')
parser.add_argument('-m', '--max-list', type=int,
                    default=100,
                    help = 'maximum individual issues to list for a rule')
parser.add_argument('input', type=str,
                    help = 'location of json issues file')

args = parser.parse_args()

with open(args.input, 'r') as f:
    data = json.load(f)
if 'issues' not in data:
    data = { 'base_url': 'BASE',
             'owner': 'OWNER',
             'repo': 'REPO',
             'issues': data }

subprojects = set([ 'Legion', 'Realm', 'Regent', 'Bishop',
                    'Build', 'Tests', 'CI', 'Tools',
                    'Documentation' ])

# construct the base url that'd be used to re-query these issues
query_url = '{}/repos/{}/{}/issues'.format(data['base_url'],
                                           data['owner'],
                                           data['repo'])
if args.assignee:
    query_url += '/assigned/{}'.format(args.assignee)

# helpers to pull things out of issues in list-of-strings form
def assignees(i):
    return list(a.get('login', '??') for a in i.get('assignees', []))

def labels(i):
    return list(l.get('name', '??') for l in i.get('labels', []))

def milestones(i):
    return [ i['milestone'].get('title', '??') ] if i.get('milestone') else []
    
def check_rule(number, name, pred, only_open=True):
    failed = {}
    for issue in data['issues'].values():
        if only_open and (issue.get('state', '?') != 'open'):
            continue
        if args.assignee and (args.assignee not in assignees(issue)):
            continue
        if not pred(issue):
            failed[issue['number']] = issue
    if failed or args.verbose:
        print('Rule {}: {} violations - {}'.format(number,
                                                   len(failed),
                                                   name))
    if failed:
        by_owner = defaultdict(list)
        for num in sorted(failed):
            issue = failed[num]
            owners = assignees(issue)
            if owners:
                for o in owners:
                    by_owner[o].append(issue)
            else:
                by_owner['--unassigned--'].append(issue)
        for o in sorted(by_owner):
            print('  {}: {} issues: {}'.format(o,
                                               len(by_owner[o]),
                                               ', '.join(str(i['number']) for i in by_owner[o])))
        print('')

        if len(failed) <= args.max_list:
            for num in sorted(failed):
                issue = failed[num]
                owners = assignees(issue)
                print('  {} ({}): {}'.format(num,
                                             ','.join(sorted(assignees(issue))),
                                             issue['title']))
            print('')
    
# rule 1: all open issues must be assigned
check_rule(1, 'all open issues must be assigned',
           lambda i: i.get('assignee', None) is not None)
    
check_rule(2, 'open issues need exactly one of: planned, backlog, will-not-fix, question',
           lambda i: len(set(['planned',
                              'backlog',
                              'question',
                              'will-not-fix']).intersection(set(labels(i)))))

check_rule('3a', 'planned issues have a milestone',
           lambda i: ('planned' not in labels(i)) or milestones(i))

check_rule('3b', 'not-planned issues do not have a milestone',
           lambda i: ('planned' in labels(i)) or not(milestones(i)))

check_rule('4', 'issues have at least one sub-project label',
           lambda i: subprojects.intersection(set(labels(i))))

check_rule('6a', 'open issues need at least one of: bug, enhancement, question',
           lambda i: len(set(['bug',
                              'enhancement',
                              'question']).intersection(labels(i))) >= 1)

check_rule('6b', 'open issues need at most one of: bug, enhancement',
           lambda i: len(set(['bug',
                              'enhancement']).intersection(labels(i))) <= 1)
