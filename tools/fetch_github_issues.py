#!/usr/bin/env python3

import urllib.request
import json
import argparse
import time

parser = argparse.ArgumentParser()

parser.add_argument('-B', '--base-url',
                    default = 'https://api.github.com',
                    help = 'base URL for issues API')
parser.add_argument('-O', '--owner',
                    default = 'StanfordLegion',
                    help = 'repository owner')
parser.add_argument('-r', '--repo',
                    default = 'legion',
                    help = 'repository name')
parser.add_argument('-s', '--state',
                    default = 'open', choices = ['open' ,'closed', 'all'],
                    help = 'state of issues to fetch')
parser.add_argument('-p', '--pull-requests', action='store_true',
                    help = 'include pull requests as well as issues')
parser.add_argument('-v', '--verbose', action='store_true',
                    help = 'verbose progress information')
parser.add_argument('-q', '--quiet', action='store_true',
                    help = 'suppress all messages to stdout')
parser.add_argument('-c', '--count', type=int,
                    default = 100,
                    help = 'issues to request per page')
parser.add_argument('-R', '--max-retries', type=int,
                    default = 3,
                    help = 'maximum retries for a single request')
parser.add_argument('-d', '--retry-delay', type=int,
                    default = 3,
                    help = 'delay (in seconds) between retries')
parser.add_argument('-t', '--tokenfile', type=str,
                    help = 'file containing API authentication token')
parser.add_argument('--partial', action='store_true',
                    help = 'write partial issue list in case of errors')
parser.add_argument('output', type=str,
                    help = 'output file location')

args = parser.parse_args()

issues = {}

headers = {}
if args.tokenfile:
    token = open(args.tokenfile, 'r').read().strip()
    headers['Authorization'] = 'token ' + token

for page in range(1, 1000):
    url = '{}/repos/{}/{}/issues?state={}&count={}&page={}'.format(args.base_url,
                                                                   args.owner,
                                                                   args.repo,
                                                                   args.state,
                                                                   args.count,
                                                                   page)
    if args.verbose:
        print('fetching: {}'.format(url))

    retry_count = 0
    while True:
        try:
            req = urllib.request.Request(url, headers=headers)
            r = urllib.request.urlopen(req)
            j = json.loads(r.read().decode('utf-8'))
            break
        except KeyboardInterrupt:
            exit(1)
        except Exception as e:
            if retry_count >= args.max_retries:
                if args.partial:
                    j = []
                    break
                raise
            if not args.quiet:
                print('error: {}'.format(e))
            retry_count += 1
            if args.retry_delay > 0:
                time.sleep(args.retry_delay)

    if args.verbose:
        print('{} issues read'.format(len(j)))

    # an empty list suggests we're at the end
    if len(j) == 0:
        break
    
    for issue in j:
        num = issue['number']
        if ('pull_request' in issue) and not(args.pull_requests):
            continue
        issues[num] = issue

if not args.quiet:
    print('writing {} issues to \'{}\''.format(len(issues),
                                               args.output))

# write data out in hopefully-human-readable json
with open(args.output, 'w') as f:
    # include information used to fetch the issues
    data = { 'base_url': args.base_url,
             'owner': args.owner,
             'repo': args.repo,
             'issues': issues }
    json.dump(data, f, sort_keys=True, indent=4)
