#!/usr/bin/env python

# parses a GitLab CI YAML configuration file and generates a Dockerfile for
#  a specified job

import yaml
import argparse
import os

parser = argparse.ArgumentParser(description='Generate Dockerfile from GitLab CI YAML configuration file')
parser.add_argument('-b', '--branch', type=str, default='master',
                    help='branch of repository to test')
parser.add_argument('--repo', type=str,
                    default='https://gitlab.com/StanfordLegion/legion.git',
                    help='repository to clone')
parser.add_argument('-o', '--outdir', type=str,
                    help='output directory')
parser.add_argument('-k', '--keep', action='store_true',
                    help='modify test.py invocation to keep results')
parser.add_argument('-i', '--interactive', action='store_true',
                    help='run container interactively')
parser.add_argument('cfgfile', type=str,
                    help='path to GitLab CI config file')
parser.add_argument('jobname', type=str,
                    help='name of job to generate Dockerfile for')
args = parser.parse_args()

try:
    f = open(args.cfgfile, 'r')
    cfg = yaml.safe_load(f)
except:
    print('ERROR: could not read configuration file: {}'.format(args.cfgfile))
    raise

if args.jobname not in cfg:
    print('ERROR: job \'{}\' not found in configuration file'.format(args.jobname))
    exit(1)

job = cfg[args.jobname]

outdir = args.outdir or tempfile.mkdtemp()

with open(os.path.join(outdir, 'Dockerfile'), 'w') as f:
    f.write('FROM {}\n'.format(job['image']))
    f.write('SHELL [ "/bin/bash", "-c" ]\n')
    for k, v in cfg['variables'].iteritems():
        f.write('ENV {}="{}"\n'.format(k, v))
    for k, v in job.get('variables', {}).iteritems():
        f.write('ENV {}="{}"\n'.format(k, v))

    f.write('COPY script.sh .\n')

    f.write('RUN git clone -b {} {} repo\n'.format(args.branch, args.repo))
    f.write('WORKDIR "/repo"\n')
    f.write('RUN /script.sh\n')

with open(os.path.join(outdir, 'script.sh'), 'w') as f:
    if 'before_script' in cfg:
        for cmd in cfg['before_script']:
            f.write(cmd.replace('\\n','\n'))
            if not cmd.endswith('\n'):
                f.write('\n')

    for cmd in job['script']:
        if args.keep:
            cmd = cmd.replace('test.py', 'test.py --keep || /bin/true')
        f.write(cmd.replace('\\n','\n'))
        if not cmd.endswith('\n'):
            f.write('\n')
os.chmod(os.path.join(outdir, 'script.sh'), 0777)

