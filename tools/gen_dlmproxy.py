#!/usr/bin/python

import subprocess
import argparse
import sys
import re

proxied_fns = {}

filename = sys.argv[1]
for l in subprocess.check_output(['readelf', '-W', '--dyn-syms', filename]).splitlines():
    cols = l.split()
    if len(cols) < 8:
        continue
    # skip header
    if cols[0] == 'Num:':
        continue
    # skip undefined and absolute symbols
    if cols[6] in ('UND', 'ABS'):
        continue
    # don't have a plan for non-function symbols
    if cols[3] != 'FUNC':
        print 'HELP! non-function symbol:\n', l
        assert False
        continue
    fnname = cols[7]
    if '@' in fnname:
        s = fnname.split('@')
        fnname = s[0]
        symver = s[-1]
    else:
        symver = None
    info = dict(fname = fnname,
                symver = symver)
    proxied_fns[fnname] = info

#print '''
##include <dlfcn.h>
#''';
print '''
#include <stdio.h>
''';

for fn in proxied_fns.itervalues():
    print '''
static void *real_{fname} __attribute__((nocommon)) = 0;
'''.format(**fn)

print '''
void dlmproxy_load_symbols(void *(*lookupfn)(const char *, const char *))
{
''';
for fn in proxied_fns.itervalues():
    if fn['symver'] is None:
        lookupsym = '0'
    else:
        lookupsym = '"' + fn['symver'] + '"'
    print '''
  if(real_{fname} == 0)
    real_{fname} = lookupfn("{fname}", {lookupsym});
'''.format(lookupsym = lookupsym,
           **fn)
print '''
}
'''

for fn in proxied_fns.itervalues():
    print '''
asm(".text\\n"
'''
    if '@' in fn['fname']:
        print '''
    ".globl {fname}\\n"
    ".symver {fname}, {fname}\\n"
'''.format(**fn)
    else:
        print '''
    ".globl {fname}\\n"
'''.format(**fn)
    print '''
    ".type {fname}, @function\\n"
    "{fname}:\\n"
    "  pushq %rax\\n"
    "  movq real_{fname}@GOTPCREL(%rip), %rax\\n"
    "  movq (%rax), %rax\\n"
    "  xchgq %rax, 0(%rsp)\\n"
    "  retq");
'''.format(**fn)
