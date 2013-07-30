#!/usr/bin/env python

# Copyright 2013 Stanford University and Los Alamos National Security, LLC
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
### Lexer
###

import ply, sys
from ply.lex import TOKEN

class Lexer:
    states = (('type', 'exclusive'),)

    keywords = ('aliased', 'array', 'as', 'assert', 'bool', 'color',
                'coloring', 'disjoint', 'double', 'downregion',
                'else', 'false', 'float', 'for', 'if', 'import', 'in',
                'int', 'int8', 'int16', 'int32', 'int64', 'isnull',
                'ispace', 'let', 'new', 'null', 'pack', 'partition',
                'read', 'reads', 'reduce', 'reduces', 'region',
                'return', 'struct', 'task', 'true', 'uint', 'uint8',
                'uint16', 'uint32', 'uint64', 'unpack', 'upregion',
                'var', 'void', 'while', 'write', 'writes', 'zip')

    tokens = [
        'ARROW', 'BITWISE_AND_EQUALS', 'BITWISE_OR_EQUALS',
        'BITWISE_XOR_EQUALS', 'DIVIDE_EQUALS', 'DOUBLE_VALUE',
        'EQUALS', 'GREATER_EQUALS', 'HEX_VALUE', 'ID', 'INT_VALUE',
        'LEFT_SHIFT', 'LEFT_SHIFT_EQUALS', 'LESS_EQUALS',
        'LOGICAL_AND', 'LOGICAL_OR', 'MINUS_EQUALS', 'MODULUS_EQUALS',
        'NOT_EQUALS', 'PLUS_EQUALS', 'RIGHT_SHIFT',
        'RIGHT_SHIFT_EQUALS', 'STRING_VALUE', 'TIMES_EQUALS',
        'UHEX_VALUE', 'UINT_VALUE',
        ] + [word.upper() for word in keywords]

    literals = '!%&()*+,-./:;<=>?@[]^_{|}~'

    whitespace = ' \t\r\f\v'

    identifier = r'([A-Za-z][A-Za-z0-9_]*)'

    t_ANY_ignore = whitespace

    def t_ANY_ignore_multiline_comment(self, t):
        r'/\*([^*]+|(\*[^\/]))*\*/'
        t.lexer.lineno += t.value.count('\n')
        last_newline = t.value.rfind('\n')
        if last_newline > 0:
            self.line_start = t.lexpos + last_newline + len('\n')
        pass

    def t_ANY_ignore_singleline_comment(self, t):
        r'//.*'
        pass

    re_whitespace = r' \t\r\f\v'
    start_line = r'(?:^|(?<=\n))'
    end_line = r'(?:$|(?=\n))'
    reset_line = r'%s\#[%s]*line[%s]+(?P<line>[0-9]+)[%s]*%s' % (
        start_line, re_whitespace, re_whitespace, re_whitespace, end_line)
    @TOKEN(reset_line)
    def t_reset_line(self, t):
        t.lexer.lineno = int(self.lexer.lexmatch.group('line')) - 1

    def t_ANY_ignore_newlines(self, t):
        r'\n+'
        t.lexer.lineno += len(t.value)
        self.line_start = t.lexpos + len(t.value)

    @TOKEN(identifier)
    def t_ANY_ID(self, t):
        if t.value in self.keywords:
            t.type = t.value.upper()
        return t

    def t_LESS_EQUALS(self, t):
        r'<='
        return t

    def t_GREATER_EQUALS(self, t):
        r'>='
        return t

    def t_EQUALS(self, t):
        r'=='
        return t

    def t_NOT_EQUALS(self, t):
        r'!='
        return t

    def t_TIMES_EQUALS(self, t):
        r'\*='
        return t

    def t_DIVIDE_EQUALS(self, t):
        r'/='
        return t

    def t_MODULUS_EQUALS(self, t):
        r'%='
        return t

    def t_PLUS_EQUALS(self, t):
        r'\+='
        return t

    def t_MINUS_EQUALS(self, t):
        r'-='
        return t

    def t_RIGHT_SHIFT_EQUALS(self, t):
        r'>>='
        return t

    def t_LEFT_SHIFT_EQUALS(self, t):
        r'<<='
        return t

    def t_BITWISE_AND_EQUALS(self, t):
        r'&='
        return t

    def t_BITWISE_XOR_EQUALS(self, t):
        r'\^='
        return t

    def t_BITWISE_OR_EQUALS(self, t):
        r'\|='
        return t

    def t_LOGICAL_AND(self, t):
        r'&&'
        return t

    def t_LOGICAL_OR(self, t):
        r'\|\|'
        return t

    def t_ARROW(self, t):
        r'->'
        return t

    def t_RIGHT_SHIFT(self, t):
        r'>>'
        return t

    def t_LEFT_SHIFT(self, t):
        r'<<'
        return t

    def t_DOUBLE_VALUE(self, t):
        r'[0-9]+\.[0-9]*'
        return t

    def t_UHEX_VALUE(self, t):
        r'0[xX][0-9a-fA-F]+[uU]'
        return t

    def t_UINT_VALUE(self, t):
        r'[0-9]+[uU]'
        return t

    def t_HEX_VALUE(self, t):
        r'0[xX][0-9a-fA-F]+'
        return t

    def t_INT_VALUE(self, t):
        r'[0-9]+'
        return t

    def t_STRING_VALUE(self, t):
        r'"([^"\n]|\")*"'
        return t

    def t_ANY_error(self, t):
        raise ply.lex.LexError('Illegal character on line ' + str(t.lexer.lineno) + ': ' + t.value, t.value)

    def __init__(self, source = None, **kwargs):
        self.lexer = ply.lex.lex(module = self, **kwargs)
        self.line_start = 0
        if source is not None:
            self.input(source)

    def input(self, source):
        self.source = source.name if hasattr(source, 'name') else '<unknown>'
        self.lexer.input(source.read())

    def token(self, **kwargs):
        t = self.lexer.token(**kwargs)
        if t is not None:
            t.column = t.lexpos - self.line_start
            t.length = len(t.value)
        return t

    def __iter__(self):
        while True:
            t = self.token()
            if t is not None:
                yield t
            else:
                return
