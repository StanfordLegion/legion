#!/usr/bin/env python

# Copyright 2017 Stanford University, NVIDIA Corporation
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

import sys

DECLARATION = "MessageDescriptor"

def quotedString(sourceFile, tokens, index):
    result = ''
    for i in range(100):
        ensureLookahead(sourceFile, tokens, i + index);
        token = tokens[i + index];
        if(token[0] == "\""):
            result = result + token[1:-1]
        else:
            return result;


def tokenize(line, tokens):
    splitters = ':;,().#'
    whitespace = ' \n\t\r'
    token = ''
    
    i = 0
    for index in range(len(line)):
        if line[i] == '\"':
            for j in range(i + 1, len(line)):
                if line[j] == '\"':
                    token = line[i : j + 1]
                    tokens.append(token)
                    token = ''
                    i = j
                    break
    
        elif line[i] in splitters or line[i] in whitespace:
            if token != '':
                tokens.append(token)
            if(line[i] not in whitespace):
                tokens.append(line[i])
            token = ''
        else:
            token = token + line[i]
        i = i + 1
        if(i >= len(line)):
            break


def ensureLookahead(sourceFile, tokens, lookahead):
    while(len(tokens) <= lookahead):
        line = sourceFile.readline()
        tokenize(line, tokens)


def parseInput(sourceFile, tokens):
    
    declaration = ''
    name = ''
    code = ''
    whatToDo = ''
    type = ''
    message = ''
    
    i = 0;
    while 1:
        ensureLookahead(sourceFile, tokens, i + 6)
        token = tokens[i]
        
        if(type == '' and name != '' and token == name):
            type = tokens[i - 2]
            assert tokens[i + 1] == '.'
            assert tokens[i + 2] == 'id'
            message = quotedString(sourceFile, tokens, i + 6);
            break
        
        if(declaration != '' and name == ''):
            name = token
            assert tokens[i + 1] == '('
            code = tokens[i + 2]
            assert tokens[i + 3] == ','
            whatToDo = quotedString(sourceFile, tokens, i + 4)
        
        if(DECLARATION in token):
            declaration = token
        i = i + 1

    return declaration, name, code, whatToDo, type, message, i



def parseSourceFile(file_name):
    with open(file_name, 'rt') as sourceFile:
        line = ' '
        tokens = []
        while len(line) > 0:
            line = sourceFile.readline()
            if(DECLARATION in line):
                tokenize(line, tokens)
                declaration, name, code, whatToDo, type, message, index = parseInput(sourceFile, tokens)
                tokens = tokens[index:]
                print '<p><b>' + type + ' ' + code + '</b>'
                print '<br>Message: ' + message
                print '<br>Remedy: ' + whatToDo
                print '</p>'



parseSourceFile(sys.argv[1])
