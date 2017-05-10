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

def quotedString(lineSplit, index):
    result = ''
    for i in range(100):
        token = lineSplit[i + index];
        if(token[0] == "\""):
            result = result + token[1:-1]
        else:
            return result;


def tokenize(lines):
    result = []
    splitters = ':;,().#'
    whitespace = ' \n\t\r'
    token = ''
    
    i = 0;
    for index in range(len(lines)):
        if lines[i] == '\"':
            for j in range(i + 1, len(lines)):
                if lines[j] == '\"':
                    token = lines[i : j + 1]
                    result.append(token)
                    token = ''
                    i = j
                    break
    
        elif lines[i] in splitters or lines[i] in whitespace:
            if token != '':
                result.append(token)
            if(lines[i] not in whitespace):
                result.append(lines[i])
            token = ''
        else:
            token = token + lines[i]
        i = i + 1
        if(i >= len(lines)):
            break
    return result;


def parseInput(multipleLines):
    tokens = tokenize(multipleLines.strip())
    
    declaration = ''
    name = ''
    code = ''
    whatToDo = ''
    type = ''
    message = ''
    
    for i in range(len(tokens)):
        token = tokens[i]
        
        if(type == '' and name != '' and token == name):
            type = tokens[i - 2]
            assert tokens[i + 1] == '.'
            assert tokens[i + 2] == 'id'
            message = quotedString(tokens, i + 6);
            break
        
        if(declaration != '' and name == ''):
            name = token
            assert tokens[i + 1] == '('
            code = tokens[i + 2]
            assert tokens[i + 3] == ','
            whatToDo = quotedString(tokens, i + 4)
        
        if(DECLARATION in token):
            declaration = token

    return declaration, name, code, whatToDo, type, message


def readMultipleLines(line, sourceFile):
    result = line
    for i in range(50):
        result = result + sourceFile.readline()
    return result



def parseSourceFile(file_name):
    with open(file_name, 'rt') as sourceFile:
        line = ' '
        while len(line) > 0:
            line = sourceFile.readline()
            if(DECLARATION in line):
                multipleLines = readMultipleLines(line, sourceFile)
                declaration, name, code, whatToDo, type, message = parseInput(multipleLines)
                #print '"' + type + '",', code, ',', message, ',', whatToDo
                print '<p><b>' + type + ' ' + code + '</b>'
                print '<br>Message: ' + message
                print '<br>Remedy: ' + whatToDo
                print '</p>'



parseSourceFile(sys.argv[1])
