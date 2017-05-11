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

import sqlite3
import sys

DECLARATION = "MessageDescriptor"
splitters = ':;,().#'
whitespace = ' \n\t\r'


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




def sanitizedMessage(message):
    sanitized = message
    serialNumber = 1
    index = sanitized.find("%")
    while index != -1:
        for i in range(10):
            if (index + i >= len(sanitized)) or (sanitized[index + i] in splitters) or (sanitized[index + i] in whitespace):
                replacement = "#" + str(serialNumber)
                sanitized = sanitized[:index] + replacement + sanitized[index + i:]
                serialNumber += 1
                break
        index = sanitized.find("%")
    return sanitized




def addToDatabase(type, code, message, whatToDo, connection):
    createTableCommand = "create table if not exists " + type + "(code int key, message text key, whatToDo text);"
    connection.execute(createTableCommand)
    insertCommand = "insert into " + type + "(code, message, whatToDo) values (" + code + ", \"" + sanitizedMessage(message) + "\", \"" + whatToDo + "\");"
    connection.execute(insertCommand)



def parseSourceFile(fileName, connection):
    
    with open(fileName, 'rt') as sourceFile:
        line = sourceFile.readline()
        tokens = []
        outputFiles = []
        while len(line) > 0:
            if(DECLARATION in line):
                tokenize(line, tokens)
                declaration, name, code, whatToDo, type, message, index = parseInput(sourceFile, tokens)
                tokens = tokens[index:]
                addToDatabase(type, code, message, whatToDo, connection)
                print type, code, name
            line = sourceFile.readline()


def parseSourceFiles(connection):
    filename = sys.stdin.readline().strip()
    while(len(filename) > 0):
        parseSourceFile(filename, connection)
        connection.commit()
        filename = sys.stdin.readline().strip()


def writeHtmlEntry(type, outputFile, row):
    (code, message, whatToDo) = row
    outputFile.write("<p> <b>" + type + " " + str(code) + "</b>\n")
    outputFile.write("<br>Message: " + message + "\n");
    outputFile.write("<br>Remedy:" + whatToDo + "\n</p>\n");


def writeHtmlSortedByField(field, connection):
    cursor = connection.cursor()
    cursor.execute("select name from sqlite_master where type = 'table' order by name;")
    tables = cursor.fetchall()
    for table in tables:
        tableName = table[0]
        outputFileName = tableName + "_" + field + ".html"
        outputFile = open(outputFileName, "wt")
        indexName = tableName + "_" + field + "_index"
        indexCommand = "create index " + indexName + " on " + tableName + "(" + field + ");"
        cursor.execute(indexCommand);
        selectCommand = "select * from " + tableName + " order by " + field  + ";"
        cursor.execute(selectCommand);
        row = cursor.fetchone();
        while row != None:
              writeHtmlEntry(tableName, outputFile, row)
              row = cursor.fetchone()
        outputFile.close()


def writeHtmlOutput(connection):
    writeHtmlSortedByField("code", connection)
    writeHtmlSortedByField("message", connection)


connection = sqlite3.connect(":memory:")
if(len(sys.argv) == 2):
    parseSourceFile(sys.argv[1], connection)
else:
    parseSourceFiles(connection)
writeHtmlOutput(connection)
