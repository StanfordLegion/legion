#!/usr/bin/env python3

# Copyright 2022 Stanford University, NVIDIA Corporation
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

from __future__ import print_function

import argparse
import sqlite3
import sys

DECLARATIONS = [ 'REPORT_LEGION_WARNING', 'REPORT_LEGION_ERROR', 'REPORT_LEGION_FATAL' ]
splitters = ':;,().#'
whitespace = ' \n\t\r'


def quotedString(sourceFile, tokens, index, lineNumber):
    result = ''
    for i in range(100):
        lineNumber = ensureLookahead(sourceFile, tokens, i + index, lineNumber)
        tuple = tokens[i + index]
        (token, tokenLineNumber) = tuple
        if(token[0] == "\""):
            result = result + token[1 : -1]
        else:
            return result, i + index, lineNumber


def tokenize(line, tokens, lineNumber):
    token = ''
    
    i = 0
    for index in range(len(line)):
        if line[i] == '\"':
            for j in range(i + 1, len(line)):
                if line[j] == '\"':
                    token = line[i : j + 1]
                    tuple = (token, lineNumber)
                    tokens.append(tuple)
                    token = ''
                    i = j
                    break
    
        elif line[i] in splitters or line[i] in whitespace:
            if token != '':
                tuple = (token, lineNumber)
                tokens.append(tuple)
            if(line[i] not in whitespace):
                tuple = (line[i], lineNumber)
                tokens.append(tuple)
            token = ''
        else:
            token = token + line[i]
        i = i + 1
        if(i >= len(line)):
            break


def ensureLookahead(sourceFile, tokens, lookahead, lineNumber):
    line = ' '
    while(len(tokens) <= lookahead and len(line) > 0):
        line = sourceFile.readline()
        lineNumber = lineNumber + 1
        tokenize(line, tokens, lineNumber)
    return lineNumber


def deleteTokens(tokens, index):
    for i in range(index):
        del tokens[0]
    return tokens



def parseInput(sourceFile, tokens, lineNumber, enums):
    
    declaration = None
    name = None
    code = None
    type = None
    message = None
    
    i = 0;
    while i < 100:
        
        lineNumber = ensureLookahead(sourceFile, tokens, i + 5, lineNumber)
        tuple = tokens[i]
        (token, tokenLineNumber) = tuple
        
        if token in DECLARATIONS:
            declaration = token
            declarationLineNumber = tokenLineNumber
            assert tokens[i + 1][0] == '('
            name = tokens[i + 2][0]
            assert tokens[i + 3][0] == ','
            message, index, lineNumber = quotedString(sourceFile, tokens, i + 4, lineNumber)
            tokens = deleteTokens(tokens, index)
            if name in enums:
                code = enums[name]
            else:
                print('undefined declaration', name, 'missing enum in legion_config.h')
                sys.exit(1)
            type = declaration.replace('REPORT_LEGION_', '').lower()
            return declaration, name, code, type, message, lineNumber, declarationLineNumber
        else:
            i = i + 1




def sanitizedMessage(message):
    sanitized = message
    serialNumber = 1
    index = sanitized.find("%")
    while index != -1:
        for i in range(50):
            if (index + i >= len(sanitized)) or (sanitized[index + i] in splitters) or (sanitized[index + i] in whitespace):
                replacement = "#" + str(serialNumber)
                sanitized = sanitized[:index] + replacement + sanitized[index + i:]
                serialNumber += 1
                break
        index = sanitized.find("%")
    return sanitized




def addToDatabase(type, name, code, message, connection, sourceFile, lineNumber):
  
    createTableCommand = "create table if not exists " + type + "(code int key constraint c unique on conflict replace, name text, message text key);"
    connection.execute(createTableCommand)
    filename = sourceFile.name
    
    insertCommand = "insert into " + type + "(code, name, message) values (" + code + ", \"" + name + "\", \"" + sanitizedMessage(message) + "\");"
    print(type, name, 'code', code, filename + ":" + str(lineNumber))
    connection.execute(insertCommand)
    
    createInstancesTableCommand = "create table if not exists instances(code int key, sourceFile text, lineNumber int);"
    connection.execute(createInstancesTableCommand)

    insertInstanceCommand = "insert into instances (code, sourceFile, lineNumber) values (" + code + ", \"" + filename + "\", " + str(lineNumber) + ");"
    connection.execute(insertInstanceCommand)



def parseSourceFile(fileName, connection, enums):
    print(fileName)
    
    with open(fileName, 'rt') as sourceFile:
        lineNumber = 0
        tokens = []
        while len(tokens) > 0 or lineNumber == 0:
            lineNumber = ensureLookahead(sourceFile, tokens, 100, lineNumber)
            tuple = tokens[0]
            (token, tokenLineNumber) = tuple
            if(token in DECLARATIONS):
                declaration, name, code, type, message, lineNumber, declarationLineNumber = parseInput(sourceFile, tokens, lineNumber, enums)
                index = 0
                if declaration != None:
                    addToDatabase(type, name, code, message, connection, sourceFile, declarationLineNumber)
            else:
                del tokens[0]


def parseSourceFiles(connection, enums):
    fileName = sys.stdin.readline().strip()
    while(len(fileName) > 0):
        fileName = fileName.replace("//", "/")
        parseSourceFile(fileName, connection, enums)
        connection.commit()
        fileName = sys.stdin.readline().strip()

def htmlMarker(type, field, code):
    if code == None:
        return type + "_" + field
    else:
        return type + "_" + field + "_" + str(code)


def fileName_(fileName):
    splitFileName = fileName.split('/')
    if len(splitFileName) > 0:
        fileName = splitFileName[-1:][0]
    return fileName


def prefixedPath(fileName, prefix, lineNumber, strip):
    fileName = fileName.replace("//", "/")
    if strip > 0:
        splitFileName = fileName.split('/')
        fileName = "/".join(splitFileName[strip:])
    if prefix == None:
        return fileName + '#L' + str(lineNumber)
    result = prefix + '/' + fileName + '#L' + str(lineNumber)
    return result



def writeHtmlEntry(type, field, outputFile, row, hrefPrefix, strip):
    (code, name, message) = row
    outputFile.write("<p id=\"" + htmlMarker(type, field, code) + "\">")
    outputFile.write("<b>" + type + " " + str(code) + "</b>\n")
    outputFile.write("<br>Message: " + message + "\n");
    
    selectInstancesCommand = "select sourceFile, lineNumber from instances where code = " + str(code) + " order by sourceFile, lineNumber;"
    cursor = connection.cursor()
    cursor.execute(selectInstancesCommand)
    instance = cursor.fetchone()
    while instance != None:
        (sourceFile, lineNumber) = instance
        href = "<a href=\"" + prefixedPath(sourceFile, hrefPrefix, lineNumber, strip) + "\">" + fileName_(sourceFile) + " line " + str(lineNumber) + "</a>"
        outputFile.write("<br>" + href + '\n')
        instance = cursor.fetchone()
    outputFile.write("<p>\n")

    outputFile.write("<br>")
    outputFile.write("<a href=\"index.html\">back to message index</a>\n")


def tables(connection):
    cursor = connection.cursor()
    cursor.execute("select name from sqlite_master where type = 'table' order by name;")
    tables = cursor.fetchall()
    return tables


def writeHtmlSortedByField(field, connection, hrefPrefix, strip, outputDir):
    for table in tables(connection):
        tableName = table[0]
        if tableName != 'instances':
            outputFileName = htmlMarker(tableName, field, None) + ".html"
            outputFile = open(outputDir + '/' + outputFileName, "wt")
            outputFile.write("---\nlayout: page\n---\n\n")
            selectCommand = "select * from " + tableName + " order by " + field  + ";"
            cursor = connection.cursor()
            cursor.execute(selectCommand)
            row = cursor.fetchone()
            while row != None:
                writeHtmlEntry(tableName, field, outputFile, row, hrefPrefix, strip)
                row = cursor.fetchone()
            outputFile.close()


def writeHtmlIndexTable(connection, tableName, field, file):
    file.write("<center>")
    file.write("<TABLE border=\"1\">\n")
    file.write("<CAPTION>" + tableName + " messages ordered by " + field + "</CAPTION>\n")
    file.write("<TR>")

    if field == "code":
        maxFieldCounter = 20
    elif field == "message":
        maxFieldCounter = 8
    else:
        maxFieldCounter = 12;

    selectCommand = "select * from " + tableName + " order by " + field  + ";"
    cursor = connection.cursor()
    cursor.execute(selectCommand)
    row = cursor.fetchone()

    fieldCounter = 0
    while row != None:
        if fieldCounter == maxFieldCounter:
            file.write("<TR>")
            fieldCounter = 0
        file.write("<TD>")
        first = False
        (code, name, message) = row
        if field == "code":
            text = str(code)
        elif field == "message":
            text = message[:20]
        file.write("<A href=\"" + htmlMarker(tableName, field, None) + ".html#" + htmlMarker(tableName, field, code) + "\">" + text + "</a>\n")
        row = cursor.fetchone()
        fieldCounter = fieldCounter + 1

    file.write("\n</TABLE>\n")
    file.write("</center>\n")
    file.write("<p></p>\n")


def writeHtmlLinks(indexFile, connection, field):
    indexFile.write("<center>")
    for table in tables(connection):
        if table[0] != 'instances':
            indexFile.write("<a href=\"" + htmlMarker(table[0], field, None) + ".html\">" + table[0] + " messages ordered by " + field + "</a>&nbsp;&nbsp;&nbsp;\n")
    indexFile.write("</center>\n<p>\n")


def writeHtmlIndexes(connection, outputDir):
    indexFile = open(outputDir + '/' + "index.html", "wt")
    indexFile.write("---\nlayout: page\n---\n\n")
    writeHtmlLinks(indexFile, connection, "code");
    writeHtmlLinks(indexFile, connection, "message")
    for table in tables(connection):
        if table[0] != 'instances':
            writeHtmlIndexTable(connection, table[0], "code", indexFile)
            writeHtmlIndexTable(connection, table[0], "message", indexFile)
    indexFile.close()


def writeHtmlOutput(connection, hrefPrefix, strip, outputDir):
    writeHtmlIndexes(connection, outputDir);
    writeHtmlSortedByField("code", connection, hrefPrefix, strip, outputDir)
    writeHtmlSortedByField("message", connection, hrefPrefix, strip, outputDir)


def loadEnums(file):
    enums = {}
    strings = file.readlines()
    sawTypedef = False
    sawEnd = False
    for string in strings:
        tokens = string.strip().split(' ')
        if len(tokens) >= 3:
            if tokens[0] == 'typedef' and tokens[1] == 'enum' and tokens[2] == 'legion_error_t':
                sawTypedef = True
                if sawTypedef and tokens[0] == '}' and tokens[2] == 'legion_error_t;':
                    sawEnd = True
        if sawTypedef and not sawEnd and len(tokens) >= 3 and tokens[1] == '=' :
            name = tokens[0]
            value = tokens[2].replace(',', '')
            enums[name] = value
    return enums


def loadMessageNames(file):
    result = []
    strings = file.readlines()
    for string in strings:
        result.append(string.strip())
    return result


class MyParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        print('error', message)
        sys.exit(2)
parser = MyParser(description = 'Legion Tools: collate messages')
parser.add_argument('--prefix', dest='prefix', action='store', help='path prefix to source files', default='')
parser.add_argument('--strip', dest='strip', action='store', type=int, help='num dirs to strip from sourcfile path', default=0)
parser.add_argument('--output_dir', dest='outputDir', action='store', help='dir to write the output files to', default='.')
parser.add_argument('--legion_config_h', dest='legionConfigH', action='store', help='path to legion_config.h', required=True)

args = parser.parse_args()

print('this program parses a list of files.  it reads the filenames from stdin and writes html files as output.')
print('example: find . -name *.cc | python3 collate_messages.py --glossaryFile=glossaryFile.txt --glossaryURL="http://legion.stanford.edu"')

legionConfigFile = open(args.legionConfigH, 'r')
if legionConfigFile == None:
    print('invalid legion config file path', legionConfigH)
    sys.exit(-1)

enums = loadEnums(legionConfigFile)
connection = sqlite3.connect(":memory:")
parseSourceFiles(connection, enums)

writeHtmlOutput(connection, args.prefix, args.strip, args.outputDir)
