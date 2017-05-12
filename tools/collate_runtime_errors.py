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


def parseInput(sourceFile, tokens, lineNumber):
    
    declaration = None
    name = None
    code = None
    whatToDo = None
    type = None
    message = None
    
    i = 0;
    while i < 100:
        
        lineNumber = ensureLookahead(sourceFile, tokens, i + 20, lineNumber)
        tuple = tokens[i]
        (token, tokenLineNumber) = tuple
        
        if(type == None and name != None and token == name):
            type = tokens[i - 2][0]
            assert tokens[i + 1][0] == '.'
            assert tokens[i + 2][0] == 'id'
            message, index, lineNumber = quotedString(sourceFile, tokens, i + 6, lineNumber);
            tokens = deleteTokens(tokens, index)
            i = 0
            break
        
        if(declaration != None and name == None):
            name = token
            assert tokens[i + 1][0] == '('
            code = tokens[i + 2][0]
            assert tokens[i + 3][0] == ','
            whatToDo, index, lineNumber = quotedString(sourceFile, tokens, i + 4, lineNumber)
            tokens = deleteTokens(tokens, index)
            i = 0
            continue
        
        if(DECLARATION in token):
            declaration = token
            declarationLineNumber = tokenLineNumber
        
        i = i + 1

    if declaration == None or name == None or code == None or whatToDo == None or type == None or message == None or lineNumber == None:
        return None, None, None, None, None, None, lineNumber, None
    else:
        deleteTokens(tokens, i)
    return declaration, name, code, whatToDo, type, message, lineNumber, declarationLineNumber




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




def addToDatabase(type, name, code, message, whatToDo, connection, sourceFile, lineNumber):
    createTableCommand = "create table if not exists " + type + "(code int key, name text, message text key, whatToDo text, filename text, lineNumber int);"
    connection.execute(createTableCommand)
    filename = sourceFile.name
    insertCommand = "insert into " + type + "(code, name, message, whatToDo, filename, lineNumber) values (" + code + ", \"" + name + "\", \"" + sanitizedMessage(message) + "\", \"" + whatToDo + "\", \"" + filename + "\", " + str(lineNumber) + ");"
    connection.execute(insertCommand)



def parseSourceFile(fileName, connection):
    print fileName
    
    with open(fileName, 'rt') as sourceFile:
        lineNumber = 0
        tokens = []
        while len(tokens) > 0 or lineNumber == 0:
            lineNumber = ensureLookahead(sourceFile, tokens, 100, lineNumber)
            tuple = tokens[0]
            (token, tokenLineNumber) = tuple
            if(DECLARATION in token):
                declaration, name, code, whatToDo, type, message, lineNumber, declarationLineNumber = parseInput(sourceFile, tokens, lineNumber)
                index = 0
                if declaration != None:
                    addToDatabase(type, name, code, message, whatToDo, connection, sourceFile, declarationLineNumber)
                    print type, code, name
            else:
                del tokens[0]


def parseSourceFiles(connection):
    filename = sys.stdin.readline().strip()
    while(len(filename) > 0):
        parseSourceFile(filename, connection)
        connection.commit()
        filename = sys.stdin.readline().strip()

def htmlMarker(type, field, code):
    if code == None:
        return type + "_" + field
    else:
        return type + "_" + field + "_" + str(code)


def writeHtmlEntry(type, field, outputFile, row):
    (code, name, message, whatToDo, fileName, lineNumber) = row
    outputFile.write("<p id=\"" + htmlMarker(type, field, code) + "\">")
    outputFile.write("<b>" + type + " " + str(code) + "</b>\n")
    outputFile.write("<br>Message: " + message + "\n");
    outputFile.write("<br>Remedy:" + whatToDo + "\n");
    outputFile.write("<br>Location: line " + str(lineNumber) + " " + fileName + "\n")
    outputFile.write("<p>\n")


def tables(connection):
    cursor = connection.cursor()
    cursor.execute("select name from sqlite_master where type = 'table' order by name;")
    tables = cursor.fetchall()
    return tables


def writeHtmlSortedByField(field, connection):
    for table in tables(connection):
        tableName = table[0]
        outputFileName = htmlMarker(tableName, field, None) + ".html"
        outputFile = open(outputFileName, "wt")
        selectCommand = "select * from " + tableName + " order by " + field  + ";"
        cursor = connection.cursor()
        cursor.execute(selectCommand)
        row = cursor.fetchone()
        while row != None:
              writeHtmlEntry(tableName, field, outputFile, row)
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
        (code, name, message, whatToDo, fileName, lineNumber) = row
        if field == "code":
            text = str(code)
        elif field == "message":
            text = message[:20]
        file.write("<A href=\"" + htmlMarker(tableName, field, None) + ".html#" + htmlMarker(tableName, field, code) + "\">" + text + "</a>")
        row = cursor.fetchone()
        fieldCounter = fieldCounter + 1

    file.write("\n</TABLE>\n")
    file.write("</center>\n")
    file.write("<p></p>\n")


def writeHtmlLinks(indexFile, connection, field):
    indexFile.write("<center>")
    for table in tables(connection):
        indexFile.write("<a href=\"" + htmlMarker(table[0], field, None) + ".html\">" + table[0] + " messages ordered by " + field + "</a>&nbsp;&nbsp;&nbsp;\n")
    indexFile.write("</center>\n<p>\n")


def writeHtmlIndexes(connection):
    indexFile = open("index.html", "wt")
    writeHtmlLinks(indexFile, connection, "code");
    writeHtmlLinks(indexFile, connection, "message");
    for table in tables(connection):
        writeHtmlIndexTable(connection, table[0], "code", indexFile)
        writeHtmlIndexTable(connection, table[0], "message", indexFile)
    indexFile.close()


def writeHtmlOutput(connection):
    writeHtmlIndexes(connection);
    writeHtmlSortedByField("code", connection)
    writeHtmlSortedByField("message", connection)


connection = sqlite3.connect(":memory:")
if(len(sys.argv) == 2):
    parseSourceFile(sys.argv[1], connection)
else:
    parseSourceFiles(connection)
writeHtmlOutput(connection)
