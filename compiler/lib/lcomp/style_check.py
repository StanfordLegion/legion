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
### Style Checker
###

import copy
from . import ast

class Context:
    def __init__(self):
        self.indent = None
    def new_block_scope(self):
        cx = copy.copy(self)
        cx.indent += 2
        return cx
    def new_program_scope(self):
        cx = copy.copy(self)
        cx.indent = 0
        return cx

class StyleError (Exception):
    def __init__(self, node, message):
        Exception.__init__(self, '\n%s:\n%s' % (
                node.span, message))

# For errors reported about files rather than AST nodes, need to
# recover the error position from the absolute position in the
# file. This gets tricky because files can contain line number resets.
def FileStyleError(span, message):
    return StyleError(ast.ASTNode(span), message)

def map_position(node, lineno, colno, cx):
    resets = [n for n in node.line_resets.iterkeys() if n <= lineno]
    if len(resets) > 0:
        reset_from = max(resets)
        reset_to = node.line_resets[reset_from]
        lineno = lineno - reset_from + reset_to
    pos = ast.SpanPosition(0, lineno, colno)
    return ast.Span(node.span.source, pos, pos)

# Files should contain:
#   * spaces (no tabs)
#   * unix line endings (no \r)
#   * no trailing whitespace
def check_file(node, cx):
    lines = node.full_text.split('\n')
    for lineno, line in zip(xrange(1, len(lines) + 1), lines):
        has_tab = line.find('\t')
        if has_tab >= 0:
            span = map_position(node, lineno, has_tab, cx)
            raise FileStyleError(span, 'Whitespace: tab')
        has_carriage_return = line.find('\r')
        if has_carriage_return >= 0:
            span = map_position(node, lineno, has_carriage_return, cx)
            raise FileStyleError(span, 'Whitespace: carriage return')
        no_trailing_whitespace = line.rstrip()
        if len(no_trailing_whitespace) != len(line):
            span = map_position(node, lineno, len(no_trailing_whitespace), cx)
            raise FileStyleError(span, 'Whitespace: trailing whitespace')

# Blocks must be indented 2 spaces.
def check_indent(node, cx):
    if node.span.start.column != cx.indent:
        raise StyleError(
            node, 'Indentation: expected %s spaces but got %s' % (
                cx.indent,
                node.span.start.column))

# Struct names must be on the same line as the struct keyword. There
# must not be any space following the name and before the parameters
# (if any).
def check_struct_name(node, name, params, cx):
    if node.span.start.line != name.span.start.line:
        raise StyleError(
            name, 'Struct declaration: expected name on same line as struct keyword')
    if node.span.start.column + 7 != name.span.start.column:
        raise StyleError(
            name, 'Struct declaration: expected name to follow exactly one space after struct keyword')
    if params is not None:
        if name.span.end.line != params.span.start.line:
            raise StyleError(
                params, 'Struct declaration: expected params to start on same line as name')
        if name.span.end.column != params.span.start.column:
            raise StyleError(
                params, 'Struct declaration: expected no space between name and start of params')

# Function names must be on the same line as the task keyword. There
# must not be any space following the name and before the parameters.
def check_function_name(node, name, params, cx):
    if node.span.start.line != name.span.start.line:
        raise StyleError(
            name, 'Function declaration: expected name on same line as task keyword')
    if node.span.start.column + 5 != name.span.start.column:
        raise StyleError(
            name, 'Function declaration: expected name to follow exactly one space after task keyword')
    if name.span.end.line != params.span.start.line:
        raise StyleError(
            params, 'Function declaration: expected params to start on same line as name')
    if name.span.end.column != params.span.start.column:
        raise StyleError(
            params, 'Function declaration: expected no space between name and start of params')

# Function parameters must start on the same line as the task
# keyword. For parameters spanning multiple lines, each new line
# should align with the first parameter.
def check_function_params(node, params, cx):
    if len(params.params) > 0:
        initial_column = params.params[0].span.start.column
        line = params.params[0].span.start.line
        for param in params.params:
            if param.span.start.line > line:
                if param.span.start.column != initial_column:
                    raise StyleError(
                        param, 'Function declaration: expected multi-line params to align with first param')
                line = param.span.start.line

# Function return type must be either on the same line as the rest of
# the function declaration, or on a new line and indented.
def check_return_type(node, return_type, cx):
    if node.span.start.line >= return_type.span.end.line:
        pass
    else:
        if node.span.start.line >= return_type.span.start.line:
            raise StyleError(
                return_type, 'Function declaration: expected multi-line return type to start on next line')
        if return_type.span.start.line != return_type.declared_type.span.start.line:
            raise StyleError(
                return_type, 'Function declaration: expected return type to start on same line as colon')
        if return_type.span.start.column + 2 != return_type.declared_type.span.start.column:
            raise StyleError(
                return_type, 'Function declaration: expected return type to follow exactly one space after colon')
        check_indent(return_type, cx)

# Function privileges must be either on the same line as the rest of
# the function declaration, or on a new line and indented.
def check_privileges(node, privileges, cx):
    if node.span.start.line >= privileges.span.end.line:
        pass
    else:
        if node.span.start.line >= privileges.span.start.line:
            raise StyleError(
                privileges, 'Function declaration: expected multi-line privileges to start on next line')
        last_line = 0
        for privilege in privileges.privileges:
            if privilege.span.start.line != privilege.privilege.span.start.line:
                raise StyleError(
                    privilege, 'Function declaration: expected privilege to start on same line as comma')
            if privilege.span.start.column + 2 != privilege.privilege.span.start.column:
                raise StyleError(
                    privilege, 'Function declaration: expected privilege to follow exactly one space after comma')
            if privilege.span.start.line > last_line:
                check_indent(privilege, cx)
            last_line = privilege.span.start.line

# Definition blocks (structs, functions) must have the opening { on
# the same line if the block is a one-liner, otherwise on the next
# line.
def check_definition_block(node, block, cx):
    if block.span.start.line == block.span.end.line:
        if node.span.start.line != block.span.start.line:
            raise StyleError(
                block, 'Curly brace placement: expected empty block to start on same line')
    else:
        if node.span.start.line >= block.span.start.line:
            raise StyleError(
                block, 'Curly brace placement: expected non-empty block to start on next line')
        check_indent(block, cx)

# Inline blocks (if, else, for, while) must have the opening { on the
# same line as the end of the previous element (condition for if, for,
# while, closing } for else).
def check_inline_block(previous, block, cx):
    if previous.span.end.line != block.span.start.line:
        raise StyleError(
            block, 'Curly brace placement: expected block to start on same line')

def style_check_node(node, cx):
    if isinstance(node, ast.Program):
        cx = cx.new_program_scope()
        check_file(node, cx)
        style_check_node(node.definitions, cx)
        return
    if isinstance(node, ast.Definitions):
        for definition in node.definitions:
            style_check_node(definition, cx)
        return
    if isinstance(node, ast.Import):
        check_indent(node, cx)
        return
    if isinstance(node, ast.Struct):
        indent_cx = cx.new_block_scope()
        check_indent(node, cx)
        params = None
        if len(node.params.params) > 0:
            params = node.params
        elif len(node.regions.regions) > 0:
            params = node.regions
        check_struct_name(node, node.name, params, cx)
        check_definition_block(node, node.field_decls, cx)
        style_check_node(node.field_decls, indent_cx)
        return
    if isinstance(node, ast.FieldDecls):
        for field_decl in node.field_decls:
            style_check_node(field_decl, cx)
        return
    if isinstance(node, ast.FieldDecl):
        check_indent(node, cx)
        return
    if isinstance(node, ast.Function):
        check_indent(node, cx)
        indent_cx = cx.new_block_scope()
        check_function_name(node, node.name, node.params, cx)
        check_function_params(node, node.params, cx)
        check_return_type(node, node.return_type, indent_cx)
        check_privileges(node, node.privileges, indent_cx)
        check_definition_block(node, node.block, cx)
        style_check_node(node.block, cx)
        return
    if isinstance(node, ast.Block):
        cx = cx.new_block_scope()
        for expr in node.block:
            style_check_node(expr, cx)
        return
    if isinstance(node, ast.StatementAssert):
        check_indent(node, cx)
        return
    if isinstance(node, ast.StatementExpr):
        check_indent(node, cx)
        return
    if isinstance(node, ast.StatementIf):
        check_indent(node, cx)
        check_inline_block(node.condition, node.then_block, cx)
        style_check_node(node.then_block, cx)
        if node.else_block is not None:
            check_inline_block(node.then_block, node.else_block, cx)
            style_check_node(node.else_block, cx)
        return
    if isinstance(node, ast.StatementFor):
        check_indent(node, cx)
        check_inline_block(node.regions, node.block, cx)
        style_check_node(node.block, cx)
        return
    if isinstance(node, ast.StatementLet):
        check_indent(node, cx)
        return
    if isinstance(node, ast.StatementLetRegion):
        check_indent(node, cx)
        return
    if isinstance(node, ast.StatementLetArray):
        check_indent(node, cx)
        return
    if isinstance(node, ast.StatementLetIspace):
        check_indent(node, cx)
        return
    if isinstance(node, ast.StatementLetPartition):
        check_indent(node, cx)
        return
    if isinstance(node, ast.StatementReturn):
        check_indent(node, cx)
        return
    if isinstance(node, ast.StatementUnpack):
        check_indent(node, cx)
        return
    if isinstance(node, ast.StatementVar):
        check_indent(node, cx)
        return
    if isinstance(node, ast.StatementWhile):
        check_indent(node, cx)
        check_inline_block(node.condition, node.block, cx)
        style_check_node(node.block, cx)
        return
    raise Exception('Style checking failed at %s' % node)

def style_check(node):
    cx = Context()
    style_check_node(node, cx)
