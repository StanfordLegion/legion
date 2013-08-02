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
### Parser
###

import os, ply, ply.yacc, sys
from . import ast, lex

class ParseError (Exception):
    def __init__(self, source, token):
        Exception.__init__(self, '\n%s:%s:\nParsing failed at %s' % (
                source, ast.SpanPosition.from_token_start(token), token.value))

class Parser:
    tokens = lex.Lexer.tokens

    start = 'program'

    precedence = (
        ('right', '=', 'TIMES_EQUALS', 'DIVIDE_EQUALS',
         'MODULUS_EQUALS', 'PLUS_EQUALS', 'MINUS_EQUALS',
         'RIGHT_SHIFT_EQUALS', 'LEFT_SHIFT_EQUALS',
         'BITWISE_AND_EQUALS', 'BITWISE_XOR_EQUALS',
         'BITWISE_OR_EQUALS'),
        ('left', 'LOGICAL_OR'),
        ('left', 'LOGICAL_AND'),
        ('left', '|'),
        ('left', '^'),
        ('left', '&'),
        ('left', 'EQUALS', 'NOT_EQUALS'),
        ('left', '<', 'LESS_EQUALS', '>', 'GREATER_EQUALS'),
        ('left', 'RIGHT_SHIFT', 'LEFT_SHIFT'),
        ('left', '+', '-'),
        ('left', '*', '/', '%'),
        ('right', '!', '~'),
        ('left', '.', 'ARROW'),
        ('left', '(', ')', '{', '}'),
        )

    def p_program_empty(self, p):
        'program : '
        p[0] = ast.Program(
            span = ast.Span.from_slice(p))

    def p_program_def(self, p):
        'program : program def'
        p[0] = ast.Program(
            span = ast.Span.from_slice(p),
            program = p[1], definition = p[2])

    def p_import(self, p):
        'def : IMPORT STRING_VALUE ";"'
        p[0] = ast.Import(
            span = ast.Span.from_slice(p),
            filename = p[2][1:-1])

    def p_struct(self, p):
        'def : STRUCT ID struct_params struct_regions struct_constraints field_decls'
        p[0] = ast.Struct(
            span = ast.Span.from_slice(p),
            name = p[2], params = p[3], regions = p[4], constraints = p[5],
            field_decls = p[6])

    def p_struct_params_empty(self, p):
        'struct_params : '
        p[0] = ast.StructParams(
            span = ast.Span.from_slice(p))

    def p_struct_params_nonempty(self, p):
        'struct_params : "<" struct_params_list ">"'
        p[0] = p[2]

    def p_struct_params_list_single_param(self, p):
        'struct_params_list : struct_param'
        p[0] = ast.StructParams(
            span = ast.Span.from_slice(p),
            param = p[1])

    def p_struct_params_list_additional_param(self, p):
        'struct_params_list : struct_params_list "," struct_param'
        p[0] = ast.StructParams(
            span = ast.Span.from_slice(p),
            params = p[1], param = p[3])

    def p_struct_param(self, p):
        'struct_param : ID ":" type_region_kind'
        p[0] = ast.StructParam(
            span = ast.Span.from_slice(p),
            name = p[1], type = p[3])

    def p_struct_regions_empty(self, p):
        'struct_regions : '
        p[0] = ast.StructRegions(
            span = ast.Span.from_slice(p))

    def p_struct_regions_nonempty(self, p):
        'struct_regions : "[" struct_regions_list "]"'
        p[0] = p[2]

    def p_struct_regions_list_single_region(self, p):
        'struct_regions_list : struct_region'
        p[0] = ast.StructRegions(
            span = ast.Span.from_slice(p),
            region = p[1])

    def p_struct_regions_list_additional_region(self, p):
        'struct_regions_list : struct_regions_list "," struct_region'
        p[0] = ast.StructRegions(
            span = ast.Span.from_slice(p),
            regions = p[1], region = p[3])

    def p_struct_region(self, p):
        'struct_region : ID ":" type_region_kind'
        p[0] = ast.StructRegion(
            span = ast.Span.from_slice(p),
            name = p[1], type = p[3])

    def p_struct_constraints_empty(self, p):
        'struct_constraints : '
        p[0] = ast.StructConstraints(
            span = ast.Span.from_slice(p))

    def p_struct_constraints_nonempty(self, p):
        'struct_constraints : "," struct_constraints_list'
        p[0] = p[2]

    def p_struct_constraints_list_single_constraint(self, p):
        'struct_constraints_list : struct_constraint'
        p[0] = ast.StructConstraints(
            span = ast.Span.from_slice(p),
            constraint = p[1])

    def p_struct_constraints_list_additional_constraint(self, p):
        'struct_constraints_list : struct_constraints_list "," struct_constraint'
        p[0] = ast.StructConstraints(
            span = ast.Span.from_slice(p),
            constraints = p[1], constraint = p[3])

    def p_struct_constraint(self, p):
        '''struct_constraint : struct_constraint_region LESS_EQUALS struct_constraint_region
                             | struct_constraint_region "*" struct_constraint_region'''
        p[0] = ast.StructConstraint(
            span = ast.Span.from_slice(p),
            op = p[2], lhs = p[1], rhs = p[3])

    def p_struct_constraint_region(self, p):
        'struct_constraint_region : ID'
        p[0] = ast.StructConstraintRegion(
            span = ast.Span.from_slice(p),
            name = p[1])

    def p_field_decls_empty(self, p):
        'field_decls : "{" "}"'
        p[0] = ast.FieldDecls(
            span = ast.Span.from_slice(p))

    def p_field_decls_nonempty(self, p):
        'field_decls : "{" field_decls_list optional_comma "}"'
        p[0] = p[2]

    def p_field_decls_list_single_field(self, p):
        'field_decls_list : field_decl'
        p[0] = ast.FieldDecls(
            span = ast.Span.from_slice(p),
            field_decl = p[1])

    def p_field_decls_list_additional_fields(self, p):
        'field_decls_list : field_decls_list "," field_decl'
        p[0] = ast.FieldDecls(
            span = ast.Span.from_slice(p),
            field_decls = p[1], field_decl = p[3])

    def p_field_decl(self, p):
        'field_decl : ID ":" type_nonvoid'
        p[0] = ast.FieldDecl(
            span = ast.Span.from_slice(p),
            name  = p[1], field_type = p[3])

    def p_optional_comma_emtpy(self, p):
        'optional_comma :'
        pass

    def p_optional_comma_comma(self, p):
        'optional_comma : ","'
        pass

    def p_function(self, p):
        'def : TASK ID params return_type privileges block'
        p[0] = ast.Function(
            span = ast.Span.from_slice(p),
            name = p[2], params = p[3],
            return_type = p[4], privileges = p[5],
            block = p[6])

    def p_params_empty(self, p):
        'params : "(" ")"'
        p[0] = ast.Params(
            span = ast.Span.from_slice(p))

    def p_params_nonempty(self, p):
        'params : "(" params_list ")"'
        p[0] = p[2]

    def p_params_list_single_param(self, p):
        'params_list : param'
        p[0] = ast.Params(
            span = ast.Span.from_slice(p),
            param = p[1])

    def p_params_list_additional_param(self, p):
        'params_list : params_list "," param'
        p[0] = ast.Params(
            span = ast.Span.from_slice(p),
            params = p[1], param = p[3])

    def p_param(self, p):
        'param : ID ":" param_type'
        p[0] = ast.Param(
            span = ast.Span.from_slice(p),
            name = p[1], declared_type = p[3])

    def p_param_type(self, p):
        '''param_type : type_nonvoid
                      | type_region_kind
                      | type_array_kind
                      | type_ispace_kind'''
        p[0] = p[1]

    def p_privileges_empty(self, p):
        'privileges : '
        p[0] = ast.Privileges(
            span = ast.Span.from_slice(p))

    def p_privileges_nonempty(self, p):
        'privileges : "," privileges_list'
        p[0] = p[2]

    def p_privileges_list_single_privilege(self, p):
        'privileges_list : privilege'
        p[0] = ast.Privileges(
            span = ast.Span.from_slice(p),
            privilege = p[1])

    def p_privileges_list_additional_privilege(self, p):
        'privileges_list : privileges_list "," privilege'
        p[0] = ast.Privileges(
            span = ast.Span.from_slice(p),
            privileges = p[1], privilege = p[3])

    def p_privilege_read_write(self, p):
        '''privilege : READS privilege_regions
                     | WRITES privilege_regions'''
        p[0] = ast.Privilege(
            span = ast.Span.from_slice(p),
            privilege = p[1], regions = p[2])

    def p_privilege_reduce(self, p):
        '''privilege : REDUCES reduce_op privilege_regions'''
        p[0] = ast.Privilege(
            span = ast.Span.from_slice(p),
            privilege = p[1], regions = p[3], op = p[2])

    def p_privilege_regions(self, p):
        'privilege_regions :  "(" privilege_regions_list ")"'
        p[0] = p[2]

    def p_privilege_regions_list_single_region(self, p):
        'privilege_regions_list : privilege_region'
        p[0] = ast.PrivilegeRegions(
            span = ast.Span.from_slice(p),
            region = p[1])

    def p_privilege_regions_list_additional_region(self, p):
        'privilege_regions_list : privilege_regions_list "," privilege_region'
        p[0] = ast.PrivilegeRegions(
            span = ast.Span.from_slice(p),
            regions = p[1],
            region = p[3])

    def p_privilege_region(self, p):
        'privilege_region : ID'
        p[0] = ast.PrivilegeRegion(
            span = ast.Span.from_slice(p),
            name = p[1],
            fields = ast.PrivilegeRegionFields(
                span = ast.Span.from_slice(p)))

    def p_privilege_region_with_field(self, p):
        'privilege_region : ID "." privilege_region_field_top'
        p[0] = ast.PrivilegeRegion(
            span = ast.Span.from_slice(p),
            name = p[1],
            fields = p[3])

    def p_privilege_region_top_field_single(self, p):
        'privilege_region_field_top : privilege_region_field'
        p[0] = ast.PrivilegeRegionFields(
                span = ast.Span.from_slice(p),
                field = p[1])

    def p_privilege_region_top_field_multiple(self, p):
        'privilege_region_field_top : privilege_region_field_multiple'
        p[0] = p[1]

    def p_privilege_region_field_single(self, p):
        'privilege_region_field : ID'
        p[0] = ast.PrivilegeRegionField(
            span = ast.Span.from_slice(p),
            name = p[1],
            fields = ast.PrivilegeRegionFields(
                span = ast.Span.from_slice(p)))

    def p_privilege_region_field_nested(self, p):
        'privilege_region_field : ID "." privilege_region_field_top'
        p[0] = ast.PrivilegeRegionField(
            span = ast.Span.from_slice(p),
            name = p[1],
            fields = p[3])

    def p_privilege_region_field_multiple(self, p):
        'privilege_region_field_multiple : "{" privilege_region_fields_list "}"'
        p[0] = p[2]

    def p_privilege_region_fields_list_single_field(self, p):
        'privilege_region_fields_list : privilege_region_field'
        p[0] = ast.PrivilegeRegionFields(
            span = ast.Span.from_slice(p),
            field = p[1])

    def p_privilege_region_fields_list_additional_field(self, p):
        'privilege_region_fields_list : privilege_region_fields_list "," privilege_region_field'
        p[0] = ast.PrivilegeRegionFields(
            span = ast.Span.from_slice(p),
            fields = p[1],
            field = p[3])

    def p_reduce_op(self, p):
        'reduce_op : type_start reduce_op_inner type_end'
        p[0] = p[2]

    def p_reduce_op_inner(self, p):
        '''reduce_op_inner : "<" "*" ">"
                           | "<" "/" ">"
                           | "<" "+" ">"
                           | "<" "-" ">"
                           | "<" "&" ">"
                           | "<" "^" ">"
                           | "<" "|" ">"'''
        p[0] = p[2]

    def p_return_type_void(self, p):
        'return_type : '
        p[0] = ast.TypeVoid(
            span = ast.Span.from_slice(p))

    def p_return_type_nonvoid(self, p):
        'return_type : ":" type'
        p[0] = p[2]

    def p_type_start(self, p):
        'type_start :'
        p.lexer.lexer.push_state('type')

    def p_type_end(self, p):
        'type_end :'
        p.lexer.lexer.pop_state()

    def p_type(self, p):
        'type : type_start type_inner type_end'
        p[0] = p[2]

    def p_type_inner_type_nonvoid(self, p):
        'type_inner : type_nonvoid_inner'
        p[0] = p[1]

    def p_type_inner_void(self, p):
        'type_inner : VOID'
        p[0] = ast.TypeVoid(
            span = ast.Span.from_slice(p))

    def p_type_nonvoid(self, p):
        'type_nonvoid : type_start type_nonvoid_inner type_end'
        p[0] = p[2]

    def p_type_nonvoid_inner_bool(self, p):
        'type_nonvoid_inner : BOOL'
        p[0] = ast.TypeBool(
            span = ast.Span.from_slice(p))

    def p_type_nonvoid_inner_numberic(self, p):
        'type_nonvoid_inner : type_numeric'
        p[0] = p[1]

    def p_type_numeric_double(self, p):
        'type_numeric : DOUBLE'
        p[0] = ast.TypeDouble(
            span = ast.Span.from_slice(p))

    def p_type_numeric_float(self, p):
        'type_numeric : FLOAT'
        p[0] = ast.TypeFloat(
            span = ast.Span.from_slice(p))

    def p_type_numeric_int(self, p):
        'type_numeric : INT'
        p[0] = ast.TypeInt(
            span = ast.Span.from_slice(p))

    def p_type_numeric_uint(self, p):
        'type_numeric : UINT'
        p[0] = ast.TypeUInt(
            span = ast.Span.from_slice(p))

    def p_type_numeric_int8(self, p):
        'type_numeric : INT8'
        p[0] = ast.TypeInt8(
            span = ast.Span.from_slice(p))

    def p_type_numeric_int16(self, p):
        'type_numeric : INT16'
        p[0] = ast.TypeInt16(
            span = ast.Span.from_slice(p))

    def p_type_numeric_int32(self, p):
        'type_numeric : INT32'
        p[0] = ast.TypeInt32(
            span = ast.Span.from_slice(p))

    def p_type_numeric_int64(self, p):
        'type_numeric : INT64'
        p[0] = ast.TypeInt64(
            span = ast.Span.from_slice(p))

    def p_type_numeric_uint8(self, p):
        'type_numeric : UINT8'
        p[0] = ast.TypeUInt8(
            span = ast.Span.from_slice(p))

    def p_type_numeric_uint16(self, p):
        'type_numeric : UINT16'
        p[0] = ast.TypeUInt16(
            span = ast.Span.from_slice(p))

    def p_type_numeric_uint32(self, p):
        'type_numeric : UINT32'
        p[0] = ast.TypeUInt32(
            span = ast.Span.from_slice(p))

    def p_type_numeric_uint64(self, p):
        'type_numeric : UINT64'
        p[0] = ast.TypeUInt64(
            span = ast.Span.from_slice(p))

    def p_type_nonvoid_inner_coloring(self, p):
        'type_nonvoid_inner : COLORING "<" type_coloring_region ">"'
        p[0] = ast.TypeColoring(
            span = ast.Span.from_slice(p),
            region = p[3])

    def p_type_coloring_region(self, p):
        'type_coloring_region : ID'
        p[0] = ast.TypeColoringRegion(
            span = ast.Span.from_slice(p),
            name = p[1])

    def p_type_nonvoid_inner_id(self, p):
        'type_nonvoid_inner : type_id'
        p[0] = p[1]

    def p_type_id(self, p):
        'type_id : ID type_args'
        p[0] = ast.TypeID(
            span = ast.Span.from_slice(p),
            name = p[1], args = p[2])

    def p_type_args_empty(self, p):
        'type_args : '
        p[0] = ast.TypeArgs(
            span = ast.Span.from_slice(p))

    def p_type_args_nonempty(self, p):
        'type_args : "<" type_args_list ">"'
        p[0] = p[2]

    def p_type_args_list_single_arg(self, p):
        'type_args_list : type_arg'
        p[0] = ast.TypeArgs(
            span = ast.Span.from_slice(p),
            arg = p[1])

    def p_type_args_list_additional_arg(self, p):
        'type_args_list : type_args_list "," type_arg'
        p[0] = ast.TypeArgs(
            span = ast.Span.from_slice(p),
            args = p[1], arg = p[3])

    def p_type_arg(self, p):
        'type_arg : ID'
        p[0] = ast.TypeArg(
            span = ast.Span.from_slice(p),
            name = p[1])

    def p_type_arg_wild(self, p):
        'type_arg : "?"'
        p[0] = ast.TypeArgWild(
            span = ast.Span.from_slice(p))

    def p_type_nonvoid_inner_pointer(self, p):
        'type_nonvoid_inner : type_pointer_inner'
        p[0] = p[1]

    def p_type_pointer(self, p):
        'type_pointer : type_start type_pointer_inner type_end'
        p[0] = p[2]

    def p_type_pointer_inner(self, p):
        'type_pointer_inner : type_nonvoid_inner "@" type_pointer_regions'
        p[0] = ast.TypePointer(
            span = ast.Span.from_slice(p),
            points_to_type = p[1], regions = p[3])

    def p_type_pointer_regions_single(self, p):
        'type_pointer_regions : type_region'
        p[0] = ast.TypePointerRegions(
            span = ast.Span.from_slice(p),
            region = p[1])

    def p_type_pointer_regions_multiple(self, p):
        'type_pointer_regions : "(" type_pointer_regions_list ")"'
        p[0] = p[2]

    def p_type_pointer_regions_list_single_region(self, p):
        'type_pointer_regions_list : type_region'
        p[0] = ast.TypePointerRegions(
            span = ast.Span.from_slice(p),
            region = p[1])

    def p_type_pointer_regions_list_additional_region(self, p):
        'type_pointer_regions_list : type_pointer_regions_list "," type_region'
        p[0] = ast.TypePointerRegions(
            span = ast.Span.from_slice(p),
            regions = p[1], region = p[3])

    def p_type_region(self, p):
        'type_region : ID'
        p[0] = ast.TypeRegion(
            span = ast.Span.from_slice(p),
            name = p[1])

    def p_type_region_wild(self, p):
        'type_region : "?"'
        p[0] = ast.TypeRegionWild(
            span = ast.Span.from_slice(p))

    def p_type_region_kind(self, p):
        'type_region_kind : type_start type_region_kind_inner type_end'
        p[0] = p[2]

    def p_type_region_kind_inner_region_type(self, p):
        'type_region_kind_inner : REGION "<" type_nonvoid_inner ">"'
        p[0] = ast.TypeRegionKind(
            span = ast.Span.from_slice(p),
            contains_type = p[3])

    def p_type_array_kind(self, p):
        'type_array_kind : type_start type_array_kind_inner type_end'
        p[0] = p[2]

    def p_type_array_kind_inner(self, p):
        'type_array_kind_inner : ARRAY "<" type_ispace "," type_nonvoid_inner ">"'
        p[0] = ast.TypeArrayKind(
            span = ast.Span.from_slice(p),
            ispace = p[3], contains_type = p[5])

    def p_type_ispace(self, p):
        'type_ispace : ID'
        p[0] = ast.TypeIspace(
            span = ast.Span.from_slice(p),
            name = p[1])

    def p_type_ispace_kind(self, p):
        'type_ispace_kind : type_start type_ispace_kind_inner type_end'
        p[0] = p[2]

    def p_type_ispace_kind_inner(self, p):
        'type_ispace_kind_inner : ISPACE "<" INT ">"'
        p[0] = ast.TypeIspaceKind(
            span = ast.Span.from_slice(p),
            index_type = ast.TypeInt(
                span = ast.Span.from_slice(p)))

    def p_type_partition_kind(self, p):
        'type_partition_kind : type_start type_partition_kind_inner type_end'
        p[0] = p[2]

    def p_type_partition_kind_inner_partition_type(self, p):
        'type_partition_kind_inner : PARTITION "<" type_region "," partition_mode ">"'
        p[0] = ast.TypePartitionKind(
            span = ast.Span.from_slice(p),
            region_type = p[3],
            mode = p[5])

    def p_block_empty(self, p):
        'block : "{" "}"'
        p[0] = ast.Block(
            span = ast.Span.from_slice(p))

    def p_block_nonempty(self, p):
        'block : "{" block_list "}"'
        p[0] = p[2]

    def p_block_list_single_statement(self, p):
        'block_list : statement'
        p[0] = ast.Block(
            span = ast.Span.from_slice(p),
            statement = p[1])

    def p_block_list_additional_statement(self, p):
        'block_list : block_list statement'
        p[0] = ast.Block(
            span = ast.Span.from_slice(p),
            block = p[1], statement = p[2])

    def p_statement_assert(self, p):
        'statement : ASSERT expr ";"'
        p[0] = ast.StatementAssert(
            span = ast.Span.from_slice(p),
            expr = p[2])

    def p_statement_expr(self, p):
        'statement : expr ";"'
        p[0] = ast.StatementExpr(
            span = ast.Span.from_slice(p),
            expr = p[1])

    def p_statement_if(self, p):
        'statement : IF "(" expr ")" block'
        p[0] = ast.StatementIf(
            span = ast.Span.from_slice(p),
            condition = p[3], then_block = p[5])

    def p_statement_if_else(self, p):
        'statement : IF "(" expr ")" block ELSE block'
        p[0] = ast.StatementIf(
            span = ast.Span.from_slice(p),
            condition = p[3], then_block = p[5], else_block = p[7])

    def p_statement_for_region(self, p):
        'statement : FOR for_index_list IN for_region block'
        p[0] = ast.StatementFor(
            span = ast.Span.from_slice(p),
            indices = p[2], regions = ast.ForRegions(
                span = ast.Span.from_slice(p),
                region = p[4]),
            block = p[5])

    def p_statement_for_zip(self, p):
        'statement : FOR for_index_list IN ZIP "(" for_region_list ")" block'
        p[0] = ast.StatementFor(
            span = ast.Span.from_slice(p),
            indices = p[2], regions = p[6], block = p[8])

    def p_for_index_list_single_index(self, p):
        'for_index_list : for_index'
        p[0] = ast.ForIndices(
            span = ast.Span.from_slice(p),
            index = p[1])

    def p_for_index_list_additional_index(self, p):
        'for_index_list : for_index_list "," for_index'
        p[0] = ast.ForIndices(
            span = ast.Span.from_slice(p),
            indices = p[1], index = p[3])

    def p_for_index_name(self, p):
        'for_index : ID'
        p[0] = ast.ForIndex(
            span = ast.Span.from_slice(p),
            name = p[1], type = None)

    def p_for_index_name_type(self, p):
        'for_index : ID ":" type_nonvoid'
        p[0] = ast.ForIndex(
            span = ast.Span.from_slice(p),
            name = p[1], type = p[3])

    def p_for_region_list_single_region(self, p):
        'for_region_list : for_region'
        p[0] = ast.ForRegions(
            span = ast.Span.from_slice(p),
            region = p[1])

    def p_for_region_list_additional_region(self, p):
        'for_region_list : for_region_list "," for_region'
        p[0] = ast.ForRegions(
            span = ast.Span.from_slice(p),
            regions = p[1], region = p[3])

    def p_for_region(self, p):
        'for_region : ID'
        p[0] = ast.ForRegion(
            span = ast.Span.from_slice(p),
            name = p[1])

    def p_statement_let_expr(self, p):
        'statement : LET ID "=" expr ";"'
        p[0] = ast.StatementLet(
            span = ast.Span.from_slice(p),
            name = p[2], type = None, expr = p[4])

    def p_statement_let_type_nonvoid_expr(self, p):
        'statement : LET ID ":" type_nonvoid "=" expr ";"'
        p[0] = ast.StatementLet(
            span = ast.Span.from_slice(p),
            name = p[2], type = p[4], expr = p[6])

    def p_statement_let_region_expr(self, p):
        'statement : LET ID "=" REGION "<" type_nonvoid ">" "(" expr ")" ";"'
        p[0] = ast.StatementLetRegion(
            span = ast.Span.from_slice(p),
            name = p[2], region_kind = None, element_type = p[6], size_expr = p[9])

    def p_statement_let_region_type_expr(self, p):
        'statement : LET ID ":" type_region_kind "=" REGION "<" type_nonvoid ">" "(" expr ")" ";"'
        p[0] = ast.StatementLetRegion(
            span = ast.Span.from_slice(p),
            name = p[2], region_kind = p[4], element_type = p[8], size_expr = p[11])

    def p_statement_let_array_expr(self, p):
        'statement : LET ID "=" ARRAY "<" type_ispace "," type_nonvoid ">" "(" ")" ";"'
        p[0] = ast.StatementLetArray(
            span = ast.Span.from_slice(p),
            name = p[2], region_kind = None, ispace_type = p[6], element_type = p[8])

    def p_statement_let_array_type_expr(self, p):
        'statement : LET ID ":" type_array_kind "=" ARRAY "<" type_ispace "," type_nonvoid ">" "(" ")" ";"'
        p[0] = ast.StatementLetArray(
            span = ast.Span.from_slice(p),
            name = p[2], region_kind = p[4], ispace_type = p[8], element_type = p[10])

    def p_statement_let_ispace_expr(self, p):
        'statement : LET ID "=" ISPACE "<" INT ">" "(" expr ")" ";"'
        p[0] = ast.StatementLetIspace(
            span = ast.Span.from_slice(p),
            name = p[2], ispace_kind = None,
            index_type = ast.TypeInt(
                span = ast.Span.from_slice(p)),
            size_expr = p[9])

    def p_statement_let_ispace_type_expr(self, p):
        'statement : LET ID ":" type_ispace_kind "=" ISPACE "<" INT ">" "(" expr ")" ";"'
        p[0] = ast.StatementLetIspace(
            span = ast.Span.from_slice(p),
            name = p[2], ispace_kind = p[4], index_type = ast.TypeInt(), size_expr = p[11])

    def p_statement_let_partition_expr(self, p):
        'statement : LET ID "=" PARTITION "<" type_region "," partition_mode ">" "(" expr ")" ";"'
        p[0] = ast.StatementLetPartition(
            span = ast.Span.from_slice(p),
            name = p[2],
            partition_kind = None,
            region_type = p[6],
            mode = p[8],
            coloring_expr = p[11])

    def p_statement_let_partition_type_expr(self, p):
        'statement : LET ID ":" type_partition_kind "=" PARTITION "<" type_region "," partition_mode ">" "(" expr ")" ";"'
        p[0] = ast.StatementLetPartition(
            span = ast.Span.from_slice(p),
            name = p[2],
            partition_kind = p[4],
            region_type = p[8],
            mode = p[10],
            coloring_expr = p[13])

    def p_partition_mode(self, p):
        '''partition_mode : DISJOINT
                          | ALIASED'''
        p[0] = ast.PartitionMode(
            span = ast.Span.from_slice(p),
            mode = p[1])

    def p_statement_return(self, p):
        'statement : RETURN expr ";"'
        p[0] = ast.StatementReturn(
            span = ast.Span.from_slice(p),
            expr = p[2])

    def p_statement_unpack(self, p):
        'statement : UNPACK expr AS ID ":" type_id unpack_regions ";"'
        p[0] = ast.StatementUnpack(
            span = ast.Span.from_slice(p),
            expr = p[2], name = p[4], type = p[6], regions = p[7])

    def p_unpack_regions_nonempty(self, p):
        'unpack_regions : "[" unpack_regions_list "]"'
        p[0] = p[2]

    def p_unpack_regions_list_single_region(self, p):
        'unpack_regions_list : unpack_region'
        p[0] = ast.UnpackRegions(
            span = ast.Span.from_slice(p),
            region = p[1])

    def p_unpack_regions_list_additional_region(self, p):
        'unpack_regions_list : unpack_regions_list "," unpack_region'
        p[0] = ast.UnpackRegions(
            span = ast.Span.from_slice(p),
            regions = p[1], region = p[3])

    def p_unpack_region(self, p):
        'unpack_region : ID'
        p[0] = ast.UnpackRegion(
            span = ast.Span.from_slice(p),
            name = p[1])

    def p_statement_var_expr(self, p):
        'statement : VAR ID "=" expr ";"'
        p[0] = ast.StatementVar(
            span = ast.Span.from_slice(p),
            name = p[2], type = None, expr = p[4])

    def p_statement_var_type_nonvoid_expr(self, p):
        'statement : VAR ID ":" type_nonvoid "=" expr ";"'
        p[0] = ast.StatementVar(
            span = ast.Span.from_slice(p),
            name = p[2], type = p[4], expr = p[6])

    def p_statement_while(self, p):
        'statement : WHILE "(" expr ")" block'
        p[0] = ast.StatementWhile(
            span = ast.Span.from_slice(p),
            condition = p[3], block = p[5])

    def p_expr_parens(self, p):
        'expr : "(" expr ")"'
        p[0] = p[2]

    def p_expr_id(self, p):
        'expr : ID'
        p[0] = ast.ExprID(
            span = ast.Span.from_slice(p),
            name = p[1])

    def p_expr_assignment(self, p):
        'expr : expr "=" expr'
        p[0] = ast.ExprAssignment(
            span = ast.Span.from_slice(p),
            lval = p[1], rval = p[3])

    def p_expr_unaryop(self, p):
        '''expr : "-" expr
                | "!" expr
                | "~" expr'''
        p[0] = ast.ExprUnaryOp(
            span = ast.Span.from_slice(p),
            op = p[1], arg = p[2])

    def p_expr_binaryop(self, p):
        '''expr : expr "*" expr
                | expr "/" expr
                | expr "%" expr
                | expr "+" expr
                | expr "-" expr
                | expr RIGHT_SHIFT expr
                | expr LEFT_SHIFT expr
                | expr "<" expr
                | expr LESS_EQUALS expr
                | expr ">" expr
                | expr GREATER_EQUALS expr
                | expr EQUALS expr
                | expr NOT_EQUALS expr
                | expr "&" expr
                | expr "|" expr
                | expr "^" expr
                | expr LOGICAL_AND expr
                | expr LOGICAL_OR expr'''
        p[0] = ast.ExprBinaryOp(
            span = ast.Span.from_slice(p),
            op = p[2], lhs = p[1], rhs = p[3])

    def p_expr_reduceop(self, p):
        '''expr : expr TIMES_EQUALS expr
                | expr DIVIDE_EQUALS expr
                | expr MODULUS_EQUALS expr
                | expr PLUS_EQUALS expr
                | expr MINUS_EQUALS expr
                | expr RIGHT_SHIFT_EQUALS expr
                | expr LEFT_SHIFT_EQUALS expr
                | expr BITWISE_AND_EQUALS expr
                | expr BITWISE_XOR_EQUALS expr
                | expr BITWISE_OR_EQUALS expr'''
        p[0] = ast.ExprReduceOp(
            span = ast.Span.from_slice(p),
            op = p[2][:-1], lhs = p[1], rhs = p[3])

    def p_expr_cast(self, p):
        'expr : type_numeric "(" expr ")"'
        p[0] = ast.ExprCast(
            span = ast.Span.from_slice(p),
            cast_to_type = p[1], expr = p[3])

    def p_expr_null(self, p):
        'expr : NULL "<" type_pointer ">" "(" ")"'
        p[0] = ast.ExprNull(
            span = ast.Span.from_slice(p),
            pointer_type = p[3])

    def p_expr_isnull(self, p):
        'expr : ISNULL "(" expr ")"'
        p[0] = ast.ExprIsnull(
            span = ast.Span.from_slice(p),
            pointer_expr = p[3])

    def p_expr_new(self, p):
        'expr : NEW "<" type_pointer ">" "(" ")"'
        p[0] = ast.ExprNew(
            span = ast.Span.from_slice(p),
            pointer_type = p[3])

    def p_expr_read(self, p):
        'expr : READ "(" expr ")"'
        p[0] = ast.ExprRead(
            span = ast.Span.from_slice(p),
            pointer_expr = p[3])

    def p_expr_write(self, p):
        'expr : WRITE "(" expr "," expr ")"'
        p[0] = ast.ExprWrite(
            span = ast.Span.from_slice(p),
            pointer_expr = p[3], value_expr = p[5])

    def p_expr_reduce(self, p):
        'expr : REDUCE reduce_op "(" expr "," expr ")"'
        p[0] = ast.ExprReduce(
            span = ast.Span.from_slice(p),
            op = p[2], pointer_expr = p[4], value_expr = p[6])

    def p_expr_dereference(self, p):
        'expr : "*" expr'
        p[0] = ast.ExprDereference(
            span = ast.Span.from_slice(p),
            pointer_expr = p[2])

    def p_expr_array_access(self, p):
        'expr : ID "[" expr "]"'
        p[0] = ast.ExprArrayAccess(
            span = ast.Span.from_slice(p),
            array_expr = ast.ExprID(
                span = ast.Span.from_slice(p),
                name = p[1]),
            index_expr = p[3])

    def p_expr_field_access(self, p):
        'expr : expr "." ID'
        p[0] = ast.ExprFieldAccess(
            span = ast.Span.from_slice(p),
            struct_expr = p[1], field_name = p[3])

    def p_expr_field_dereference(self, p):
        'expr : expr ARROW ID'
        p[0] = ast.ExprFieldDereference(
            span = ast.Span.from_slice(p),
            pointer_expr = p[1], field_name = p[3])

    def p_expr_field_values(self, p):
        'expr : field_values'
        p[0] = ast.ExprFieldValues(
            span = ast.Span.from_slice(p),
            field_values = p[1])

    def p_field_values_nonempty(self, p):
        'field_values : "{" field_values_list optional_comma "}"'
        p[0] = p[2]

    def p_field_values_list_single_arg(self, p):
        'field_values_list : field_value'
        p[0] = ast.FieldValues(
            span = ast.Span.from_slice(p),
            field_value = p[1])

    def p_field_values_list_additional_arg(self, p):
        'field_values_list : field_values_list "," field_value'
        p[0] = ast.FieldValues(
            span = ast.Span.from_slice(p),
            field_values = p[1], field_value = p[3])

    def p_field_value(self, p):
        'field_value : ID ":" expr'
        p[0] = ast.FieldValue(
            span = ast.Span.from_slice(p),
            field_name = p[1], value_expr = p[3])

    def p_expr_field_update(self, p):
        'expr : expr field_updates'
        p[0] = ast.ExprFieldUpdates(
            span = ast.Span.from_slice(p),
            struct_expr = p[1], field_updates = p[2])

    def p_field_updates_nonempty(self, p):
        'field_updates : "{" field_updates_list optional_comma "}"'
        p[0] = p[2]

    def p_field_updates_list_single_arg(self, p):
        'field_updates_list : field_update'
        p[0] = ast.FieldUpdates(
            span = ast.Span.from_slice(p),
            field_update = p[1])

    def p_field_updates_list_additional_arg(self, p):
        'field_updates_list : field_updates_list "," field_update'
        p[0] = ast.FieldUpdates(
            span = ast.Span.from_slice(p),
            field_updates = p[1], field_update = p[3])

    def p_field_update(self, p):
        'field_update : ID ":" expr'
        p[0] = ast.FieldUpdate(
            span = ast.Span.from_slice(p),
            field_name = p[1], update_expr = p[3])

    def p_expr_coloring(self, p):
        'expr : COLORING "<" coloring_region ">" "(" ")"'
        p[0] = ast.ExprColoring(
            span = ast.Span.from_slice(p),
            region = p[3])

    def p_coloring_region(self, p):
        'coloring_region : ID'
        p[0] = ast.ColoringRegion(
            span = ast.Span.from_slice(p),
            name = p[1])

    def p_expr_color(self, p):
        'expr : COLOR "(" expr "," expr "," expr ")"'
        p[0] = ast.ExprColor(
            span = ast.Span.from_slice(p),
            coloring_expr = p[3], pointer_expr = p[5], color_expr = p[7])

    def p_expr_upregion(self, p):
        'expr : UPREGION upregion_regions "(" expr ")"'
        p[0] = ast.ExprUpregion(
            span = ast.Span.from_slice(p),
            regions = p[2], expr = p[4])

    def p_upregion_regions(self, p):
        'upregion_regions : "<" upregion_regions_list ">"'
        p[0] = p[2]

    def p_upregion_regions_list_single_region(self, p):
        'upregion_regions_list : upregion_region'
        p[0] = ast.UpregionRegions(
            span = ast.Span.from_slice(p),
            region = p[1])

    def p_upregion_regions_list_additional_region(self, p):
        'upregion_regions_list : upregion_regions_list "," upregion_region'
        p[0] = ast.UpregionRegions(
            span = ast.Span.from_slice(p),
            regions = p[1], region = p[3])

    def p_upregion_region(self, p):
        'upregion_region : ID'
        p[0] = ast.UpregionRegion(
            span = ast.Span.from_slice(p),
            name = p[1])

    def p_expr_downregion(self, p):
        'expr : DOWNREGION downregion_regions "(" expr ")"'
        p[0] = ast.ExprDownregion(
            span = ast.Span.from_slice(p),
            regions = p[2], expr = p[4])

    def p_downregion_regions(self, p):
        'downregion_regions : "<" downregion_regions_list ">"'
        p[0] = p[2]

    def p_downregion_regions_list_single_region(self, p):
        'downregion_regions_list : downregion_region'
        p[0] = ast.DownregionRegions(
            span = ast.Span.from_slice(p),
            region = p[1])

    def p_downregion_regions_list_additional_region(self, p):
        'downregion_regions_list : downregion_regions_list "," downregion_region'
        p[0] = ast.DownregionRegions(
            span = ast.Span.from_slice(p),
            regions = p[1], region = p[3])

    def p_downregion_region(self, p):
        'downregion_region : ID'
        p[0] = ast.DownregionRegion(
            span = ast.Span.from_slice(p),
            name = p[1])

    def p_expr_pack(self, p):
        'expr : PACK expr AS type_id pack_regions'
        p[0] = ast.ExprPack(
            span = ast.Span.from_slice(p),
            expr = p[2], type = p[4], regions = p[5])

    def p_pack_regions_nonempty(self, p):
        'pack_regions : "[" pack_regions_list "]"'
        p[0] = p[2]

    def p_pack_regions_list_single_region(self, p):
        'pack_regions_list : pack_region'
        p[0] = ast.PackRegions(
            span = ast.Span.from_slice(p),
            region = p[1])

    def p_pack_regions_list_additional_region(self, p):
        'pack_regions_list : pack_regions_list "," pack_region'
        p[0] = ast.PackRegions(
            span = ast.Span.from_slice(p),
            regions = p[1], region = p[3])

    def p_pack_region(self, p):
        'pack_region : ID'
        p[0] = ast.PackRegion(
            span = ast.Span.from_slice(p),
            name = p[1])

    def p_expr_call(self, p):
        'expr : ID args'
        p[0] = ast.ExprCall(
            span = ast.Span.from_slice(p),
            function = ast.ExprID(
                span = ast.Span.from_slice(p),
                name = p[1]),
            args = p[2])

    def p_args_empty(self, p):
        'args : "(" ")"'
        p[0] = ast.Args(
            span = ast.Span.from_slice(p))

    def p_args_nonempty(self, p):
        'args : "(" args_list ")"'
        p[0] = p[2]

    def p_args_list_single_arg(self, p):
        'args_list : expr'
        p[0] = ast.Args(
            span = ast.Span.from_slice(p),
            arg = p[1])

    def p_args_list_additional_arg(self, p):
        'args_list : args_list "," expr'
        p[0] = ast.Args(
            span = ast.Span.from_slice(p),
            args = p[1], arg = p[3])

    def p_expr_const_true(self, p):
        'expr : TRUE'
        p[0] = ast.ExprConstBool(
            span = ast.Span.from_slice(p),
            value = p[1])

    def p_expr_const_false(self, p):
        'expr : FALSE'
        p[0] = ast.ExprConstBool(
            span = ast.Span.from_slice(p),
            value = p[1])

    def p_expr_const_double(self, p):
        'expr : DOUBLE_VALUE'
        p[0] = ast.ExprConstDouble(
            span = ast.Span.from_slice(p),
            value = p[1])

    def p_expr_const_float(self, p):
        'expr : FLOAT_VALUE'
        p[0] = ast.ExprConstFloat(
            span = ast.Span.from_slice(p),
            value = p[1][:-1])

    def p_expr_const_int_hex(self, p):
        'expr : HEX_VALUE'
        p[0] = ast.ExprConstInt(
            span = ast.Span.from_slice(p),
            value = int(p[1], 16))

    def p_expr_const_int_int(self, p):
        'expr : INT_VALUE'
        p[0] = ast.ExprConstInt(
            span = ast.Span.from_slice(p),
            value = int(p[1]))

    def p_expr_const_uint_uhex(self, p):
        'expr : UHEX_VALUE'
        p[0] = ast.ExprConstUInt(
            span = ast.Span.from_slice(p),
            value = int(p[1][:-1], 16))

    def p_expr_const_uint_uint(self, p):
        'expr : UINT_VALUE'
        p[0] = ast.ExprConstUInt(
            span = ast.Span.from_slice(p),
            value = int(p[1][:-1]))

    def p_error(self, p):
        raise ParseError(self.lexer.source, p)

    def __init__(self, tabmodule = 'parsetab_lcomp', **kwargs):
        self.lexer = lex.Lexer()

        # Hide the output files from parsing so they don't litter the
        # current directory.
        output_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

        # Make sure the output directory is on path, otherwise PLY
        # will fail to see the cached file.
        if output_dir not in sys.path:
            sys.path.append(output_dir)

        # Change to the output directory. This is necessary, in
        # addition to the outputdir parameter, because PLY does not
        # respect outputdir for the parser.out file.
        current_dir = os.getcwd()
        try:
            os.chdir(output_dir)
            self.parser = ply.yacc.yacc(module = self, tabmodule = tabmodule, outputdir = output_dir, **kwargs)
        finally:
            os.chdir(current_dir)

    def parse(self, source, **kwargs):
        self.lexer.input(source)
        return self.parser.parse(lexer = self.lexer)
