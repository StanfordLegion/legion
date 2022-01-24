-- Copyright 2022 Stanford University, NVIDIA Corporation
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

-- Regent De-sugaring Pass

local ast = require("regent/ast")
local ast_util = require("regent/ast_util")
local data = require("common/data")
local report = require("common/report")
local std = require("regent/std")

local desugar = {}

--
-- De-sugar statement "var ip = image(r, p, f)" into the following statements:
--
-- var coloring : legion_domain_point_coloring_t
-- coloring = legion_domain_point_coloring_create()
-- for color in p.colors do
--   legion_domain_point_coloring_color_domain(
--     coloring, color, f(p[color].bounds))
-- end
-- var ip = partition(aliased, r, coloring)
-- legion_domain_point_coloring_destroy(coloring)
--
local capi = std.c

local function desugar_image_by_task(cx, node)
  local parent = node.value.parent.value
  local parent_type = parent:gettype()
  local partition = node.value.partition
  local partition_type = std.as_read(partition.expr_type)
  local image_partition_type = node.type

  local stats = terralib.newlist()

  local coloring_symbol =
    std.newsymbol(capi.legion_domain_point_coloring_t)
  local coloring_expr = ast_util.mk_expr_id(coloring_symbol)
  stats:insert(
    ast_util.mk_stat_var(
      coloring_symbol, nil,
      ast_util.mk_expr_call(capi.legion_domain_point_coloring_create)))

  local colors_symbol = std.newsymbol(partition_type:colors())
  local color_symbol =
    std.newsymbol(partition_type:colors().index_type(colors_symbol))
  local colors_expr = ast_util.mk_expr_colors_access(partition)
  local subregion_type = partition_type:subregion_dynamic()
  std.add_constraint(cx, subregion_type, partition_type, std.subregion, false)

  local subregion_expr =
    ast_util.mk_expr_index_access(partition,
                                  ast_util.mk_expr_id(color_symbol),
                                  subregion_type)
  local rect_expr =
    ast_util.mk_expr_call(node.value.task.value,
                          ast_util.mk_expr_bounds_access(subregion_expr))
  local loop_body =
    ast_util.mk_stat_expr(
      ast_util.mk_expr_call(capi.legion_domain_point_coloring_color_domain,
                            terralib.newlist { coloring_expr,
                                               ast_util.mk_expr_id(color_symbol),
                                               rect_expr }))

  stats:insert(ast_util.mk_stat_var(colors_symbol, nil, colors_expr))
  stats:insert(
    ast_util.mk_stat_for_list(color_symbol,
                              ast_util.mk_expr_id(colors_symbol),
                              ast_util.mk_block(loop_body)))

  stats:insert(
    ast_util.mk_stat_var(node.symbol, image_partition_type,
                         ast_util.mk_expr_partition(image_partition_type,
                                                    ast_util.mk_expr_id(colors_symbol),
                                                    coloring_expr)))
  std.add_constraint(cx, image_partition_type, parent_type, std.subregion, false)

  stats:insert(
    ast_util.mk_stat_expr(
      ast_util.mk_expr_call(capi.legion_domain_point_coloring_destroy,
                            coloring_expr)))

  return stats
end

local function desugar_block(cx, node)
  return node { block = desugar.block(cx, node.block) }
end

local function desugar_if(cx, node)
  local then_block = desugar.block(cx, node.then_block)
  local elseif_blocks = node.elseif_blocks:map(function(elseif_block)
    return desugar.stat(cx, elseif_block)
  end)
  local else_block = desugar.block(cx, node.else_block)
  return node {
    then_block = then_block,
    elseif_blocks = elseif_blocks,
    else_block = else_block,
  }
end

local function desugar_elseif(cx, node)
  local block = desugar.block(cx, node.block)
  return node { block = block }
end

local function desugar_parallelize_with(cx, node)
  if not std.config["parallelize"] then
    return ast.typed.stat.Block {
      block = node.block,
      span = node.span,
      annotations = node.annotations,
    }
  else
    return node
  end
end

local function desugar_var(cx, node)
  if node.value and node.value:is(ast.typed.expr.ImageByTask) then
    return desugar_image_by_task(cx, node)
  else
    return node
  end
end

local function do_nothing(cx, node) return node end

local desugar_stat_table = {
  [ast.typed.stat.While]            = desugar_block,
  [ast.typed.stat.ForNum]           = desugar_block,
  [ast.typed.stat.ForList]          = desugar_block,
  [ast.typed.stat.Repeat]           = desugar_block,
  [ast.typed.stat.Block]            = desugar_block,
  [ast.typed.stat.MustEpoch]        = desugar_block,
  [ast.typed.stat.If]               = desugar_if,
  [ast.typed.stat.Elseif]           = desugar_elseif,
  [ast.typed.stat.ParallelizeWith]  = desugar_parallelize_with,
  [ast.typed.stat.Var]              = desugar_var,
  [ast.typed.stat]                  = do_nothing,
}

local desugar_stat = ast.make_single_dispatch(
  desugar_stat_table,
  {})

function desugar.stat(cx, node)
  return desugar_stat(cx)(node)
end

function desugar.block(cx, block)
  local stats = terralib.newlist()
  block.stats:map(function(stat)
    local new_stat = desugar.stat(cx, stat)
    if terralib.islist(new_stat) then
      stats:insertall(new_stat)
    else
      stats:insert(new_stat)
    end
  end)
  return block { stats = stats }
end

function desugar.top_task(node)
  local cx = { constraints = node.prototype:get_constraints() }
  local body = node.body and desugar.block(cx, node.body) or false

  return node { body = body }
end

function desugar.entry(node)
  if node:is(ast.typed.top.Task) then
    return desugar.top_task(node)
  else
    return node
  end
end

return desugar
