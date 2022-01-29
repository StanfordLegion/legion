-- Copyright 2022 Stanford University
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

-- Math functions supported in Regent

-- Read math.h to get function signatures
local cmath = terralib.includec("math.h")

local all_math_fns = terralib.newlist({
  "acos", "acosh", "asin", "asinh", "atan", "atan2", "atanh",
  "cbrt", "ceil", "copysign", "cos", "cosh",
  "erf", "erfc", "exp", "exp2", "expm1",
  "fabs", "fdim", "finite", "floor", "fma", "fmax", "fmin", "fmod", "frexp",
  "hypot",
  "ilogb", "isinf", "isnan",
  "j0", "j1", "jn",
  "ldexp", "lgamma", "llrint", "llround", "log", "log10", "log1p", "log2", "logb",
  "modf",
  "nan", "nearbyint", "nextafter",
  "pow",
  "remainder", "remquo", "rint", "round",
  "scalbn", "sin", "sinh", "sqrt", "tan",
  "tanh", "tgamma", "trunc",
  "y0", "y1", "yn",
})

-- Slow Terra implementations for CUDA-only math functions
local function rsqrt_cpu(arg_type)
  if arg_type == double then
    return terra(v : double) return cmath.pow(v, -0.5) end
  else
    return terra(v : float) return cmath.powf(v, -0.5) end
  end
end

local cuda_only_math_fns = terralib.newlist({
  {
    name = "rsqrt",
    default_variant = rsqrt_cpu,
  },
})

local math_fn = {}
math_fn.__index = math_fn

function math_fn.new_math_fn(fn_name, arg_type)
  local t = {
    fn_name = fn_name,
    arg_type = arg_type,
    variants = {},
  }
  return setmetatable(t, math_fn)
end

function math_fn:has_variant(variant_name)
  return self.variants[variant_name]
end

function math_fn:get_variant(variant_name)
  assert(self:has_variant(variant_name))
  return self.variants[variant_name]
end

function math_fn:set_variant(variant_name, variant)
  assert(self:has_variant(variant_name) == nil)
  self.variants[variant_name] = variant
end

-- Returns the C math implementation by default.
-- Other optimization passes (such as vectorizer or CUDA code generation) override
-- this function to return a platform specific implementation
function math_fn:get_definition()
  local d
  if self:has_variant("default") then
    d = self:get_variant("default")
  else
    local func_name = ((self.arg_type == float) and self.fn_name .. "f") or self.fn_name
    d = cmath[func_name]
    assert(d ~= nil)
    self:set_variant("default", d)
  end
  return d
end

function math_fn:get_name()
  assert(self.fn_name)
  return self.fn_name
end

function math_fn:get_arg_type()
  assert(self.arg_type)
  return self.arg_type
end

function math_fn:override(new)
  local copy = math_fn.new_math_fn(self.fn_name, self.arg_type)
  copy.super = self
  copy.get_definition = new
  return copy
end

function math_fn:printpretty()
  return '[regentlib.' .. self.fn_name .. '(' .. tostring(self.arg_type) .. ')]'
end

local math = {}

all_math_fns:map(function(fn_name)
  math[fn_name] = terralib.memoize(function(arg_type)
    assert(arg_type:isfloat(), "Math operations support only float types.")
    return math_fn.new_math_fn(fn_name, arg_type)
  end)
end)

cuda_only_math_fns:map(function(fn)
  math[fn.name] = terralib.memoize(function(arg_type)
    assert(arg_type:isfloat(), "Math operations support only float types.")
    local f = math_fn.new_math_fn(fn.name, arg_type)
    f:set_variant("default", fn.default_variant(arg_type))
    return f
  end)
end)

function math.is_math_fn(fn)
  return getmetatable(fn) == math_fn
end

return math
