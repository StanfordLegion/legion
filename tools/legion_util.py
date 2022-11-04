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

# mypy: ignore-errors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import annotations

from inspect import signature, Parameter, _empty
from functools import wraps
from typing import Union, List, Dict, Set, get_type_hints
import sys
import io
import collections
import os

if sys.version_info >= (3,8):
    from typing import get_origin, get_args

type_check_env = os.getenv("USE_TYPE_CHECK")
if type_check_env is None:
    type_check_enabled = False
else:
    type_check_enabled = bool(int(type_check_env))
#type_check_enabled = True
if type_check_enabled:
    print("Type checking is enabled.")

def typeassert(*ty_args, **ty_kwargs):
    def decorate(func):
        # If in optimized mode, disable type checking
        if not type_check_enabled:
            return func

        # Map function argument names to supplied types
        sig = signature(func)
        bound_types = sig.bind_partial(*ty_args, **ty_kwargs).arguments

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_values = sig.bind(*args, **kwargs)
            # Enforce type assertions across supplied arguments
            for name, value in bound_values.arguments.items():
                if name in bound_types:
                    if not isinstance(value, bound_types[name]):
                        raise TypeError(
                            'Function: {}, argument {} must be {}, but it is {}'.format(func, name, bound_types[name], type(value))
                            )
            return func(*args, **kwargs)
        return wrapper
    return decorate

def typecheck36(func):
    if type_check_enabled:
        assert 0, "Please use at least Python 3.8 when enabling type checking"
    return func

def get_annotation_type(annotation_type):
    annotation_type_origin = get_origin(annotation_type)
    # print(annotation_type_origin)
    if annotation_type_origin is Union:
        actual_annotation_type = get_args(annotation_type)
    elif annotation_type_origin is dict:
        actual_annotation_type = annotation_type_origin
    elif annotation_type_origin is list:
        actual_annotation_type = annotation_type_origin
    elif annotation_type_origin is set:
        actual_annotation_type = annotation_type_origin
    elif annotation_type_origin is tuple:
        actual_annotation_type = annotation_type_origin
    elif annotation_type_origin is collections.abc.Callable:
        actual_annotation_type = annotation_type_origin
    else:
        actual_annotation_type = annotation_type
    # print(annotation_type, get_args(annotation_type))
    return actual_annotation_type

def typecheck38(func):
    # If in optimized mode, disable type checking
    global type_check_enabled
    if not type_check_enabled:
        return func
    sig = signature(func)
    annotation_types = get_type_hints(func)
    @wraps(func)
    def wrapper(*args, **kwargs):
        bound_values = sig.bind(*args, **kwargs)
        # Enforce type assertions across supplied arguments
        for name, value in bound_values.arguments.items():
            # annotation_type = sig.parameters[name].annotation
            # if annotation_type == _empty:
            #     continue
            if name not in annotation_types:
                continue
            annotation_type = annotation_types[name]
            # annotation_type = eval(annotation_type)
            actual_annotation_type = get_annotation_type(annotation_type)
            if actual_annotation_type is not Parameter.empty:
                if not isinstance(value, actual_annotation_type):
                    raise TypeError(
                        'Function: {}, argument {} must be {}, but it is {}'.format(func, name, actual_annotation_type, type(value))
                        )
        # print(annotation_types)
        return_value = func(*args, **kwargs)
        if "return" not in annotation_types:
            assert 0, "The return value of Function: " + str(func) + " is not annotated."
        return_annotation_type = annotation_types["return"]
        actual_return_annotation_type = get_annotation_type(return_annotation_type)
        # print(type(return_value), actual_return_annotation_type)
        if not isinstance(return_value, actual_return_annotation_type):
            raise TypeError(
                'Return must be {}, but it is {}'.format(return_annotation_type, type(return_value))
            )
        return return_value
    return wrapper

typecheck = typecheck38 if sys.version_info >= (3,8) else typecheck36
