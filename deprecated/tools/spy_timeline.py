#!/usr/bin/env python3

# Copyright 2024 Stanford University
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

from pyx import *

DEFAULT_LINE_HEIGHT = 200 
DEFAULT_LINE_SPACE = 200

class TimeLine(object):
    def __init__(self, name, start, end):
        self.name = name
        assert start <= end
        self.global_start = start
        self.global_end = end
        self.lines = list()
        self.canvas = canvas.canvas()
        self.line_height = DEFAULT_LINE_HEIGHT
        self.line_space = DEFAULT_LINE_SPACE

    def add_line(self, name):
        idx = len(self.lines)
        self.lines.append(Line(name))
        return idx

    def add_instance(self, line, name, start, end, color):
        assert line < len(self.lines)
        assert start >= self.global_start
        assert end <= self.global_end
        self.lines[line].add_instance(name, start, end, color)

    def add_axis(self, splits):
        pass

    def write_pdf(self, xpixels):
        xscale = (self.global_end - self.global_start)/xpixels
        line_offset = 0
        line_size = self.line_height = self.line_space
        for line in self.lines:
            line.draw(self.canvas, self.global_start, xscale, line_offset, line_size) 
            line_offset = line_offset + line_size
        self.canvas.writePDFfile(self.name)

class Line(object):
    def __init__(self, name):
        self.name = name
        self.instances = list()

    def add_instance(self, name, start, end, color):
        self.instances.append(LineInstance(name, start, end, color))

    def draw(self, canvas, xstart, xscale, ystart, ysize):
        for inst in self.instances:
            inst.draw(canvas, xstart, xscale, ystart, ysize) 

class LineInstance(object):
    def __init__(self, name, start, end, color):
        self.name = name
        self.start = start
        self.end = end
        self.color = color

    def draw(self, canvas, xstart, xscale, ystart, ysize):
        canvas.fill(path.rect((self.start - xstart)/xscale, ystart, (self.end-self.start)/xscale, ysize)) 
      
        

