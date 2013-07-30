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
### Union Find Data Structure
###

class UnionFind:
    def __init__(self):
        self.forest = {}
    def add(self, k):
        if k not in self.forest:
            self.forest[k] = k
        return k
    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        self.forest[root_b] = root_a
        return root_a
    def find(self, k):
        if k not in self.forest:
            self.forest[k] = k

        # Find the root.
        root = k
        while root != self.forest[root]:
            root = self.forest[root]

        # Path compression.
        node = k
        while node != self.forest[node]:
            self.forest[node] = root
            node = self.forest[node]

        return root
