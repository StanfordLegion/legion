#
# Copyright 2017 Stanford University, NVIDIA Corporation
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
# 
#      http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  

README.txt

You probably just want to do "./makeAllTheThings.bash".
This will produce a new set of glossary and design_patterns pages in publish/.
It will also produce a set of collated and cross-referenced error messages in publish/.


glossary/markdown
contains a set of files, each file defines a glossary term, file names are exact (no extensions, no spaces in class names)

design_patterns/markdown
contains a set of files, each file contains a design pattern page, file names are full sentences

makeAllTheThings.bash
runs all of the scripts, no parameters required, results in publish/.

crossIndex.bash
script to automatically cross index the glossary and design patterns, results go into
publish/glossary
publish/design_patterns

crossIndexGlossaryTerms.bash
this script is called by crossIndex.bash

createIndex.bash
this creates publish/glossary/index.html and publish/design_patterns/index.html after the other scripts have run

