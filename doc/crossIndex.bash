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
#!/bin/bash

set -e

echo 
echo INITIALIZE
echo

cd "$(dirname "${BASH_SOURCE[0]}")"

rm -rf ./publish
mkdir -p ./publish/glossary ./publish/design_patterns

cp glossary/markdown/* ./publish/glossary
rm -f .tmp_glossary
ls -1 ./publish/glossary/* | sed -e "s://:/:g" > .tmp_glossary
while read GLOSSARY
  do
    rm -f "${GLOSSARY}.html"
    echo "---" >> "${GLOSSARY}.html"
    echo "layout: page" >> "${GLOSSARY}.html"
    echo "---" >> "${GLOSSARY}.html"
    echo "<a href=\"../index.html\">index</a><p>" >> "${GLOSSARY}.html"
    echo Convert "${GLOSSARY}" from markdown to html
    pandoc "${GLOSSARY}" >> "${GLOSSARY}.html" || echo Please install pandoc
    rm "${GLOSSARY}"
  done < .tmp_glossary
wc -l .tmp_glossary
rm -f .tmp_glossary

cp design_patterns/markdown/* ./publish/design_patterns
rm -f .tmp_design_patterns
ls -1 ./publish/design_patterns/* | sed -e "s://:/:g" > .tmp_design_patterns
while read DESIGN_PATTERN
  do
    rm -f "${DESIGN_PATTERN}.html"
    echo "---" >> "${DESIGN_PATTERN}.html"
    echo "layout: page" >> "${DESIGN_PATTERN}.html"
    echo "---" >> "${DESIGN_PATTERN}.html"
    echo "<a href=\"../index.html\">index</a><p>" >> "${DESIGN_PATTERN}.html"
    echo Convert "${DESIGN_PATTERN}" from markdown to html
    pandoc "${DESIGN_PATTERN}" >> "${DESIGN_PATTERN}.html" || echo Please install pandoc
    rm "${DESIGN_PATTERN}"
  done < .tmp_design_patterns
wc -l .tmp_design_patterns
rm -f .tmp_design_patterns

echo 
echo ADD GLOSSARY TERMS TO GLOSSARY
echo 

(
cd publish 
../crossIndexGlossaryTerms.bash glossary/ ../glossary/markdown/ glossary/ "<p> See also:" true true
)

echo 
echo ADD GLOSSARY TERMS TO DESIGN PATTERNS
echo 

(
cd publish
../crossIndexGlossaryTerms.bash design_patterns/ ../glossary/markdown/ glossary/ "<p> See also:" false true
)

echo 
echo ADD DESIGN PATTERNS TO GLOSSARY
echo 

(
cd publish
../crossIndexGlossaryTerms.bash glossary/ ../glossary/markdown/ design_patterns/ "<p> Relevant design patterns:" true false
)

echo 
echo CREATE THE INDEX PAGE
echo 

./createIndexPage.bash 

echo 
echo DONE
echo 

