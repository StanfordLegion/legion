#!/bin/bash
#
# Copyright 2018 Stanford University, NVIDIA Corporation
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

if [[ ! -e $LG_RT_DIR ]]; then
  echo "LG_RT_DIR is not set or does not exist"
  exit 1
fi

ROOT="${LG_RT_DIR}/../doc"
PUBLISH_DIR="${ROOT}/publish"
TOOLS_DIR="${ROOT}/../tools"
FILES="${PUBLISH_DIR}/files.txt"
GLOSSARY_FILE="${PUBLISH_DIR}/glossaryFile.txt"
REMEDIES_DIR="${PUBLISH_DIR}/remedies"
MESSAGES_DIR="${PUBLISH_DIR}/messages"

rm -rf ${PUBLISH_DIR}

mkdir -p ${PUBLISH_DIR}/glossary ${PUBLISH_DIR}/design_patterns ${PUBLISH_DIR}/messages

# cross index the glossary and design_patterns

./crossIndex.bash

ls -1 glossary/markdown/ | sed -e "s/.md//" > "${GLOSSARY_FILE}"


# prepare remedies html

echo Convert remedies
rm -rf ${REMEDIES_DIR}
mkdir -p ${REMEDIES_DIR}
cp remedies/markdown/* ${REMEDIES_DIR}
rm -f .tmp_remedies
ls -1 ${REMEDIES_DIR}/* | sed -e "s://:/:g" | sed -e "s/.md//" > .tmp_remedies
cat .tmp_remedies | sed -e "s:${REMEDIES_DIR}/::" > ${REMEDIES_DIR}/remediesList.txt
while read REMEDY
  do
    rm -f "${REMEDY}.html"
    echo Convert "${REMEDY}" from markdown to html
    pandoc "${REMEDY}.md" >> "${REMEDY}.html" || echo Please install pandoc
    rm "${REMEDY}.md"
  done < .tmp_remedies
wc -l .tmp_remedies
rm -f .tmp_remedies

# collate and cross index the error messages

BLOB=`git rev-parse HEAD`

( cd ${LG_RT_DIR} ; \
	find . -name \*.cc | python "${TOOLS_DIR}/collate_messages.py" \
	--prefix="https://github.com/StanfordLegion/legion/blob/${BLOB}/runtime" \
	--strip=0 \
  --output_dir="${MESSAGES_DIR}" \
	--glossaryFile="${GLOSSARY_FILE}" \
  --glossaryURL="http://legion.stanford.edu/" \
  --legion_config_h="../runtime/legion/legion_config.h" \
  --remediesDir="${REMEDIES_DIR}" \
)

