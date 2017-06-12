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

PUBLISHED_DIR=$1
OUTPUT_FILE="${PUBLISHED_DIR}/index.html"

echo === Creating ${OUTPUT_FILE} ===

rm -f "${OUTPUT_FILE}"
rm -f .tmp_pages
ls -1 "${PUBLISHED_DIR}" > .tmp_pages

while read PAGE
  do
    echo "<table style=\"width:100%\" border=1>" >> "${OUTPUT_FILE}"
    echo "<tr> ${PAGE} </tr>" >> "${OUTPUT_FILE}"

    rm -f .tmp_thispage
    ls -1 "${PUBLISHED_DIR}/${PAGE}" > .tmp_thispage

    if [[ "${PAGE}" == "glossary" ]]
    then
      ENTRIES_PER_ROW=10
    else
      ENTRIES_PER_ROW=1
    fi
    declare -i ENTRIES
    ENTRIES=0
    while read PAGE_ENTRY
      do
        if [[ ${ENTRIES} -eq 0 ]]
        then
          echo "<tr>" >> "${OUTPUT_FILE}"
        fi
        STRIPPED_PAGE_ENTRY=`echo "${PAGE_ENTRY}" | sed -e "s/.html//"`
        echo "<td><a href=\"${PAGE}/${PAGE_ENTRY}\">${STRIPPED_PAGE_ENTRY}</a></td>" | sed -e "s://:/:g" >> "${OUTPUT_FILE}"
        ENTRIES=$((${ENTRIES}+1))
        if [[ ${ENTRIES} -ge ${ENTRIES_PER_ROW} ]]
        then
          echo "</tr>" >> "${OUTPUT_FILE}"
          ENTRIES=0
        fi
      done < .tmp_thispage

    echo "</table>" >> "${OUTPUT_FILE}"
    echo "<p>" >> "${OUTPUT_FILE}"
  done < .tmp_pages
rm -f .tmp_pages .tmp_thispage

