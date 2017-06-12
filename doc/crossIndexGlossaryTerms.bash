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

TARGET_DIR=$1 # these files will be scanned and modified
SOURCE_DIR=$2 # design patterns dir
GLOSSARY_DIR=$3 # filenames are glossary terms
IS_DESIGN_PATTERN=$4 # false for glossary, true for design patterns
LEGEND=$5 # "See also:" 

echo === Indexes will be appended to files in ${TARGET_DIR}
echo === Indexes will point to files in ${SOURCE_DIR}
echo === Indexed terms are filenames in ${GLOSSARY_DIR}
echo === Design patterns? ${IS_DESIGN_PATTERN}
echo === Legend ${LEGEND}


rm -f .tmp_targets .tmp_glossary .tmp_legends
if [[ "${IS_DESIGN_PATTERN}" == "true" ]]
then
  ls -1 ${SOURCE_DIR}/* > .tmp_targets
  EFFECTIVE_SOURCE_DIR="${SOURCE_DIR}"
else
  ls -1 ${TARGET_DIR}/* > .tmp_targets
  EFFECTIVE_SOURCE_DIR="../glossary"
fi
ls -1 ${GLOSSARY_DIR} > .tmp_glossary

echo "" > .tmp_legends

while read SOURCE_FILE
  do
    SOURCE_FILE=`echo "${SOURCE_FILE}" | sed -e "s://:/:g"`
    echo "Process ${SOURCE_FILE}" 
    THIS_LEGEND=${LEGEND}
    while read GLOSSARY_TERM
      do
        FOUND_IN_SOURCE=`grep "${GLOSSARY_TERM}" "${SOURCE_FILE}"`
        if [[ "${FOUND_IN_SOURCE}" != "" ]]
        then
          IS_SELF=`echo "${SOURCE_FILE}" | grep "${GLOSSARY_TERM}"`
          if [[ ${IS_SELF} && "${IS_DESIGN_PATTERN}" == "false" ]]
          then
            echo > /dev/null
          else
            FILENAME=$(basename "${SOURCE_FILE}")
            if [[ "${IS_DESIGN_PATTERN}" == "true" ]]
            then
              TARGET_FILE="${TARGET_DIR}/${GLOSSARY_TERM}.html"
              TEXT="<a href=\"../design_patterns/${FILENAME}\">${FILENAME}</a>" 
              if [[ `grep "${GLOSSARY_TERM}" .tmp_legends` ]]
              then
                echo > /dev/null
              else
                echo "${THIS_LEGEND}" >> "${TARGET_FILE}"
                echo "${GLOSSARY_TERM}" >> .tmp_legends
              fi
            else
              TARGET_FILE="${TARGET_DIR}/${FILENAME}"
              TEXT="<a href=\"${EFFECTIVE_SOURCE_DIR}/${GLOSSARY_TERM}.html\">${GLOSSARY_TERM}</a>" 

              if [[ "${THIS_LEGEND}" != "" ]]
              then
                echo "${THIS_LEGEND}" >> "${TARGET_FILE}"
                THIS_LEGEND=
              fi
            fi
            echo "${TEXT}" | sed -e "s://:/:g" >> "${TARGET_FILE}"
          fi
        else
          echo FOUND_IN_SOURCE ${FOUND_IN_SOURCE}
          echo GLOSSARY_TERM ${GLOSSARY_TERM}
          echo SOURCE_FILE ${SOURCE_FILE}
        fi
      done < .tmp_glossary
  done < .tmp_targets
rm -f .tmp_targets .tmp_glossary .tmp_legends

