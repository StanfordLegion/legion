#!/bin/bash

# Copyright 2023 Stanford University, NVIDIA Corporation
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

ROOT=.

TARGET_DIR=$1 # these files will be scanned and modified
GLOSSARY_DIR=$2 # filenames are glossary terms
REFERENCE_DIR=$3 # references will point to here
LEGEND=$4 # "<p>See also:"
TARGET_IS_GLOSSARY=$5 # "true" or "false"
REFERENCE_IS_GLOSSARY=$6 # "true" or "false"


# Generate all references
#

TMP=${ROOT}/.tmp
REFERENCES=${TMP}/references
TARGETS=${TMP}/targets
GLOSSARY_TERMS=${TMP}/glossary_terms

rm -rf ${TMP}
mkdir -p ${TMP}

# find all references in files in target dir

ls -1 "${GLOSSARY_DIR}" > "${GLOSSARY_TERMS}"
if [[ "${REFERENCE_IS_GLOSSARY}" == "true" ]]
then
  ls -1 "${TARGET_DIR}"/* | sed -e "s://:/:g" > "${TARGETS}"
  echo === Files will be indexed from ${TARGET_DIR}
else
  ls -1 "${REFERENCE_DIR}"/* | sed -e "s://:/:g" > "${TARGETS}"
  echo === Files will be indexed from ${REFERENCE_DIR}
fi

echo === Indexed terms are filenames in ${GLOSSARY_DIR}
echo === References will point to filenames in ${REFERENCE_DIR}
echo === Legend ${LEGEND}
echo === TARGET_IS_GLOSSARY ${TARGET_IS_GLOSSARY}
echo === REFERENCE_IS_GLOSSARY ${REFERENCE_IS_GLOSSARY}


wc -l ${TMP}/*
mkdir -p ${REFERENCES}


while read TARGET
do
  echo Indexing glossary terms used in "${TARGET}"
  TARGET_NAME=$(basename "${TARGET}")
  while read GLOSSARY_TERM
  do
# do the comparison with no spaces
    GLOSSARY_TERM_NO_SPACES=`echo "${GLOSSARY_TERM}" | sed -e "s/ //g"`
    rm -f .tmp_target
    cat "${TARGET}" | sed -e "s/ //g" > .tmp_target
    FOUND_REFERENCE=`grep -i "${GLOSSARY_TERM_NO_SPACES}" .tmp_target`
    rm -f .tmp_target

    if [[ "${FOUND_REFERENCE}" != "" ]]
    then
# don't let a glossary page refer to itself
      if [[ `echo "${TARGET}" | grep -i "${GLOSSARY_TERM}"` && "${TARGET_IS_GLOSSARY}" == "true" ]]
      then
        echo > /dev/null
      else
        STRIPPED_TARGET_NAME="`echo ${TARGET_NAME} | sed -e 's/.html//' `"

        if [[ "${TARGET_IS_GLOSSARY}" == "true" && "${REFERENCE_IS_GLOSSARY}" == "true" ]]
        then
# modifying glossary with self references
          TARGET_REFERENCES="${REFERENCES}/${TARGET_NAME}"
          HREF="\"<a href=\\\"${GLOSSARY_TERM}.html\\\">${GLOSSARY_TERM}</a>\""
          MODIFIED_FILE=`echo "${TARGET}" | sed -e "s://:/:g"`
          OUT_LINE="echo ${HREF} >> \"${MODIFIED_FILE}\""
          echo "${OUT_LINE}" >> "${TARGET_REFERENCES}"

          TARGET_REFERENCES2="${REFERENCES}/${GLOSSARY_TERM}.html"
          MODIFIED_FILE2=`echo "${REFERENCE_DIR}/${GLOSSARY_TERM}.html" | sed -e "s://:/:g"`
          HREF2="\"<a href=\\\"${TARGET_NAME}\\\">${STRIPPED_TARGET_NAME}</a>\""
          OUT_LINE2="echo ${HREF2} >> \"${MODIFIED_FILE2}\""
          echo "${OUT_LINE2}" >> "${TARGET_REFERENCES2}"

        elif [[ "${TARGET_IS_GLOSSARY}" == "false" && "${REFERENCE_IS_GLOSSARY}" == "true" ]]
         then
# modifying design patterns with references to glossary
          TARGET_REFERENCES="${REFERENCES}/${TARGET_NAME}"
          MODIFIED_FILE=`echo "${TARGET_DIR}/${TARGET_NAME}" | sed -e "s://:/:g"`
          HREF_NAME=`echo "../${REFERENCE_DIR}/${GLOSSARY_TERM}.html" | sed -e "s://:/:g"`
          HREF="\"<a href=\\\"${HREF_NAME}\\\">${GLOSSARY_TERM}</a>\""
          echo "echo ${HREF} >> \"${MODIFIED_FILE}\"" >> "${TARGET_REFERENCES}"

        elif [[ "${TARGET_IS_GLOSSARY}" == "true" && "${REFERENCE_IS_GLOSSARY}" == "false" ]]
        then
# modifying glossary with references to design patterns
          TARGET_REFERENCES="${REFERENCES}/${GLOSSARY_TERM}.html"
          MODIFIED_FILE=`echo "${TARGET_DIR}/${GLOSSARY_TERM}.html" | sed -e "s://:/:g"`
          HREF_NAME=`echo "../${REFERENCE_DIR}/${STRIPPED_TARGET_NAME}.html" | sed -e "s://:/:g"`
          HREF="\"<a href=\\\"${HREF_NAME}\\\">${STRIPPED_TARGET_NAME}</a>\""
          echo "echo ${HREF} >> \"${MODIFIED_FILE}\"" >> "${TARGET_REFERENCES}"

        elif [[ "${TARGET_IS_GLOSSARY}" == "false" && "${REFERENCE_IS_GLOSSARY}" == "false" ]]
        then
          echo ERROR THIS CONFIGURATION IS NOT SUPPORTED
          exit
        fi

      fi
    fi
  done < "${GLOSSARY_TERMS}"
done < "${TARGETS}"




# Sort and eliminate duplicates, append to target files
#

for TARGET_REFERENCES in "${REFERENCES}"/*
do
  TARGET_NAME=$(basename "${TARGET_REFERENCES}")
  if [[ "${TARGET_IS_GLOSSARY}" == "true" ]]
  then
    TARGET=`echo "${REFERENCE_DIR}/${TARGET_NAME}" | sed -e "s://:/:g"`
  else
    TARGET=`echo "${TARGET_DIR}/${TARGET_NAME}" | sed -e "s://:/:g"`
  fi
  FILTERED_TARGET_REFERENCES="${TARGET_REFERENCES}.filtered"
  cat "${TARGET_REFERENCES}" | sort | uniq >> .tmp_filtered
  if [[ `wc -l .tmp_filtered | sed -e "s:.tmp_filtered::"` -gt 0 ]]
  then
    LEGEND_TARGET=`echo "${TARGET_DIR}/${TARGET_NAME}" | sed -e "s://:/:g"`
    OUT_LINE="echo \"${LEGEND}\" >> \"${LEGEND_TARGET}\""
    echo "${OUT_LINE}" > "${FILTERED_TARGET_REFERENCES}"
    cat .tmp_filtered >> "${FILTERED_TARGET_REFERENCES}"
    rm .tmp_filtered
    /bin/bash "${FILTERED_TARGET_REFERENCES}"
  fi
done

