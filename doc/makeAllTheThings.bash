#!/bin/bash
#
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

if [[ ! -e $LG_RT_DIR ]]; then
  echo "LG_RT_DIR is not set or does not exist"
  exit 1
fi

ROOT="${LG_RT_DIR}/../doc"
PUBLISH_DIR="${ROOT}/publish"
TOOLS_DIR="${ROOT}/../tools"
FILES="${PUBLISH_DIR}/files.txt"
MESSAGES_DIR="${PUBLISH_DIR}/messages"

rm -rf ${PUBLISH_DIR}

mkdir -p ${PUBLISH_DIR}/messages

# collate and cross index the error messages

BLOB=`git rev-parse HEAD`

( cd ${LG_RT_DIR} ; \
	find . -name \*.cc | python3 "${TOOLS_DIR}/collate_messages.py" \
	--prefix="https://github.com/StanfordLegion/legion/blob/${BLOB}/runtime" \
	--strip=0 \
  --output_dir="${MESSAGES_DIR}" \
  --legion_config_h="../runtime/legion/legion_config.h" \
)

