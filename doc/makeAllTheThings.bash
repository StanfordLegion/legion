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
#
#!/bin/bash

ROOT="${LG_RT_DIR}/../doc"
PUBLISH_DIR="${ROOT}/publish"
TOOLS_DIR="${ROOT}/../tools"
FILES="${PUBLISH_DIR}/files.txt"
GLOSSARY_FILE="${PUBLISH_DIR}/glossaryFile.txt"
MESSAGE_DIR="${PUBLISH_DIR}/messages"

rm -rf ./publish

# cross index the glossary and design_patterns

./crossIndex.bash

# prepare messages html

echo Convert messages
rm -rf ./publish/messages/
mkdir -p ./publish/messages/
cp messages/markdown/* ./publish/messages/
rm -f .tmp_messages
ls -1 ./publish/messages/* | sed -e "s://:/:g" > .tmp_messages
cat .tmp_messages | sed -e "s:./publish/messages/::" > ./publish/messages/messageList.txt
while read MESSAGE
  do
    rm -f "${MESSAGE}.html"
    echo Convert "${MESSAGE}" from markdown to html
    pandoc "${MESSAGE}" >> "${MESSAGE}.html" || echo Please install pandoc
    rm "${MESSAGE}"
  done < .tmp_messages
wc -l .tmp_messages
rm -f .tmp_messages

# collate and cross index the error messages

( cd ${LG_RT_DIR} ; \
  BLOB=`git -C legion rev-parse HEAD` ;\
	find . -name \*.cc | python "${TOOLS_DIR}/collate_messages.py" \
	--prefix="https://github.com/StanfordLegion/legion/blob/${BLOB}/runtime" \
	--strip=0 \
  --output_dir="${PUBLISH_DIR}" \
	--glossaryFile="${GLOSSARY_FILE}" \
  --glossaryURL="http://legion.stanford.edu/" \
  --legion_config_h="../runtime/legion/legion_config.h" \
  --messageDir="${MESSAGE_DIR}" \
)

