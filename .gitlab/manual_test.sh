#!/bin/bash

set -e
set -x

source .gitlab/manual_env.sh

.gitlab/test_script.sh
