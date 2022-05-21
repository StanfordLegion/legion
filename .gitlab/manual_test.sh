#!/bin/bash

set -e
set -x

source .gitlab/manual_env_crusher.sh

.gitlab/test_script.sh
