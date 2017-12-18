#!/bin/bash

root_dir="$(dirname "${BASH_SOURCE[0]}")"

source "$root_dir"/build_vars.sh

"$root_dir"/../../../language/scripts/setup_env.py --prefix="$root_dir"
