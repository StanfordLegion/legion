#!/bin/bash

wget https://bootstrap.pypa.io/get-pip.py
python get-pip.py --user
export PATH="$PATH:$HOME/.local/bin"
pip install --user github3.py
