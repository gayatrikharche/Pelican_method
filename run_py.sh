#!/bin/bash
tar -xzf my_env.tar.gz
export PYTHONPATH=$PWD/my_env
python3 main.py --use-pelican `grep '^GLIDEIN_Site ' .machine.ad | awk '{print $3}' | cut -c 2- | rev | cut -c 2- | rev`