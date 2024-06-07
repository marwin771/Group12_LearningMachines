#!/usr/bin/env zsh

python3.9 -m venv .venv
cp -r catkin_ws/src/robobo_interface/src/robobo_interface .venv/lib/python3.9/site-packages/
cp -r catkin_ws/src/data_files/src/data_files .venv/lib/python3.9/site-packages/
pip install -r requirements.txt
