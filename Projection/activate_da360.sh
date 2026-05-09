#!/bin/bash

INPUT_PATH="$1"

conda run -n da360 python RunStitcher.py "$INPUT_PATH"