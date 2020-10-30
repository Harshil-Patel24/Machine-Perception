#!/bin/bash
start=$SECONDS
python3 svm.py
python3 assignment.py
duration=$(( SECONDS - start))
echo $duration " seconds"
