cs224w-project
==============
Okay, so we're going to just do everything in Python and forget about speed. We're also going to use NetworkX instead of Snap.py since it supports weighted edges. Right now the organizational structure is as follows:

main.py : the main method for the program

initialization.py : contains functions for converting the raw data into snappy graphs

detection.py : contains functions which implement the various community detection algorithms

evaluation.py : contains functions for evaluating the accuracy of a specific community partitioning
