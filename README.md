cs224w-project
==============
We can use the readme to coordinate who's working on what. I figured a good plan of action is to do all of the milestone stuff in python as a first pass, then if we want to we can rewrite it in something faster like C++ (or Fortran :D) for the final project. Right now the organizational structure is as follows:
main.py : the main method for the program
initialization.py : contains functions for converting the raw data into snappy graphs
detection.py : contains functions which implement the various community detection algorithms
evaluation.py : contains functions for evaluating the accuracy of a specific community partitioning
