# initialization.py
# ------------
# Contains functions for converting the raw data into snappy graphs.
# All functions take the following form:
# Input: text file containing data from Bitly
# Output: snappy graph

import sys
sys.path.append('../snappy/')
import snap

# Returns the complete bipartite graph from a given input file
def initializeFull(fileName):
   # TODO
   graph = snap.TUNGraph.New()
   return graph

# Returns the induced unipartite graph from a given input file
def initializeInduced(fileName):
   # TODO
   graph = snap.TUNGraph.New()
   return graph
