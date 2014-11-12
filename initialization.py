# initialization.py
# ------------
# Contains functions for converting the raw data into snappy graphs.
# All functions take the following form:
# Input: text file containing data from Bitly
# Output: tuple containing (snappy graph, map from node ids to ints representing ground truth communities)

import sys
sys.path.append('../snappy/')
import snap

# Returns the complete bipartite graph from a given input file
def initializeFull(fileName):
   snapMap = snap.TStrIntH()
   graph = snap.LoadEdgeList(snap.PUNGraph, fileName, 0, 1, snapMap)
   return (graph, mapping)

# Returns the induced unipartite graph from a given input file
def initializeInduced(fileName):
   # TODO
   snapMap = snap.TStrIntH()
   graph = snap.LoadEdgeListStr(snap.PUNGraph, fileName, 0, 1, snapMap)
   return (graph, mapping)
