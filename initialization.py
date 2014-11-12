# initialization.py
# ------------
# Exports functions for converting the raw data into snappy graphs. All
# conversion from global_hash and user_kookie strings to integer node ids is
# handled internally here. Exported functions take the following form:
# Input: text file containing data from Bitly
# Output: tuple containing (snappy graph, map from node ids to ints representing ground truth communities)

import sys
sys.path.append('../snappy/')
import snap

# Returns a map from user kookies to ints representing ground truth communities
def __getOriginalGroundTruth(fileName):
   # TODO parse the file containing ground truth information into a map
   return {} 

# Returns a map from node ids to ints representing ground truth communities
def __getConvertedGroundTruth(originalGroundTruth, snapMap):
   # TODO 
   return {}

# Returns the complete bipartite graph from a given input file
def initializeFull(fileName):
   snapMap = snap.TStrIntH()
   graph = snap.LoadEdgeListStr(snap.PUNGraph, fileName, 0, 1, snapMap)
   originalGroundTruth = __getOriginalGroundTruth(fileName)
   groundTruth = getConvertedGroundTruth(originalGroundTruth, snapMap)
   return (graph, groundTruth)

# Returns the induced unipartite graph from a given input file
def initializeInduced(fileName):
   # TODO
   graph = snap.TUNGraph.New()
   return (graph, {})
