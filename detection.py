# detection.py
# ------------
# Contains functions which implement the various community detection algorithms.
# All functions take the following form:
# Input: snappy graph
# Output: map from user_kookies to ints representing community membership

import sys
sys.path.append('../snappy/')
import snap

# Implements the hierarchical clustering algorithm described in [4]
def detectHierarchical(graph):
   # TODO
   return {}

# Implements the edge-betweenness algorithm described in [4]
def detectBetweenness(graph):
   # TODO
   return {}

# Implements the basic modularity-based algorithm described in [6]
def detectModularity(graph):
   # TODO
   return {}

# Implements the basic spectral algorithm described in [5]
def detectSpectral(graph):
   # TODO
   return {}

# Implements the BRIM algorithm described in [1]
def detectBRIM(graph):
   # TODO
   return {}

# Implements the spectral co-clustering algorithm described in [2]
def detectCocluster(graph):
   # TODO
   return {}
