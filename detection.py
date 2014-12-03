# detection.py
# ------------
# Exports functions which implement the various community detection algorithms.
# All exported functions take the following form:
# Input: NetworkX graph
# Output: NetworkX graph with detected communities stored as 'detected' attribute

import networkx as nx
import scipy as sp

# Implements the hierarchical clustering algorithm described in [4]
def detectHierarchical(G):
   # TODO Laura
   return G

# Implements the edge-betweenness algorithm described in [4]
def detectBetweenness(G):
   # TODO Laura
   return G

# Helper function which computes the generalized modularity matrix of a subset of nodes S in a graph G
def _getGeneralizedModularityMatrix(G, S):
   V = G.nodes()

   # Compute adjacency matrix A
   A = nx.adjacency_matrix(G)

   # Compute modularity matrix B
   B = A.copy()
   for i in range(len(V)):
      for j in range(len(V)):
         B[i][j] -= G.degree(V[i]) * G.degree(V[j]) / (2*G.size())

   # Compute generalized modularity matrix Bg
   Bg = B.copy()
   for i in range(len(V)):
      for k in range(len(V)):
         if S[k] == 1:
            Bg[i][i] -= B[i][k]

   return Bg

# Implements the basic modularity-based algorithm described in [6]
def detectModularity(G):
   clusters = {}

   return G

# Implements the basic spectral algorithm described in [5]
def detectSpectral(G):
   # TODO Arjun
   return G

# Implements the BRIM algorithm described in [1]
def detectBRIM(G):
   # TODO Bodoia
   return G

# Implements the spectral co-clustering algorithm described in [2]
def detectCocluster(G):
   # TODO Arjun
   return G
