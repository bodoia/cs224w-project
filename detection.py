# detection.py
# ------------
# Exports functions which implement the various community detection algorithms.
# All exported functions take the following form:
# Input: NetworkX graph
# Output: NetworkX graph with detected communities stored as 'detected' attribute

import networkx as nx
import scipy as sp
from scipy import sparse
import numpy as np
import random as random
from networkx.algorithms import bipartite

num_clust = 10

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
   # Compute adjacency matrix A
   A = nx.adjacency_matrix(G, nodelist=S)
   print "Computed A."

   # Compute modularity matrix B
   degrees = np.matrix([G.degree(i) for i in S])
   print "Computed degrees."
   D = np.transpose(degrees).dot(degrees)
   print "Computed D."
   B = A - (D / float(2*G.size()))
   print "Computed B."

   # Compute generalized modularity matrix Bg
   diag = np.sum(B, axis=1).getA1()
   print "Computed diag."
   Bg = B - np.diag(diag)
   print "Computed Bg."

   return Bg

# Helper function which computes the modularity-maximizing division for a generalized modularity matrix Bg
def _getDivisionFromGeneralizedModularityMatrix(Bg):
   (eigenVals, eigenVecs) = np.linalg.eigh(Bg)
   print "Computed eigenvals."
   maxIndex = sp.argmax(eigenVals)
   print "Computed maxIndex."
   maxVec = eigenVecs[:,maxIndex]
   print "Computed maxVec."
   s = [1 if x > 0 else -1 for x in maxVec]
   s = np.array(s)
   #s = np.ceil(maxVec) * 2 - 1
   print "Computed s."
   return s

# Implements the basic modularity-based algorithm described in [6]
def detectModularity(G):
   clusters = {1 : range(G.order())}
   while len(clusters) < num_clust:
      print "Computed {0} clusters so far.".format(len(clusters))
      bestDeltaQ, bestCluster, bestS = 0, None, None
      for clusterKey in clusters:
         Bg = _getGeneralizedModularityMatrix(G, clusters[clusterKey])
         s = _getDivisionFromGeneralizedModularityMatrix(Bg)
         deltaQ = s.dot(Bg).dot(s) / float(4 * G.size())
         deltaQ = deltaQ[0,0]

         if deltaQ > bestDeltaQ:
            bestDeltaQ = deltaQ
            bestCluster = clusterKey
            bestS = s 

      if bestDeltaQ > 0:
         posSplit, negSplit = [], []
         for i in range(len(clusters[bestCluster])):
            if bestS[i] == 1:
               posSplit.append(clusters[bestCluster][i]) 
            else:
               negSplit.append(clusters[bestCluster][i]) 
         clusters[bestCluster] = posSplit
         clusters[len(clusters) + 1] = negSplit
      else:
         break

   return clusters

# Implements the basic spectral algorithm described in [5]
def detectSpectral(G):
   # TODO Arjun
   return G

def _getBipartition(G):
   topSet, botSet = bipartite.sets(G)
   topSet, botSet = list(topSet), list(botSet)
   topIndices, botIndices = [], []
   V = G.nodes()
   for i in range(G.order()):
      if V[i] in topSet:
         topIndices.append(i)
      else:
         botIndices.append(i) 
   print "Computed bipartition."
   return topIndices, botIndices

def _getRandomClusters(m, c):
   row = np.zeros((m, c))
   for i in range(m):
      row[i, random.randint(0,c-1)] = 1.
   print "Computed random clusters."
   return row

def _getNewRT(RTBar):
   maxes = np.argmax(RTBar, axis=1).flatten()
   newRT = np.zeros(RTBar.shape)
   newRT[np.arange(RTBar.shape[0]), maxes] = 1
   #print "Computed newRT."
   return newRT

def _getBipartiteModularityMatrix(G, V1, V2):
   V = G.nodes()

   # Compute adjacency matrix A
   A = nx.adjacency_matrix(G)
   print "Computed A."

   # Compute modularity matrix B
   degrees = np.matrix([G.degree(i) for i in V])
   print "Computed degrees."
   D = np.transpose(degrees).dot(degrees)
   print "Computed D."
   B = A - (D / float(G.size()))
   print "Computed B."
   return B[[[x] for x in V1], V2]

# Implements the BRIM algorithm described in [1]
def detectBRIM(G):
   V1, V2 = _getBipartition(G) # List of the p and q node indices in two sides of graph
   Bbar = _getBipartiteModularityMatrix(G, V1, V2)
   BbarT = np.transpose(Bbar)
   maxR, maxT, maxQ = None, None, 0
   count = 0
   while count < 1:
      count += 1
      print count
      R = _getRandomClusters(len(V1), num_clust)
      T = _getRandomClusters(len(V2), num_clust)
      Q = np.trace(sp.transpose(R).dot(Bbar).dot(T)) / float(G.size())
      while True:
         newR = _getNewRT(Bbar.dot(T))
         newT = _getNewRT(BbarT.dot(newR))
         newQ = np.trace(np.transpose(newR).dot(Bbar).dot(newT)) / float(G.size())
         if newQ <= Q:
            break
         R, T, Q = newR, newT, newQ
      if Q > maxQ:
         maxR, maxT, maxQ = R, T, Q
         print "Found new max {0} on iteration {1}".format(Q, count)
   clusters = {}
   clusterIds = np.argmax(maxR, axis=1).flatten()
   for i in range(len(V1)):
      clusters[V1[i]] = clusterIds[i]
   return clusters

# Implements the spectral co-clustering algorithm described in [2]
def detectCocluster(G):
   # TODO Arjun
   return G
