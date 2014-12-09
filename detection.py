# detection.py
# ------------
# Exports functions which implement the various community detection algorithms.
# All exported functions take the following form:
# Input: NetworkX graph
# Output: NetworkX graph with detected communities stored as 'detected' attribute

import networkx as nx
import scipy as sp
import numpy as np
import random as random
from networkx.algorithms import bipartite
from scipy import cluster, sparse
from scipy import linalg

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
   V = G.nodes()

   # Compute adjacency matrix A
   A = nx.adjacency_matrix(G)

   # Compute modularity matrix B
   degrees = np.matrix([G.degree(i) for i in V])
   D = np.transpose(degrees).dot(degrees)
   B = A - (D / float(2*G.size()))

   # Compute generalized modularity matrix Bg
   Bg = B
   for i in range(len(V)):
      Bg[i,i] = 0
      for k in S:
         if k != i:
            Bg[i,i] -= B[i,k]

   return Bg[[[x] for x in S],S]

# Helper function which computes the modularity-maximizing division for a generalized modularity matrix Bg
def _getDivisionFromGeneralizedModularityMatrix(Bg):
   (eigenVals, eigenVecs) = np.linalg.eig(Bg)
   maxIndex = sp.argmax(eigenVals)
   maxVec = eigenVecs[:,maxIndex]
   s = [1 if x > 0 else -1 for x in maxVec]
   s = np.array(s)
   return s

# Implements the basic modularity-based algorithm described in [6]
def detectModularity(G):
   clusters = {1 : range(G.order())}
   while len(clusters) < num_clust:
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
# ASSUMES THAT WE WANT TO CLUSTER NODES FROM 0 TO NUMSITES-1
def detectSpectral(G, numSites, num_clusters=num_clust):
   A = nx.to_scipy_sparse_matrix(G, nodelist=range(len(G.nodes())),dtype=float)
   d = A.sum(axis=0)
   Dinvsqrt = sp.sparse.diags(np.squeeze(np.array(1./np.sqrt(d))), 0)
   L = Dinvsqrt*A*Dinvsqrt
   l = np.ceil(np.log(num_clusters)/np.log(2)) #l = num_clust?
   vals, X = sp.sparse.linalg.eigsh(L, k=num_clusters+1, which='LA')
   ourEigs = [pair[0] for pair in sorted(enumerate(vals), key = lambda x:x[1], reverse=True)]
   #X = X with normalized rows? whiten?
   X = X[:,ourEigs[1:]] #eigenvalues 2 to l
   codebook,_ = sp.cluster.vq.kmeans(X, num_clusters)
   clusts,_ = sp.cluster.vq.vq(X, codebook)
   detected = {i: clusts[i] for i in range(numSites)}
   return detected

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
   return topIndices, botIndices

def _getNumClusters(G):
   return 4 

def _getRandomClusters(m, c):
   row = np.zeros((m, c))
   for i in range(m):
      row[i, random.randint(0,c-1)] = 1.
   return row

def _getNewRT(RTBar):
   maxes = np.argmax(RTBar, axis=1)
   newRT = np.zeros(RTBar.shape)
   for i in range(newRT.shape[0]):
      newRT[i, maxes[i,0]] = 1.
   return newRT

def _getBipartiteModularityMatrix(G, V1, V2):
   V = G.nodes()

   # Compute adjacency matrix A
   A = nx.adjacency_matrix(G)

   # Compute modularity matrix B
   degrees = np.matrix([G.degree(i) for i in V])
   D = np.transpose(degrees).dot(degrees)
   B = A - (D / float(G.size()))
   return B[[[x] for x in V1], V2]

# Implements the BRIM algorithm described in [1]
def detectBRIM(G):
   V1, V2 = _getBipartition(G) # List of the p and q node indices in two sides of graph
   Bbar = _getBipartiteModularityMatrix(G, V1, V2)
   maxQ = 0
   while True:
      R = _getRandomClusters(len(V1), num_clust)
      T = _getRandomClusters(len(V2), num_clust)
      Q = np.trace(sp.transpose(R).dot(Bbar).dot(T)) / float(G.size())
      while True:
         newR = _getNewRT(Bbar.dot(T))
         newT = _getNewRT(np.transpose(Bbar).dot(newR))
         newQ = np.trace(np.transpose(newR).dot(Bbar).dot(newT)) / float(G.size())
         if newQ <= Q:
            break
         R, T, Q = newR, newT, newQ
      if Q > maxQ:
         maxQ = Q
         print Q
   return G

# Implements the spectral co-clustering algorithm described in [2]
def detectCocluster(G):
   # TODO Arjun
   return G
