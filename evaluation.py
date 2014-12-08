# evaluation.py
# -------------
# Exports functions for evaluating the accuracy of a specific community
# partitioning. All exported functions take the following form:
# Input: detected - map from node ids to ints representing detected community membership
#        groundTruth - map from node ids to ints representing ground truth community membership
# Output: real representing score

import sys
sys.path.append('../snappy/')
import snap
import numpy
import math
import networkx as nx

def _invertDict(d):
   inv = {}
   for k,v in d.iteritems():
      inv[v] = inv.get(v, set([]))
      inv[v].add(k)
   return inv

#a probability distribution is a hash of x: P(x)
def _entropy(P):
   ent = 0
   for x in P.itervalues():
      if x != 0:
         ent -= x*math.log(x)/math.log(2)
   return ent

# Implements best cluster-matching error of [3]
def evaluateBCME(detected, groundTruth):
   detClust = _invertDict(detected)
   trueClust = _invertDict(groundTruth)
   
   detKeys = detClust.keys()
   trueKeys = trueClust.keys()
   
   l = len(detClust)
   
   clusts = nx.Graph()
   for i in range(l + len(trueClust)):
      clusts.add_node(i)
   
   for i in range(l):
      for j in range(len(trueClust)):
         clusts.add_edge(i, l+j, weight=len(detClust[detKeys[i]] & detClust[detKeys[j]]))
   
   mate = nx.max_weight_matching(clusts)
   matched = 0
   for v in mate:
      matched = matched + clusts[v][mate[v]]['weight']
   return float(matched)/(2*len(detected))

# Implements (somewhat improperly) the fraction correctly classified metric described in [3]
def evaluateFCC(detected, groundTruth):
   cc = 0 #correctly classified
   
   trueClust = _invertDict(groundTruth)
   
   for n in detected.iterkeys():
      same = 0
      diff = 0
      for partner in trueClust[groundTruth[n]]:
         if n == partner:
            continue
         if detected[n] == detected[partner]:
            same += 1
      
      if same >= 0.5*(len(trueClust[groundTruth[n]]) - 1):
         cc += 1
   
   return float(cc)/len(detected)

# Implements the Rand index metric described in [3]
def evaluateRI(detected, groundTruth):
   trueClust = _invertDict(groundTruth)
   detectedClust = _invertDict(detected)
   
   a11 = 0
   a10 = 0 #true partners, but not detected partners
   a01 = 0 #not detected partners, but true partners
   
   # counting over all n(n-1) (i,j) for i != j
   
   for n in detected.iterkeys():
      correctPartners = 0
      for truePartner in trueClust[groundTruth[n]]:
         if truePartner == n:
            continue
         if detected[n] == detected[truePartner]:
            correctPartners += 1
      a11 += correctPartners
      a01 += len(trueClust[groundTruth[n]]) - 1 - correctPartners
      a10 += len(detectedClust[detected[n]]) - 1 - correctPartners
   
   numNodes = len(detected)
   a00 = numNodes*(numNodes - 1) - a11 - a10 - a01
   return float(a11 + a00)/(a11 + a01 + a10 + a00)

# Implements the normalized mutual information metric described in [3]
def evaluateNMI(detected, groundTruth):
   trueClust = _invertDict(groundTruth)
   detectedClust = _invertDict(detected)
   
   Pxy = {}
   
   for x in detectedClust.iterkeys():
      for y in trueClust.iterkeys():
         Pxy[(x,y)] = len(detectedClust[x] & trueClust[y])/float(len(detected))
   
   Px = {}
   Py = {}
   for (x,y) in Pxy.iterkeys():
      if x in Px:
         Px[x] += Pxy[(x,y)]
      else:
         Px[x] = Pxy[(x,y)]
      if y in Py:
         Py[y] += Pxy[(x,y)]
      else:
         Py[y] = Pxy[(x,y)]
   
   HX = _entropy(Px)
   HY = _entropy(Py)
   HXY = _entropy(Pxy)
   IXY = HX + HY - HXY
   return 2*IXY/(HX + HY)
