import networkx as nx
from initialization import *
from detection import *
from evaluation import *
import time

def main():
   #G = nx.read_graphml('Davis.GraphML')
   #G = nx.read_edgelist('Scotland.net')
   (G, groundTruth, sites, users) = fastInitializeBipartite()
   print "Loaded graph with {0} nodes and {1} edges".format(G.order(), G.size())
   #detectModularity(G)
   start = time.clock()
   clusters = detectBRIM(G)
   end = time.clock()
   print "Time elasped (seconds):", end - start
   print "BCMA:", evaluateBCMA(clusters, groundTruth)
   print "RI:", evaluateRI(clusters, groundTruth)
   print "NMI:", evaluateNMI(clusters, groundTruth)
   print "JI:", evaluateJI(clusters, groundTruth)

main()
