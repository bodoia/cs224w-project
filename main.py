import networkx as nx
from initialization import *
from detection import *
from evaluation import *
import time

def main():
   #G = nx.karate_club_graph()
   #G = nx.read_graphml('Davis.GraphML')
   #G = nx.read_edgelist('Scotland.net')
   (G, groundTruth, sites, users) = fastInitializeBipartite()
   G = initializeUnipartite(G, sites.values(), True)
   print "Loaded graph with {0} nodes and {1} edges".format(G.order(), G.size())
   start = time.clock()
   clusters = detectModularity(G)
   #clusters = detectBRIM(G)
   end = time.clock()
   print "Time elasped (seconds):", end - start
   inv = {}
   for k,v in clusters.iteritems():
      inv[v] = inv.get(v, set([]))
      inv[v].add(k)
   print "Cluster sizes:", [len(cluster) for cluster in inv.itervalues()]
   print "BCMA:", evaluateBCMA(clusters, groundTruth)
   print "RI:", evaluateRI(clusters, groundTruth)
   print "NMI:", evaluateNMI(clusters, groundTruth)
   print "JI:", evaluateJI(clusters, groundTruth)

main()
