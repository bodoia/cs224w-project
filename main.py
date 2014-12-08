import networkx as nx
from initialization import *
from detection import *
from evaluation import *

def main():
   #G = nx.read_graphml('Davis.GraphML')
   #G = nx.read_edgelist('Scotland.net')
   (G, groundTruth, sites, users) = fastInitializeBipartite()
   print "Loaded graph with {0} nodes and {1} edges".format(G.order(), G.size())
   #detectBRIM(G)

main()
