import networkx as nx
from initialization import *
from detection import *
from evaluation import *

def main():
   #G = nx.read_graphml('Davis.GraphML')
   G = nx.read_edgelist('Scotland.net')
   detectBRIM(G)

main()
