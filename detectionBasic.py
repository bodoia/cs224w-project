import networkx as nx
import scipy as sp
import numpy
import copy
import sys
import math
import operator
import itertools
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy
from collections import defaultdict
import datetime
#from evaluation import *
from evaluateNew import *
#from initialize import *
from fastInitialize import *

## Get the weight for each pair of nodes i, j where the weight between a pair of nodes is defined
## as the length of the shortest path between them (commented out code is max flow)
def getStartingWeights(G, numNodes, unipartite):
	
	if unipartite == True:
		weights=numpy.zeros((numNodes,numNodes))
		for edge in G.edges():
			source = edge[0]
			dest = edge[1]
			flow = nx.maximum_flow_value(G, source, dest)
			weights[source][dest] = flow
			weights[dest][source] = flow
	else:
		shortestPaths=nx.all_pairs_shortest_path_length(G)
		weights=numpy.zeros((numNodes,numNodes))
		for sourceNode,paths in shortestPaths.items():
				for destNode,pathLength in paths.items():
					weights[sourceNode][destNode]=pathLength
	
	
	Y = squareform(weights)	
	return Y



# Implements the hierarchical clustering algorithm described in [4]
def detectHierarchical(G, numClusters, sites, unipartite, fast):
	numNodes = G.number_of_nodes()
	
	if unipartite == True:
		if fast == True:
			W = pickle.load(open("weightsUnipartite.p", "rb"))
		else:
			W = getStartingWeights(G, numNodes, True)
			pickle.dump(W, open("weightsUnipartite.p", "wb"))
	else:
		if fast == True:
			W = W = pickle.load(open("weightsBipartite.p", "rb"))
		else:
			W = getStartingWeights(G, numNodes, False)
			pickle.dump(W, open("weightsBipartite.p", "wb"))
	
	if unipartite == True:	
		Z=hierarchy.weighted(W)  
		#pickle.dump(Z, open("ZUnipartite.p", "wb"))
		#Z = pickle.load(open("ZUnipartite.p", "rb"))
	else:		
		Z=hierarchy.weighted(W)  
		#pickle.dump(Z, open("ZBipartite.p", "wb"))
		#Z = pickle.load(open"ZBipartite.p", "rb")
	
	membership=list(hierarchy.fcluster(Z,numClusters, 'maxclust')) 

	# print "number of distinct clusters: ", len(set(membership))
	# for i in xrange(len(set(membership))):
	# 	k = 0
	# 	for j in xrange(len(membership)):
	# 		if membership[j] == i+1:
	# 			k+=1
	# 	print k, "nodes in cluster number", i+1
	# 	k=0


	clusters = {}
	for i in xrange(len(membership)):
		if i in sites:
			clusters[i] = membership[i]
	return clusters


# Implements the edge-betweenness algorithm described in [4]
#def detectBetweenness(G, numClusters, Bipartite, sites):
def detectBetweenness(G, numClusters, sites, bipartite):
	Gnew = copy.deepcopy(G)
	numComponents = nx.number_connected_components(G)

	betweenness = nx.edge_betweenness_centrality(Gnew,  weight='capacity')
	pickle.dump(betweenness, open("betweennessUnipartite.p", "wb"))
	#betweenness = pickle.load("betweenessUnipartite.p", "rb")
	
	while (numComponents < numClusters):
		print "num components is now ",  numComponents ### REMEMBER TO DELETE THIS ###

		# calculate betweenness of each edge
		betweenness = nx.edge_betweenness_centrality(Gnew,  weight='capacity')

		## identify and remove the edge with highest betweenness
		max_ = max(betweenness.values())
		for k, v in betweenness.iteritems():
			if float(v) == max_:
				G.remove_edge(k[0], k[1])
		numComponents = nx.number_connected_components(G)

	clusters = {}
	i=0
	j = 0
	for component in list(nx.connected_components(Gnew)):
		for node in component:
			if node in sites:
				clusters[node] = i
				j +=1
		print j, "Nodes in cluster ", i
		j = 0
		i += 1

	return clusters



def printEvaluationResults(G, groundTruth, modelType):
	print "Evaluation results for:", modelType
	print "evaluateBCMA", evaluateBCMA(G, groundTruth)
	print "evaluateRI", evaluateRI(G, groundTruth)
	print "evaluateJI", evaluateJI(G, groundTruth)
	print "evaluateNMI", evaluateNMI(G, groundTruth)


def main():

	# ###### REAL DATA #########
	# (G, groundTruth, sites, users) = initializeBipartite()
	# U = initializeUnipartite(G, sites.values(), True)

	(G, groundTruth, sites, users) = fastInitializeBipartite()
	U = fastInitializeUnipartite()
	numClusters = len(set(groundTruth.values()))
	
	print "Starting Bipartite Hierarchical", datetime.datetime.now()
	hierarchicalResult_G = detectHierarchical(G, numClusters, sites.values(), False, True)
	print "Finishing Bipartite Hierarchical", datetime.datetime.now()

	printEvaluationResults(hierarchicalResult_G, groundTruth, "Bipartite Hierarchical")

	print "Starting Unipartite Hierarchical", datetime.datetime.now()
	hierarchicalResult_U = detectHierarchical(U, numClusters, sites.values(), True, False)
	print "Finishing Unipartite Hierarchical", datetime.datetime.now()
	printEvaluationResults(hierarchicalResult_U, groundTruth, "Unipartite Hierarchical")

	
	print "Starting Bipartite Betweenness", datetime.datetime.now()
	betweennessResult_G = detectBetweenness(G, numClusters)
	print "Finishing Bipartite Betweenness", datetime.datetime.now()
	printEvaluationResults(betweennessResult_G, groundTruth, "Bipartite Betweenness")

	print "Starting Unipartite Betweenness", datetime.datetime.now()
	betweennessResult_U = detectBetweenness(U, numClusters)
	print "Finishing Unipartite Betweenness", datetime.datetime.now()
	printEvaluationResults(betweennessResult_U, groundTruth, "Bipartite Betweenness")







if __name__ == '__main__':
	main()