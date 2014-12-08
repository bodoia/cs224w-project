import networkx as nx
import sys
import operator
import itertools
import cPickle as pickle

## Use:
#  	(G, groundTruth, sites, users) = initializeBipartite()
#  	U = initializeUnipartite(G, sites.values(), True)



#### reads in the edge list for the connected component w/ our chosen topics that we pre-processed.
#### returns four things:
###(1) A networkx graph where nodes are users and sites and edges exist betweer a user and a site they clicked on
###(2) A 'ground truth' map. Which maps  (nodeID) -> (topic). Where the nodeID is an integer, and topic is a string (i.e. 'world')
###(3) A 'sites' map. Which maps (siteHash) -> (nodeID). Where the site hash is the string global hash And the node ID is the integer
###(4) A 'users' map. Maps (kookie) -> (nodeID)
def initializeBipartite():
	G=nx.Graph()
	sites = {}
	users = {}
	numSites = 0
	numUsers = 0

	with open("finalEdgeList.txt") as f:
		for line in f.read().splitlines():
			(site, user) = line.split() 
			if site not in sites.keys():
				sites[site.strip()] = numSites
   				G.add_node(site.strip())
   				numSites +=1
   			if user not in users.keys():
   				users[user.strip()] = numUsers
   				G.add_node(user.strip())
   				numUsers += 1
   			G.add_edge(site, user, capacity=1)

   	for key, value in users.items():
   		users[key] = value + numSites

   	H = nx.Graph()
   	for edge in G.edges():
   		if (edge[0] in sites.keys()):   
   			sourceID = sites[edge[0]]
   			destID = users[edge[1]]
   		elif (edge[0] in users.keys()):
   			destID = sites[edge[1]]
   			sourceID = users[edge[0]]

   		if (edge[0] not in H.nodes()):  
   			H.add_node(sourceID)
   		if (edge[1] not in H.nodes()):
   			H.add_node(destID)
   		H.add_edge(sourceID, destID, capacity=1)

	groundTruth = {}
 	with open("topics.txt") as f4:
 		for line in f4.read().splitlines():
 			(site, topic) = line.split()
 			if site in sites.keys():  
 				groundTruth[sites[site]] = topic

   # ADDED BY BODOIA - save the graph, groundtruth, sites and users as pickles for faster loading
 	nx.write_gpickle(H, "graph.p")
 	pickle.dump(groundTruth, open("groundTruth.p", "wb"))
 	pickle.dump(sites, open("sites.p", "wb"))
 	pickle.dump(users, open("users.p", "wb"))
 	print "Pickled graph data"

 	return (H, groundTruth, sites, users)

# ADDED BY BODOIA - load the graph, groundtruth, sites and users from pickles
def fastInitializeBipartite():
   G = nx.read_gpickle("graph.p")
   groundTruth = pickle.load(open("groundTruth.p", "rb"))
   sites = pickle.load(open("sites.p", "rb"))
   users = pickle.load(open("users.p", "rb"))
   print "Loaded graph data from pickle"
   return (G, groundTruth, sites, users)

## Takes in a bipartite graph and constructs a unipartite graph where the nodes are the values
## in the included array 'nodeType' and edges represent shared users or sites respectively
## If weighted == True we weight each edge by how many co-occurences there are
def initializeUnipartite(G, nodeType, weighted):
	U = nx.Graph()

	if weighted==False:
		for node in nodeType:
			U.add_node(node)
			for neighbors in G.neighbors(node):
				U.add_edges_from([(node,dest) for dest in G.neighbors(neighbors) if dest != node])
	elif weighted==True:
		for node in nodeType:
			U.add_node(node)
			for neighbors in G.neighbors(node):
				for dest in G.neighbors(neighbors):
					if dest != node:
						if U.has_edge(node, dest):
							U[node][dest]['capacity'] += .5
						else:
							U.add_edge(node, dest, capacity=.5)
	return U





