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

# Implements the fraction correctly classified metric described in [3]
def evaluateFCC(detected, groundTruth):
   for node in detected.iterkeys():
      
   return 0

# Implements the Rand index metric described in [3]
def evaluateRI(detected, groundTruth):
   # TODO
   return 0

# Implements the normalized mutual information metric described in [3]
def evaluateNMI(detected, groundTruth):
   # TODO
   return 0
