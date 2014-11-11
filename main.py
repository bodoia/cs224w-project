import sys
sys.path.append('../snappy/')
import snap
from initialization import *
from detection import *
from evaluation import *

def main():
   # TODO
   print evaluateFCC(detectBRIM(initializeFull('file')))

main()
