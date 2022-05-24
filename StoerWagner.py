import math
import time
from collections import defaultdict

#Create a graph
class Graph:
    #Create list of lists containing all vertices and weights in format [u,v,weight]
    def buildGraph(self, input):
        lines = input.readlines()
        numEdges = int(lines[0].split()[1])
        lines.pop(0) 
        info = []
        info_int = []
        for iterate in range(numEdges):
            info.append(lines[iterate].split())
        for i in range(0,numEdges): 
            info_int.append([int(info[i][0]), int(info[i][1]), int(info[i][2])])
        return info_int

    #Create number of edges
    def numEdges(self, input):
        lines = input.readlines(0)
        numEdges = int(lines[0].split()[1])
        return numEdges

    #Create number of edges
    def numNodes(self, input):
        lines = input.readlines(0)
        numNodes = int(lines[0].split()[0])
        return numNodes