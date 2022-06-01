import math
from math import log
import time
import copy
import random
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
        V=[]
        E=[]
        n=0
        for iterate in range(numEdges):
            n=n+1
            info.append(lines[iterate].split())
        for i in range(0,numEdges):
            if (int(info[i][0])) not in V: 
                V.append(int(info[i][0]))
            if (int(info[i][1])) not in V: 
                V.append(int(info[i][1]))
            E.append(int(info[i][2]))
            info_int.append([int(info[i][0]), int(info[i][1]), int(info[i][2])])
        k=int(((n**2)/2)*log(n))
        return info_int, k, V, E

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

def Full_Contraction(G):
    while len(V)>2:
       e=random.choice(G)
       print(e)
       G.remove(e)
       E.remove(e[2])
       V.remove(e[0])
    return len(E)
    

def Karger(G, k):
    start = time.time()
    min=math.inf
    for i in range(0,k):
        copyG=copy.deepcopy(G)
        t=Full_Contraction(copyG)
        if t<min:
            min=t
    end = time.time()
    time_cost =  end - start
    return min, time_cost


graph, k, V, E= Graph().buildGraph(open("r_dataset/input_random_03_10.txt", "r"))
min, time_cost= Karger(graph, k)
