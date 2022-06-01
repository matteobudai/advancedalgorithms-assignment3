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
        n=0
        for iterate in range(numEdges):
            n=n+1
            info.append(lines[iterate].split())
        for i in range(0,numEdges):
            if (int(info[i][0])) not in V: 
                V.append(int(info[i][0]))
            if (int(info[i][1])) not in V: 
                V.append(int(info[i][1]))
            info_int.append([int(info[i][0]), int(info[i][1]), int(info[i][2])])
        k=int(((n**2)/2)*log(n))
        return info_int, k, V

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

def Full_Contraction(G,V):
    while len(V)>2:
        e = random.choice(G)
        G.remove(e)
        l=len(G)
        i=0
        while i<l:
            if G[i][0] == e[0]:
               G[i][0]= e[1]
            if G[i][1] == e[0]:
                G[i][1]= e[1]
            i=i+1
        j=0
        while j<l:
            w=0
            while w<l:               
                if j!=w:      
                    if G[j][0]==G[w][0] and G[j][1]==G[w][1]:
                        G[j][2]=G[j][2]+G[w][2]
                        G.remove(G[w])
                        w=w-1
                        l=l-1
                    if G[j][0]==G[w][1] and G[j][1]==G[w][0]:
                        G[j][2]=G[j][2]+G[w][2]
                        G.remove(G[w])
                        w=w-1
                        l=l-1
                w=w+1         
            j=j+1
        V.remove(e[0]) 
    return G[0][2]
    

def Karger(G, k):
    start = time.time()
    min=math.inf
    for i in range(0,k):
        copyV=copy.deepcopy(V)
        copyG=copy.deepcopy(G)
        t=Full_Contraction(copyG, copyV)
        print('Distance founded: ', t)
        if t<min:
            min=t
    print('Minimum:', min)
    end = time.time()
    time_cost =  end - start
    print('Time:', time_cost)
    return min, time_cost


graph, k, V= Graph().buildGraph(open("r_dataset/input_random_03_10.txt", "r"))
min, time_cost= Karger(graph, k)
