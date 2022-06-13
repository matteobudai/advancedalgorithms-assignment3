import math
from math import log,sqrt
import time
import copy
import random
import glob
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
        W=[]
        D=[]
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
        #k=int(((n**2)/2)*log(n))
        k = int(math.log(n))**2
        i=0
        j=0
        l=1
        while l <= len(V)*len(V):

            W.append([j+1, i+1, 0])
            i=i+1
            if i==len(V):
                j=j+1
                i=0   
            l=l+1
        u=0
        while u<numEdges:
            d=int(info[u][0])
            f=int(info[u][1])
            if d!=f:

                g=(d-1)*len(V)+f-1
                h=(f-1)*len(V)+d-1

                W[g][2]=int(info[u][2])
                W[h][2]=int(info[u][2])
            u=u+1
        i=0
        u=0
        z=1
        D.append(int(0))
        while i<len(W):
            D[u]=D[u]+W[i][2]
            if z==len(V):
                z=0
                u=u+1
                D.append(int(0))
            z=z+1
            i=i+1
        D.pop()
        return info_int, k, V, W, D

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

'''
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
'''
# Binary search (special case for random_select)
# Return INDEX
def binarySearch(array, x):
    
    low = 0
    high = len(array) - 1
    #mid = (high + low) // 2
    i = 0
    if (x < array[low]):
        return 0
    while low <= high:
        #print(30*"-")
        i+=1
        #print("Iteration: ", i)
        mid = (high + low) // 2 
        #print("low: ",low)
        #print("high: ",high)
        #print("mid: ",mid)
        if x>=array[mid+1]:
            low=mid+1
        else:
            if x>=array[mid]:
                return mid+1
            high=mid-1
    return mid + 1

def Random_Select(C):
    #print()
    #print(30*"-")
    #print("Random_select implementation:")
    K = []
    c = 0
    for i in range(len(C)):
        c += C[i]
        
        K.append(c)
    
    r = random.choice(range(0, K[len(K) - 1]))
    
    #print("r: ", r)
    
    pos = binarySearch(K, r) 
    #print("K: ", K)
    #print("Len K: ", len(K))
    #print("pos: ", pos)
    #print("K[pos]: ", K[pos])
    #print(30*"-")
    #print()
    return pos  

def Edge_Select(V1, D, W):
    U2 = 0 # node 1
    V2 = 0 # node 2 
    
    #print("D: ", D)
    u_ = Random_Select(D) 
    U2 = u_ + 1
    #print("U: ", U)
    
    
    W_ = []
    t = 0

    for i in range(len(W)):
        if W[i][0] == U2:
            #t += W[i][2]
            W_.append(W[i][2])
    
    
    #print("W_: ", W_)
    #print("Len(W_): ", len(W_))
    v_ = Random_Select(W_)
    V2 = v_ + 1
    
    #print("V: ", V)
    #print("Edge: [", U,", ",V,"]")
    return U2,V2


def Contract_Edge(u,v, W, D):
    u=u-1
    v=v-1
    
    D[u]=D[u]+D[v]-2*W[(u)*(len(V))+v][2]
    D[v]=0
    W[(u)*(len(V))+v][2]=0
    W[(v)*(len(V))+u][2]=0

    z=0
    i=0
    while i<len(V):
        if(i!=u and i!=v):     
            W[(u)*(len(V))+z][2]=W[(u)*(len(V))+z][2]+W[(v)*(len(V))+z][2]
            W[(i)*(len(V))+u][2]=W[(i)*(len(V))+u][2]+W[(i)*(len(V))+v][2]
            W[(v)*(len(V))+z][2]=0
            W[(i)*(len(V))+v][2]=0
        z=z+1
        i=i+1   
    
   

def Contract(s, V,  W, D):
    n=len(V)
    i=1
    while i<=n-s:   
        u,v=Edge_Select(V, D, W)      
        Contract_Edge(u, v, W, D)
        V.remove(v)      
        i=i+1
    return V,D,W


def Recursive_Contract(V, W, D):
    n=len(V)
    if n<=6:
        V, D, W=Contract(2, V, W, D)
        return D[V[1]-1]

    t=n/sqrt(2)+1

    V, D, W=Contract( t, V,  W, D)
    w1=Recursive_Contract(V, W, D)
    V, D, W= Contract(t, V,  W, D)
    w2=Recursive_Contract(V, W, D)
    return min(w1,w2)

def Karger(G,k):
    timeout = 60
    start = time.time()

    min1=math.inf

    for i in range(0,k):
        if time.time() - start > timeout:
            print("Timed Out at iteration =", i, " out of k =", k)
            break
        copyV=copy.deepcopy(V)
        copyW=copy.deepcopy(W)
        copyD=copy.deepcopy(D)
        t=Recursive_Contract(copyV, copyW, copyD)

        #t=Full_Contraction(copyG, copyV)
        if t<min1:
            discovery_time = time.time() - start
            min1=t

    print('Minimum:', min1)
    end = time.time()
    time_cost =  end - start
    print('Total Time:', time_cost)
    print('Discovery Time: ', discovery_time)
    
    return min, time_cost, discovery_time


graph, k, V, W, D= Graph().buildGraph(open("r_dataset/input_random_03_10.txt", "r"))
min, time_cost, disc_time= Karger(graph, k)

'''
for filepath in glob.iglob('r_dataset//*.txt'):
    new=Graph()
    graph, k, V, W, D= new.buildGraph(open(filepath, "r"))
    print(filepath)
    min, time_cost, discovery_time= Karger(graph,k)
''' 
 
  
 
  
