import math
import matplotlib.pyplot as plt                   
from math import log,sqrt
import time
import copy
import random
import glob
from collections import defaultdict
from matplotlib.lines import Line2D


#Create a graph
class Graph:
    #Create list of lists containing all vertices and weights in format [u,v,weight]
    def buildGraph(self, input):
        lines = input.readlines()
        numEdges = int(lines[0].split()[1])
        numNodes = int(lines[0].split()[1])
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
        #k
        k = int(math.log(numNodes))**2
        # Data structure of W
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
        #Data structure of D
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

    #Calculate asymptotic complexity 
    def asymComplexity(self, input):
        lines = input.readlines(0)
        numNodes = int(lines[0].split()[0]) #n
        numEdges = int(lines[0].split()[1]) #m
        asymComplexity = numNodes**2 * math.log(numNodes)**3 # O ( n^2 log^3(n))
        return numNodes, asymComplexity

# Function: Implement binary search (special for RandomSelect function)
# Input: Сumulative weights vector, random value x (aka r) 
# Return: Number/Name of node
def binarySearch(array, x):
    # set low and high limit by extract from array
    low = 0
    high = len(array) - 1

    # return first value in special case
    if (x < array[low]):
        return 0
    # binary search: 
    while low <= high:
        # update the mid index
        mid = (high + low) // 2 
        # check if first element of right part if equal or higher 
        if x>=array[mid+1]:
            # update lower limit
            low=mid+1
        else:
            # check mid element is require the conditions 
            if x>=array[mid]:
                # return the required index 
                return mid+1
            high=mid-1
    return mid + 1

# Function: Implement of Random Select function
# Input: Array of weights 
# Return: Number/Name of node
def Random_Select(C):
    K = []
    c = 0
    # Build Сumulative weights vector
    for i in range(len(C)):
        c += C[i]
        K.append(c)
    # Select random value for r in range (0, last element of K)
    r = random.choice(range(0, K[len(K) - 1]))
    # run the binary search to get position/node
    pos = binarySearch(K, r)    
    return pos  

# Function: Implement of Edge Select function
# Input: Input 2 arrays: 
# D -- array of sum of all weights connected to node (by index + 1); 
# W -- array of all edges (i.e. [node1, node2, weight])
# Return: Selected edge (i.e. [node1, node2])
def Edge_Select(V1, D, W):
    U2 = 0 # node 1
    V2 = 0 # node 2 
    
    # get node U
    u_ = Random_Select(D) 
    U2 = u_ + 1
    
    W_ = []
    t = 0

    # build Сumulative weights vector for node V
    for i in range(len(W)):
        if W[i][0] == U2:
            #t += W[i][2]
            W_.append(W[i][2])
    
    # get node V
    v_ = Random_Select(W_)
    V2 = v_ + 1

    return U2,V2


# contraction of the edges
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
        #except u and v
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
        #remove the edge
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
    timeout = 90
    start = time.time()
    min1=math.inf
    for i in range(0,k):
        if time.time() - start > timeout:
            #print("Timed Out at iteration =", i, " out of k =", k)
            break
        #copy of the data structure
        copyV=copy.deepcopy(V)
        copyW=copy.deepcopy(W)
        copyD=copy.deepcopy(D)
        #call the contraction
        t=Recursive_Contract(copyV, copyW, copyD)
        if t<min1:
            #update the new min
            discovery_time = time.time() - start
            min1=t

    #print('Minimum:', min1)
    end = time.time()
    time_cost =  end - start
    #print('Total Time:', time_cost)
    #print('Discovery Time: ', discovery_time)
    
    return min1, time_cost, discovery_time

'''
graph, k, V, W, D= Graph().buildGraph(open("r_dataset/input_random_03_10.txt", "r"))
min, time_cost, disc_time= Karger(graph, k)

'''
# array for print graph
results = []
results_x_nodes = []
results_y_timecost = []
results_y_discovertime = []
results_y_asymCompl = []

for filepath in glob.iglob('r_dataset//input_random_*.txt'):
    new=Graph()
    graph, k, V, W, D= new.buildGraph(open(filepath, "r"))
    numNodes, asymCom = new.asymComplexity(open(filepath, "r"))
    #print(filepath)
    min1, time_cost, discovery_time= Karger(graph,k)
    print("File: ",filepath)
    print("Minimum Cut: ",min1)
    if time_cost >= 90:
        print("Execution Time: -Timed Out- ", time_cost)
    else: print("Execution Time: ", time_cost)
    print("Discovery Time: ", discovery_time)
    results.append([numNodes, filepath, min1, round(time_cost,5),asymCom, discovery_time])

#-----plotting----

results.sort(key=lambda x: x[0])

for i in range(0, len(results)):
    results_x_nodes.append(results[i][0])
    results_y_timecost.append(results[i][3])
    results_y_asymCompl.append(results[i][4])
    results_y_discovertime.append(results[i][5])


#complexity comparison

#plot data
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.scatter(results_x_nodes, results_y_timecost, c ="blue")
ax2.plot(results_x_nodes, results_y_asymCompl, c ="green")

#labels
ax1.set_xlabel('Number of Vertices')
ax1.set_ylabel('Execution Time (s)')
ax2.set_ylabel('Asymptotic Complexity (10e6)')
ax1.set_title('Karger Stein - Asymptotic Complexity Comparison')

#legend
bl_circle = Line2D([0], [0], marker='o', color='w', label='Execution Time',
                        markerfacecolor='b', markersize=7)
gr_line = Line2D([0], [0], label='Asymptotic Complexity', color = 'g')
ax1.legend(handles=[bl_circle, gr_line])

plt.show()

#time cost

#plot data
plt.plot(results_x_nodes, results_y_timecost, c ="blue")

#labels
plt.xlabel('Number of Vertices')
plt.ylabel('Execution Time (s)')
plt.title('Karger Stein - Execution Time')

#legend
plt.show()

#execution discovery comparison

#plot data
fig, ax1 = plt.subplots()
ax1.plot(results_x_nodes, results_y_timecost, c ="blue")
ax1.plot(results_x_nodes, results_y_discovertime, c ="green")

#labels
ax1.set_xlabel('Number of Vertices')
ax1.set_ylabel('Time (s)')
ax1.set_title('Karger Stein- Execution vs Discovery Time Comparison')

#legend
bl_circle = Line2D([0], [0], label='Execution Time', color = 'b')
gr_line = Line2D([0], [0], label='Discovery Time', color = 'g')
ax1.legend(handles=[bl_circle, gr_line])

plt.show()
