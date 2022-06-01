import math
from re import T
import time
from collections import defaultdict

from numpy import empty

#Create a node
class Node:
    def __init__(self, tag: int):
        self.tag = tag
        self.key = None
        self.parent = None
        self.isPresent = True
        self.notMerged = True
        self.index = tag-1 # Track the index of the node in the heap instead of using list.index() method which is O(n)
        self.adjacencyList = []

#Create a graph
class Graph:
    def __init__(self):
        self.nodes = defaultdict(Node)

    def createNodes(self, nums: int):
        for i in range(1, nums+1): # nums+1 in order to cover the last node
            self.nodes[i] = Node(i)
    
    def buildGraph(self, input):
        lines = input.readlines()
        self.createNodes(int(lines[0].split()[0]))
        lines.pop(0) 
        for iterate in range(len(lines)):
            info = list(map(int, lines[iterate].split()))
            self.makeNodes(info[0], info[1], info[2])

    #Create all of the nodes
    def makeNodes(self, tag, adjTag, adjCost):
        self.nodes[tag].adjacencyList.append([self.nodes[adjTag], adjCost])
        self.nodes[adjTag].adjacencyList.append([self.nodes[tag], adjCost])

    #Define number of edges
    def numNodes(self, input):
        lines = input.readlines(0)
        numNodes = int(lines[0].split()[0])
        return numNodes

class ArrayHeap(list):
    def __init__(self, array):
        super().__init__(array)
        self.heapSize = len(array)

class MaxHeap:
    def __init__(self, array: list, root: Node):
        self.arrayHeap = ArrayHeap(array)
        
        # Check if the root node is not the first
        # If it is not, reset the starting node by
        # removing the root node from it's original position
        # inserting it in the first position
        # and then update all indexes
        if self.arrayHeap[0] != self.arrayHeap[root.tag-1]: 
            rootNode = self.arrayHeap[root.tag-1]
            self.arrayHeap.remove(rootNode)
            self.arrayHeap.insert(0,rootNode)
            for i in range(0,self.arrayHeap.heapSize):
                self.arrayHeap[i].index = i

    def getParentIndex(self,index):
        if index%2 == 0: 
            return index//2 - 1
        else:
            return index//2

    def getLeftChildIndex(self,index):
        return 2*index+1

    def getRightChildIndex(self,index):
        return 2*index+2

    def maxHeapify(self,i):
        #maxHeapify is always called after we call extractMax(), in order to maintain the max heap structure
        #we always call maxHeapify() FIRST with node 0, then it checks if it needs to perform a swap with the 
        #left or right child, and if so it calls maxHeapify() on the newly swapped max node until nodes are arranged properly 
        #such that the root node is larger than it's children
        #if node 0 is already the maximum node then a swap is not performed 
        l = self.getLeftChildIndex(i)
        r = self.getRightChildIndex(i)
        if l <= self.arrayHeap.heapSize-1 and self.arrayHeap[l].key > self.arrayHeap[i].key:
            max = l
        else:
            max = i
        if r <= self.arrayHeap.heapSize-1 and self.arrayHeap[r].key > self.arrayHeap[max].key:
            max = r
        if max != i:
            self.arrayHeap[i].index, self.arrayHeap[max].index = max, i # Update indexes
            self.arrayHeap[i], self.arrayHeap[max] = self.arrayHeap[max], self.arrayHeap[i]
            self.maxHeapify(max)

    def shiftUp(self, index):
        parent = self.getParentIndex(index)
        current = index
        while current > 0 and self.arrayHeap[parent].key < self.arrayHeap[current].key:
            self.arrayHeap[current].index, self.arrayHeap[parent].index = parent, current # Update indexes
            self.arrayHeap[current], self.arrayHeap[parent] = self.arrayHeap[parent], self.arrayHeap[current]
            current = parent
            parent = self.getParentIndex(parent)

    def extractMax(self):
        #this function both extracts the max and pops it out of the max heap
        # in a max heap data structure, the root node at index 0 will always be the priority(maximum) 
        #so we extract the node at index zero and define it as the max to pass back
        max=self.arrayHeap[0]
        self.arrayHeap[0].isPresent = False
        #then swap the right most node and first(max) node
        self.arrayHeap[0], self.arrayHeap[self.arrayHeap.heapSize-1] = self.arrayHeap[self.arrayHeap.heapSize-1], self.arrayHeap[0]
        self.arrayHeap[0].index = 0
        #reduce the heapSize by 1 to pop out the min node
        self.arrayHeap.heapSize -=1
        #finally call minheapify() to restructure/maintain the min heap data structure
        self.maxHeapify(0)
        return max
        

def stMinCut(G: Graph):
    a = G.nodes.get(1)

    for u in G.nodes.values():
         u.key= 0
    
    Q = MaxHeap(list(G.nodes.values()), a)
    s = empty
    t = empty
    V_minus_t = []
    ST_cut = [] 

    while Q.arrayHeap.heapSize != 0:

        u = Q.extractMax()

        if Q.arrayHeap.heapSize > 0:
            V_minus_t.append(u)

        s = t
        t = u
        for v in u.adjacencyList:
            if v[0].isPresent and v[0].notMerged:
                v[0].key += v[1]
                Q.shiftUp(v[0].index)  
    
    ST_cut.append(V_minus_t)
    ST_cut.append(t)
    #G_contract_st = contract(G, s, t)

    return ST_cut, s, t

def contract(G: Graph, s: Node, t: Node):
    #prepare to contract nodes s and t to the node with the minimum tag 
    if (s.tag > t.tag):
        s.notMerged = False
        tag_to_pop = s.tag
        tag_to_keep = t.tag
        t.adjacencyList += s.adjacencyList
    else: 
        t.notMerged = False
        tag_to_pop = t.tag
        tag_to_keep = s.tag
        s.adjacencyList += t.adjacencyList

    #traverse through each node and check if it connects to both s and t
    #if so then calculate the new adjacent cost to the merged s,t node
    for u in G.nodes.values():
        flag_update = 0
        new_adjCost = 0
        for v in u.adjacencyList:
            if v[0] == s or v[0] == t:
                flag_update += 1
                new_adjCost += v[1]
        #print(flag_update, new_adjCost)

        if flag_update == 2:
            for v in u.adjacencyList:
                if v[0] == s and v[0].notMerged:
                        v[1] = new_adjCost
                elif v[0] == t and v[0].notMerged: 
                        v[1] = new_adjCost

    for u in G.nodes.get(tag_to_keep).adjacencyList:
        print("test")

    #G.nodes.pop(tag_to_pop)

    return G

def GlobalMinCut (G :Graph):
    if len(list(G.nodes.values())) == 2:
        return G
    else:
        C1, s, t = stMinCut(G)
        C2 = stMinCut(contract(G, s, t))


new = Graph()
new.buildGraph(open("r_dataset/test.txt", "r"))

ST_cut, s, t = stMinCut(new)

new_G = contract(new, s, t)

for u in new_G.nodes.values():
    print(u, u.tag, u.adjacencyList,"\n")

new_ST, new_s, new_t = stMinCut(new_G)

new_G2 = contract(new_G, new_s, new_t)

for u in new_G2.nodes.values():
    print(u, u.tag, u.notMerged, u.adjacencyList,"\n")







