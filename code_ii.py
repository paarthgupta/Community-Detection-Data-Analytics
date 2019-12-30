import numpy as np
import pandas as pd
import math
import math
import matplotlib.pyplot as plt 
import networkx as nx
from sklearn.cluster import KMeans as KMeans
from scipy import sparse
from numpy import linalg 
import community
import networkx as nx
import matplotlib.pyplot as plt

G = nx.read_gml('dolphins.gml')
A = nx.adjacency_matrix(G)
B=A.todense()
l=nx.normalized_laplacian_matrix(G)
lap=l.todense()
G1 = nx.from_scipy_sparse_matrix(l)
A1 = nx.adjacency_matrix(G1)
partition = community.best_partition(G)
size = float(len(set(partition.values())))
pos = nx.spring_layout(G)
count = 0
# cluster_0=[]
# cluster_1=[]
# cluster_2=[]
# cluster_3=[]
# cluster_4=[]
# for i in G.nodes():
#     if (partition[i] == 0):
#       cluster_0.append(i)
#     elif (partition[i] == 1):
#       cluster_1.append(i)
#     elif (partition[i] == 2):
#       cluster_2.append(i)
#     elif (partition[i] == 3):
#       cluster_3.append(i)
#     elif (partition[i] == 4):
#       cluster_4.append(i)   
#print(len(cluster_0))
# print(len(cluster_1))
# print(len(cluster_2))
# print(len(cluster_3))
# print(len(cluster_4))
num_part = max(partition.values()) + 1
list_1 = []
for i in range(num_part):
  temp_list=[]
  for j in G.nodes():
    if (partition[j] == i):        
      temp_list.append(j)
  list_1.append(temp_list)
# we will pick two largest communities
size=[]
for i in list_1:
  size.append(len(i))
#print(size)
valid_candidates = np.argsort(size)[-2:]

b=np.argsort(size)
c=b[:-2]
def Reverse(lst): 
    lst.reverse() 
    return lst 
      
# lst = [10, 11, 12, 13, 14, 15] 
c=Reverse(list(c))
b=Reverse(list(b))
# c

# merging greedily 

# d=c.copy()
# done=[]
# cou=1
# for i in c:
#   first_community_number=valid_candidates[0]
#   second_community_number=valid_candidates[1]
#   loss_merge_first=0
#   loss_merge_second=0
#   next_largest_partition=i
#   temp1_merged=list_1[first_community_number]
#   for j in range(len(list_1[next_largest_partition])):
#     zz=list_1[next_largest_partition]
#     temp1_merged.append(zz[j])
#   total={}
#   for jj in d[cou:]:
#     par = list_1[jj]
#     for k in range(len(par)):
#       tem = list(par[k])
#       tem[1] = cluster_number1
#       par[k] = tuple(tem)
#     print(par)
#     print(type(par))
#     total=total+par
#   cou=cou+1
    
    
partition = community.best_partition(G)
num_com = max(partition.values()) + 1
list_for_tracking_done_indices=Reverse(list(c))
list_for_updatibg=Reverse(list(b))
s = []
for i in range(num_com):
    partition_list_i=[]
    for j in partition.items():
      if (j[1] == i):
        partition_list_i.append(j)
    s.append(partition_list_i)
    
# s is the list of list containing all the nodes of the graph corresponding to its key values

size_dist = []
for i in s:
    size_dist.append(len(partition))
    
valid_candidates = np.argsort(size_dist)[-2:]
for i in range(num_com):
#extracting clusters numbers
    cluster_1 = s[valid_candidates[0]][0][1]
    cluster_2 = s[valid_candidates[1]][0][1]
    
# we will merge the group of nodes with both the valid candidates and check the modularity and finally merge with the one in which loss is minimized    
    
    if i not in valid_candidates:
        temp_merged = s[i]
        for j in range(len(temp_merged)):
            pp = list(temp_merged[j])
            pp[1] = cluster_1
            temp_merged[j] = tuple(pp)
        parti = temp_merged + s[cluster_1]
        tp = parti + s[cluster_2]
        for k in range(num_com):
            if k != i and k not in valid_candidates:
                tp += s[k]
        loss1 = community.modularity(dict(tp),G)
        temp_mer = s[i]
        
# same procedure for the second cluster        
        
        for j in range(len(temp_mer)):
            pp = list(temp_mer[j])
            pp[1] = cluster_2
            temp_mer[j] = tuple(pp)
        mp = temp_mer + s[cluster_2]
        tt = mp + s[cluster_1]
        for k in range(num_com):
            if k != i and k not in valid_candidates:
                tt += s[k]
        loss2 = community.modularity(dict(tt),G)
        if loss1 > loss2:
            s[i] = temp_merged
        else:
            s[i] = temp_mer
zz = []
for i in range(len(s)):
    partition_i = s[i]
    for j in range(len(partition_i)):
        zz.append(partition_i[j])
pos = list(set(dict(zz).values()))[0]
neg = list(set(dict(zz).values()))[1]
cluster_0 = []
cluster_1 = []
for i in range(len(zz)):
    ii = zz[i][1]
    if ii == pos:
        cluster_0.append(zz[i][0])
    else:
        cluster_1.append(zz[i][0])
        
        
def draw_graph(G):
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos,cluster_0, node_color='r')
    nx.draw_networkx_nodes(G, pos,cluster_1, node_color='b')
    nx.draw_networkx_labels(G, pos,font_size=10)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
draw_graph(G)
#plt.savefig('/content/gdrive/My Drive/Colab Notebooks/data/community_fiedler.jpg')
plt.show()











