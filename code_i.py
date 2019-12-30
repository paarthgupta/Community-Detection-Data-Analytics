import numpy as np
import pandas as pd
import math
import math
import matplotlib.pyplot as plt 
import matplotlib.patches as pt
import networkx as nx
from sklearn.cluster import KMeans as KMeans
from numpy import linalg 



G = nx.read_gml('./../data/dolphins.gml')
A = nx.adjacency_matrix(G)
B=A.todense()
l=nx.normalized_laplacian_matrix(G)
lap=l.todense()
G1 = nx.from_scipy_sparse_matrix(l)
A1 = nx.adjacency_matrix(G1)



eigenValues, eigenVectors = linalg.eigh(l.todense())
temp=eigenVectors.T
idx = np.argsort(eigenValues) 
a=temp[idx[1]]

print(" Fiedler vector of normalized_laplacian_matrix :")
print(a)
#b=temp[idx[2]]
#c=temp[idx[3]]
#d=np.vstack((a,b))
#e=np.vstack((d,c))
import matplotlib.pyplot as plt

kmeans = KMeans(n_clusters=2, random_state=0).fit(a.T)
cluster_0=[]
cluster_1=[]
#kmeans = KMeans(n_clusters=2)
kmeans.fit(a.reshape((len(G.nodes()),1)))
# print(kmeans.cluster_centers_)
# print(kmeans.labels_)
node_list=list(G.nodes())

for i in range(len(kmeans.labels_)):
  if (kmeans.labels_[i] == 0):
    cluster_0.append(node_list[i])
  elif (kmeans.labels_[i] == 1):
    cluster_1.append(node_list[i])
# ref: https://towardsdatascience.com/unsupervised-machine-learning-spectral-clustering-algorithm-implemented-from-scratch-in-python-205c87271045  
def draw_graph(G):
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos,cluster_0, node_color='r')
    nx.draw_networkx_nodes(G, pos,cluster_1, node_color='b')

    nx.draw_networkx_labels(G, pos,font_size=10)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
draw_graph(G)
#plt.savefig('/content/gdrive/My Drive/Colab Notebooks/data/community_fiedler.jpg')
plt.show()

















