import numpy as np
import pandas as pd
import math
import math
import matplotlib.pyplot as plt 
import networkx as nx
from sklearn.cluster import KMeans as KMeans
from scipy import sparse
from numpy import linalg 

# picture has been attached in the mail 

df1 = pd.read_csv("./../data/11_twoCirclesData.csv")      # reading the data
total_data= df1.values       # taking out imp columns
# x=total_data['x'].values  
# y=total_data['y'].values  


m=len(total_data)
A=np.zeros((m,m))
var=0.0004900000000000002
cluster_0=[]
cluster_1=[]
    
for i in range(m):
  for j in range (m):
    a=np.square(np.linalg.norm(total_data[i]-total_data[j]))
    A[i][j] = np.exp(- a/(var))    
G=nx.from_numpy_matrix(A)
node_list=list(G.nodes())
l=nx.normalized_laplacian_matrix(G)
eigenValues, eigenVectors = linalg.eigh(l.todense())
temp=eigenVectors.T
idx = np.argsort(eigenValues) 
a=temp[idx[1]]
b=temp[idx[2]]
#c=temp[idx[3]]
d=np.vstack((a,b))
#e=np.vstack((d,c))
import matplotlib.pyplot as plt

kmeans = KMeans(n_clusters=2, random_state=0).fit(d.T)
cluster_0=[]
cluster_1=[]
for i in range(len(kmeans.labels_)):
  if (kmeans.labels_[i] == 0):
    cluster_0.append(node_list[i])
    x, y = total_data[i]
    plt.scatter(x, y,c='red')
  elif (kmeans.labels_[i] == 1):
    cluster_1.append(node_list[i])
    x, y = total_data[i]
    plt.scatter(x, y,c='blue')
#var=var+0.00003
plt.show()
#plt.savefig('/content/gdrive/My Drive/Colab Notebooks/data/community_fiedler.jpg')













