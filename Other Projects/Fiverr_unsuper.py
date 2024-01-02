

from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, normalize
import scipy.cluster.hierarchy as shc
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np


from preprocessing import * 


Y = final_data['FTR']
X = final_data.drop(['FTR','FTHG', 'FTAG'], axis=1)




df_num = final_data.select_dtypes(include = ['float64', 'int64'])
df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)




# In[83]:

X = X.dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
  
# Normalizing the data so that the data approximately 
# follows a Gaussian distribution
X_normalized = normalize(X_scaled)
  
# Converting the numpy array into a pandas DataFrame
X_normalized = pd.DataFrame(X_normalized)


# In[84]:


pca = PCA(n_components = 2)
X_principal = pca.fit_transform(X_normalized)
X_principal = pd.DataFrame(X_principal)
X_principal.columns = ['P1', 'P2']


# In[90]:
## To determine the optimal number of clusters by visualizing the data, imagine all the horizontal lines as being completely horizontal and then after calculating the maximum distance between any two horizontal lines, draw a horizontal line in the maximum distance calculated.
## Dendograms are used to divide a given cluster into many different clusters.
## From here we can see ideal number of clusters should be 3 (blue color)

plt.figure(figsize =(8, 8))
plt.title('Visualising the data')
Dendrogram = shc.dendrogram((shc.linkage(X_principal, method ='ward')))


# In[90]:

ac2 = AgglomerativeClustering(n_clusters = 2)
  
# Visualizing the clustering
plt.figure(figsize =(6, 6))
plt.scatter(X_principal['P1'], X_principal['P2'], 
           c = ac2.fit_predict(X_principal), cmap ='rainbow')
plt.show()


# In[ ]:

ac3 = AgglomerativeClustering(n_clusters = 3)
plt.figure(figsize =(6, 6))
plt.scatter(X_principal['P1'], X_principal['P2'], 
           c = ac3.fit_predict(X_principal), cmap ='rainbow')
plt.show()


# In[ ]:

ac4 = AgglomerativeClustering(n_clusters = 4)
plt.figure(figsize =(6, 6))
plt.scatter(X_principal['P1'], X_principal['P2'], 
           c = ac4.fit_predict(X_principal), cmap ='rainbow')
plt.show()


# In[ ]:

ac5 = AgglomerativeClustering(n_clusters = 5)
plt.figure(figsize =(6, 6))
plt.scatter(X_principal['P1'], X_principal['P2'], 
           c = ac5.fit_predict(X_principal), cmap ='rainbow')
plt.show()


# In[ ]:

ac6 = AgglomerativeClustering(n_clusters = 6)
plt.figure(figsize =(6, 6))
plt.scatter(X_principal['P1'], X_principal['P2'], 
           c = ac6.fit_predict(X_principal), cmap ='rainbow')
plt.show()


# In[94]:


k = [2, 3, 4, 5, 6, 7]

ac2 = AgglomerativeClustering(n_clusters = 2)
ac3 = AgglomerativeClustering(n_clusters = 3)
ac4 = AgglomerativeClustering(n_clusters = 4)
ac5 = AgglomerativeClustering(n_clusters = 5)
ac6 = AgglomerativeClustering(n_clusters = 6)
ac7 = AgglomerativeClustering(n_clusters = 7)

# Appending the silhouette scores of the different models to the list
silhouette_scores = []
silhouette_scores.append(
        silhouette_score(X_principal, ac2.fit_predict(X_principal)))
silhouette_scores.append(
        silhouette_score(X_principal, ac3.fit_predict(X_principal)))
silhouette_scores.append(
        silhouette_score(X_principal, ac4.fit_predict(X_principal)))
silhouette_scores.append(
        silhouette_score(X_principal, ac5.fit_predict(X_principal)))
silhouette_scores.append(
        silhouette_score(X_principal, ac6.fit_predict(X_principal)))
silhouette_scores.append(
        silhouette_score(X_principal, ac7.fit_predict(X_principal)))
  
# Plotting a bar graph to compare the results
plt.bar(k, silhouette_scores)
plt.xlabel('Number of clusters', fontsize = 20)
plt.ylabel('S(i)', fontsize = 20)
plt.show()


# In[80]:


def make_generator(parameters):
    if not parameters:
        yield dict()
    else:
        key_to_iterate = list(parameters.keys())[0]
        next_round_parameters = {p : parameters[p]
                    for p in parameters if p != key_to_iterate}
        for val in parameters[key_to_iterate]:
            for pars in make_generator(next_round_parameters):
                temp_res = pars
                temp_res[key_to_iterate] = val
                yield temp_res

# add fix parameters - here - it's just a random one
fixed_params = {"max_iter":300 } 

param_grid = {"n_clusters": range(2, 33,3)}


distortions = []
inertias = []
silhouette = []
mapping1 = {}
mapping2 = {}

for params in make_generator(param_grid):
    params.update(fixed_params)
    print(params)
    ca = KMeans( **params )
    ca.fit(X)
    labels = ca.labels_
    distortions.append(sum(np.min(cdist(X, ca.cluster_centers_,
                                        'euclidean'), axis=1)) / X.shape[0])
    inertias.append(ca.inertia_)
    silhouette.append(silhouette_score(X, labels))
 
    mapping1[params['n_clusters']] = sum(np.min(cdist(X, ca.cluster_centers_,
                                   'euclidean'), axis=1)) / X.shape[0]
    mapping2[params['n_clusters']] = ca.inertia_


# In[14]:


plt.plot(range(2,33,3), distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()


# In[81]:


plt.plot(range(2,33,3), silhouette, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Silhouette')
plt.title('The Elbow Method using Silhouette')
plt.show()


# In[15]:


plt.plot(range(2,33,3), inertias, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Inertia')
plt.title('The Elbow Method using Inertia')
plt.show()


# In[16]:


ca = KMeans( n_clusters=17, max_iter=200 )
X = X.dropna()
ca.fit(X)
labels = ca.labels_
X['labels'] = labels

X.labels.value_counts()


# In[102]:


X.labels.hist(bins=17)


# In[153]:


## Returns the features with most difference
## Comparing cluster 1 and 10

def compare_cluster(label_x, label_y):
    left = X[X['labels']==label_x].mean().reset_index()
    right = X[X['labels']==label_y].mean().reset_index()
    both = left.merge(right, on='index',how='inner').reset_index()
    both.columns = [''.join(tup).rstrip('_') for tup in both.columns.values]
    both = both.iloc[:,1:]
    both = both.rename(columns={'0_x':'cluster_'+ str(label_x), '0_y':'cluster_' + str(label_y)})
    both['difference'] = abs(both['cluster_'+ str(label_x)] - both['cluster_'+ str(label_y)])
    return both[both['difference']>=1]

compare_cluster(1, 10)


# In[154]:


compare_cluster(12, 0)


# In[155]:


compare_cluster(7, 8)
## Similarly can do for remaining clusters, giving you more freedom to understand the data


# In[22]:


plt.figure(figsize=(10,6))
plt.title("Fouls Frequency")
sns.axes_style("dark")
sns.violinplot(y=X["AF"])
sns.violinplot(y=X["HF"])
plt.show()


# In[103]:


plt.figure(figsize=(10,6))
plt.title("Fouls Frequency")
sns.axes_style("dark")
sns.violinplot(y=X[X['labels']==1]["AF"])
sns.violinplot(y=X[X['labels']==10]["HF"])
plt.show()


# In[34]:
## Here you can visualize top columns difference between clusters.
## Current code is just for comparing cluster 1 and 10 on fouls
## There's no limit to visulaizations hence requeest you to explore as you feel

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111)
ax.scatter( X["HS"][X.labels == 10], X["AS"][X.labels == 10], c='blue', s=60)
ax.scatter( X["HS"][X.labels == 1], X["AS"][X.labels == 1], c='red', s=60)
# ax.view_init(30, 185)
plt.ylabel("AS")
plt.xlabel("HS")
# ax.set_zlabel('AF')
plt.show()

