# 1 importing library
import inline as inline
import matplotlib
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

# 2 import data set in csv file
df = pd.read_csv('C:\weather\cluster data set.csv')

# 3 using 3 clusters
km = KMeans(n_clusters=3)
# print(km)

# 4 predicting Age and Income
y_predicted = km.fit_predict(df[['Age', 'Income']])
# print(y_predicted)
df['cluster'] = y_predicted
# print(df.head())

# 5 dividing cluster in 3 sets
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]

# 6 plotting data in scatter graph and using colour for distinguish image
plt.scatter(df1.Age, df1['Income'], color='green')
plt.scatter(df2.Age, df2['Income'], color='red')
plt.scatter(df3.Age, df3['Income'], color='black')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='purple', marker='*', label='centroid')

# 7 plotting graph
x = plt.xlabel('Age')
y = plt.ylabel('Income')
plt.show()

# 8 checking new clean data
k_rng = range(1, 10)
sse = []
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['Age', 'Income']])
    sse.append(km.inertia_)
    print(sse)

# 9 checking elbow for nuber of clusters
plt.xlabel('k')
plt.ylabel('sum of squared error')
#plt.plot(k_rng, sse)

