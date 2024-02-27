# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 08:44:03 2023

@author: user
"""


#k-means clustering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#lets us try to understand first how k means works fro two
#dimensional data
#for that, generate random numbers in the range 0 to 1
#and with uniform probability of 1/50
X=np.random.uniform(0,1,50)
Y=np.random.uniform(0,1,50)
#create a empty dataframe with 0 rows and 2 column
df_xy=pd.DataFrame(columns=["X","Y"])
#assign the value of X and Y to these columns
df_xy.X=X
df_xy.Y=Y
df_xy.plot(x="X",y="Y",kind="scatter")
model1=KMeans(n_clusters=3).fit(df_xy)

"""with data X and Y,apply kmeans model,
generate scatter plot with scale/font=10

cmap=plt.coolwarm:cool color combination"""

model1.labels_
df_xy.plot(x="X",y="Y",c=model1.labels_,kind="scatter",
           s=10,cmap=plt.cm.coolwarm)

Univ1=pd.read_excel("University_Clustering.xlsx")
Univ1.describe()
Univ=Univ1.drop(["State"],axis=1)
#we know that there is scale difference among the columns,which we have
#we have either by using normalization or standaratizzation

def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
#Now apply this normalization function to Univ dataframe fro all the rowa

df_norm=norm_func(Univ1.iloc[:,1:])

'''what will be ideal cluster number will it be 1,2,or 3 '''

TWSS=[]
k=list(range(2,8))
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    
TWSS.append(kmeans.inertia_)#total within the sum of sqaure
"""kmeans inertia also known as sum of sqaures errors (SSE)
calculate the sum of the distances of all points within a cluster
from the centroid the oint.it is the difference between the obtain
value and the predicted value"""

TWSS
#as k value increases the TWSS value decreases
plt.plot(k,TWSS,'ro-')
plt.xlabel("No_of_clusters")
plt.ylabel("Total_within_SS")


"""How to select the value of k from elbow curve
when k changes from  2 to 3 then  decrease in kwss iis higher from
k changes from 3 to 4 
when k values from 3  to 4
When k values changes from 5 to 6 decrease
is twss is considerably less,hence considered k=3"""

model=KMeans(n_clusters=3)
model.fit(df_norm)
model.labels_
mb=pd.Series(model.labels_)
Univ['clust']=mb
Univ.head()
Univ=Univ.iloc[:,[7,0,1,2,3,4,5,6]]
Univ
Univ.iloc[:,2:8].groupby(Univ.clust).mean()
Univ.to_csv("kmeans_University.csv",encoding="utf-8")


