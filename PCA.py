# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 16:26:16 2023

@author: user
"""
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
uni1 = pd.read_excel("University_CLustering.xlsx")
uni1.describe()
uni1.info()
uni = uni1.drop(["State"], axis=1)

# Considering only numerical data
uni.data = uni.iloc[:, 1:]

# Normalizing the numerical data
uni_normal = scale(uni.data)
uni_normal

pca = PCA(n_components=6)
pca_values = pca.fit_transform(uni_normal)

# The amount of variance that each PCA explains is
var = pca.explained_variance_ratio_
var

# PCA weights
# pca.components_
# pca.components_[0]


# Cimulative variance
var1 = np.cumsum(np.round(var, decimals=4)*100)
var1
# Variance plot for PCA components obtained
plt.plot(var1, color="red")

# PCA Scores
pca_values
pca_data = pd.DataFrame(pca_values)
pca_data.columns="comp0","comp1","comp3","comp4","comp5"
final=pd.concat([uni.Univ,pca_data.iloc[:, 0:3]],axis=1)
#this is Univ column of uni data frame
#scatter diagram
import matplotlib.pylab as plt
ax=final.plot(x='comp0',y="comp1",kind="scatter",figsize=(12,8))
final[["comp0","comp1","Univ"]].apply(lambda x:ax.text(x),axis=1)