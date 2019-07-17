# -*- coding: utf-8 -*-
"""
Created on Wed Apr 03 11:33:04 2019

@author: Zaki
"""

import csv
import numpy as np
import pandas as pd
#from pyAudioAnalysis import audioBasicIO #A
#from pyAudioAnalysis import audioFeatureExtraction #B
#import librosa.display
import matplotlib.pyplot as plt
from copy import deepcopy
#import librosa
#import os
import math

data = pd.read_csv('E:\\proximotex\\results.csv',index_col=False,header=None)
data=data.T
name=data[0]
name1=name[0].split('-',-1).pop(0)
name2=name[0].split('-',-1).pop(1)
name=str(name1)+'-'+str(name2)
f1=data[1].values.astype('float32')
f2=data[2].values.astype('float32')
f3=data[3].values.astype('float32')
f4=data[4].values.astype('float32')
f7=data[5].values.astype(str)
#f5=data[4].values
f8=[]
for i in range(0,len(f7)):
    f8.append(float(f7[i].split('.',-1).pop(1)))
f8=np.asarray(f8)
f5=f8.astype('float32')
f6=data[6].values.astype('float32')
plt.scatter(f1,f5)
plt.show()
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans
X=np.array([f1,f5])
X=X.transpose()
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print(centroids)
print(labels)       
colors = ["g.","r.","c.","y."]

for i in range(len(X)):
    print("coordinate:",X[i], "label:", labels[i])
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)


plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)

plt.show()
pdd=data[[0]]
labels=pd.DataFrame(labels)
report=pd.concat([pdd,labels],axis=1)
filename=str(name)+'.csv'
report.to_csv(filename,encoding='utf-8',index=None)
plt.savefig(plt.savefig(str(name) +".png", format="PNG"))
       
     
