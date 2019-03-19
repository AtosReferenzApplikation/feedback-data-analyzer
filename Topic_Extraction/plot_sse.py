# Plottet square sum of errors, also wie hoch bei welcher K-Anzahl
# Sse auch für Anzahl von K benutzt
# rescaledData (features: vector)

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyspark import SparkContext
from pyspark.ml.clustering import KMeans

sse = np.zeros(70)
for k in range(10,70):
    kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
    model = kmeans.fit(rescaledData.sample(False,0.1, seed=42))
    sse[k] = model.computeCost(rescaledData)

fig, ax = plt.subplots(1,1, figsize =(8,6))
ax.plot(range(10,70),sse[10:70])
ax.set_xlabel('k')
ax.set_ylabel('cost')

#Anzeigen von figure mit fig.show()