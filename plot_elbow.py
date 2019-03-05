# rescaledData (features: vector)

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyspark import SparkContext
from pyspark.ml.clustering import KMeans

cost = np.zeros(50)
for k in range(2,50):
    kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
    model = kmeans.fit(rescaledData.sample(False,0.1, seed=42))
    cost[k] = model.computeCost(rescaledData)

fig, ax = plt.subplots(1,1, figsize =(8,6))
ax.plot(range(2,50),cost[2:50])
ax.set_xlabel('k')
ax.set_ylabel('cost')

#Anzeigen von figure mit fig.show()