## k-means
from pyspark.ml.clustering import KMeans

kmeans = KMeans().setK(2).setSeed(1)
model = kmeans.fit(rescaledData)

wssse = model.computeCost(rescaledData)
"Within Set Sum of Squared Errors = " + str(wssse)

centers = model.clusterCenters()
"Cluster Centers: "
for center in centers:
    center