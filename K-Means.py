## k-means
from pyspark.ml.clustering import KMeans

rescaledData = tfidf.selectExpr("idf as features")

kmeans = KMeans().setK(11).setSeed(1)
model = kmeans.fit(rescaledData)

predictions = model.transform(rescaledData)

wssse = model.computeCost(rescaledData)
"Within Set Sum of Squared Errors = " + str(wssse)

centers = model.clusterCenters()
"Cluster Centers: "
for center in centers:
    center