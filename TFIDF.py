from pyspark.ml.feature import HashingTF, IDF, Tokenizer

## fertig == DF von Vorverarbeitung, "value" mit Arrays

hashingTF = HashingTF(inputCol="value", outputCol="rawFeatures", numFeatures=40)
featurizedData = hashingTF.transform(fertig)
# alternatively, CountVectorizer can also be used to get term frequency vectors

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

rescaledData.select("value", "features").show()

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
