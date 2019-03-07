from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import IDF
from pyspark.ml.clustering import KMeans

# rem2 von komplett übernehmen

additionalstopwords = ["doesn", "didn", "isn", "wasn", "get", "time", "know", "feel", "instead", "realli", "re", \
        "shouldn", "tho", "even", "well", "film","movi", "like", \
        "see", "everi", "great", "br", "good", "go", "think", "stori", "one", "make", "bad", "watch"]
remover3 = StopWordsRemover(inputCol="rem", outputCol="addsw", stopWords = additionalstopwords)
rem3 = remover3.transform(rem2)
fertig = rem3.selectExpr("addsw as value")

#Vektor aus Dokumenten, tfidf-df erstellen
cv = CountVectorizer(inputCol="value", outputCol="tf", vocabSize = 300, minDF = 800)
model = cv.fit(fertig)
dict = model.vocabulary
tf = model.transform(fertig)
idf = IDF(inputCol="tf", outputCol="idf")
idfModel = idf.fit(tf)
tfidf = idfModel.transform(tf)

rescaledData = tfidf.selectExpr("idf as features")

#predictions erstellen
kmeans = KMeans(featuresCol="features", predictionCol="prediction", k=67, initSteps=70)
model = kmeans.fit(rescaledData)
predictions = model.transform(rescaledData)

#dict mit Cluster-Größe erstellen
dictpred = {}
x = predictions.select("prediction").take(predictions.count())
for i in x:
    for j in i:
        dictpred[j] = 0

for i in x:
    for j in i:
        dictpred[j] += 1

#Centroide anzeigen, centers[0] = Centroid von 0. Cluster
centers = model.clusterCenters()
"Cluster Centers: "
for center in centers:
    center

#Sum of squared errors
wssse = model.computeCost(rescaledData)
"Within Set Sum of Squared Errors = " + str(wssse)
