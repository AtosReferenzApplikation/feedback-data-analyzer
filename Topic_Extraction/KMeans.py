from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local").appName("KMeans").getOrCreate()

from pyspark.ml.feature import RegexTokenizer
from pyspark.ml.feature import StopWordsRemover
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer

from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import IDF
from pyspark.ml.clustering import KMeans

from pyspark.sql import Row

df = spark.read.text(r"C:\Users\A704194\projects\feedback-data-analyzer\TR\Projekt\train\neg")
regexTokenizer = RegexTokenizer(inputCol="value", outputCol="words", pattern="\\W")
regtok = regexTokenizer.transform(df)
remover1 = StopWordsRemover(inputCol="words", outputCol="filtered")
rem1 = remover1.transform(regtok)
fil = rem1.select("filtered")
filrow = fil.take(fil.count())

lemmatizer = WordNetLemmatizer()
words = []
stemmer = SnowballStemmer("english")
all = []
rowsneu = []

for k in filrow:
	for m in k:
		for n in m:	
			n = lemmatizer.lemmatize(n, "v")
			n = lemmatizer.lemmatize(n)
			n = stemmer.stem(n)
			words.append(n)
		all.append(words)
		words = []

for x in all:
    rowsneu.append(Row(value=x))

fertig = spark.createDataFrame(rowsneu)

remover2 = StopWordsRemover(inputCol="value", outputCol="rem")
rem2 = remover2.transform(fertig)

additionalstopwords = ["doesn", "didn", "isn", "wasn", "get", "time", "know", "feel", "instead", "realli", "re", \
        "shouldn", "tho", "even", "well", "film","movi", "like", \
        "see", "everi", "great", "br", "good", "go", "think", "stori", "one", "make", "bad", "watch"]
remover3 = StopWordsRemover(inputCol="rem", outputCol="addsw", stopWords = additionalstopwords)
rem3 = remover3.transform(rem2)
fertig = rem3.selectExpr("addsw as features")

#Vektor aus Dokumenten, tfidf-df erstellen
cv = CountVectorizer(inputCol="features", outputCol="tf", vocabSize = 300, minDF = 2)
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
centers = model.clusterCenters()

#Wie ähnlich Objekt dem eigentlichen Cluster ist, verglichen mit anderen
#Gemessen an euklidischer oder Manhatten-Distanz (metrische)
#Range: −1 to +1 (high = well matched, low = poorly matched)
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)

#Within set sum of squared errors
wssse = model.computeCost(rescaledData)

#dict mit Cluster-Größe erstellen
dictpred = {}
x = predictions.select("prediction").take(predictions.count())
for i in x:
    for j in i:
        dictpred[j] = 0

for i in x:
    for j in i:
        dictpred[j] += 1


#Erstellen von Diagrammen zur Verteilung der Dokumente in Cluster, braucht Dict
import pandas as pd
import matplotlib.pyplot as plt

x = pd.Series(dictpred)
y = pd.Series.sort_values(x)
z = pd.DataFrame(y)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,9))

#specify which column of the dataframe to plot (here 0)
z.plot(y=0, kind = 'pie', ax = axes[0])
z.plot(kind = 'bar', ax = axes[1])
#make aspect equal (such that circle is not eliptic)
axes[0].set_aspect("equal")
#place the legend at a decent position
axes[0].legend(loc=1, bbox_to_anchor= (0,1.1), fontsize=8)
plt.show()