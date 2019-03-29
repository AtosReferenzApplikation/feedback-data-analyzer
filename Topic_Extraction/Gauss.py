from pyspark.ml.feature import RegexTokenizer
from pyspark.ml.feature import StopWordsRemover
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer

from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import IDF
from pyspark.ml.clustering import GaussianMixture

from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local").appName("Gauss").getOrCreate()

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


for k in filrow:
	for m in k:
		for n in m:	
			n = lemmatizer.lemmatize(n, "v")
			n = lemmatizer.lemmatize(n)
			n = stemmer.stem(n)
			words.append(n)
		all.append(words)
		words = []

fertig = spark.createDataFrame(all, ["value"])

additionalstopwords = ["doesn", "didn", "isn", "wasn", "get", "time", "know", "feel", "instead", "realli", "re", \
        "shouldn", "tho", "even", "well", "film","movi", "like", \
        "see", "everi", "great", "br", "good", "go", "think", "stori", "one", "make", "bad", "watch"]

remover2 = StopWordsRemover(inputCol="value", outputCol="rem")
rem2 = remover2.transform(fertig)
remover3 = StopWordsRemover(inputCol="rem", outputCol="addsw", stopWords = additionalstopwords)
rem3 = remover3.transform(rem2)
fertig = rem3.selectExpr("addsw as features")

cv = CountVectorizer(inputCol="features", outputCol="tf", vocabSize = 300, minDF = 20)
model = cv.fit(fertig)
dict = model.vocabulary
tf = model.transform(fertig)
idf = IDF(inputCol="tf", outputCol="idf")
idfModel = idf.fit(tf)
tfidf = idfModel.transform(tf)

rescaledData = tfidf.selectExpr("idf as features")

gmm = GaussianMixture(k=4, tol=0.001, maxIter=1000, seed = 1489897845565)
model1 = gmm.fit(rescaledData)

dfgmm = model1.gaussiansDF
x = model1.transform(rescaledData)
x.show()
#Output = mean für Centroid (Koordinaten) und cov für Kovarianz