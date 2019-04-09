####################################### Def Model

from pyspark.sql.functions import lit
from pyspark.ml.feature import RegexTokenizer
from pyspark.ml.feature import StopWordsRemover
from nltk.stem import WordNetLemmatizer

from pyspark.ml.feature import CountVectorizer, CountVectorizerModel
from pyspark.ml.feature import IDF, IDFModel
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel

additionalstopwords = ["doesn", "didn", "isn", "wasn", "get", "realli", "re", "shouldn", "tho", "everi", "br"]

regexTokenizer = RegexTokenizer(inputCol="value", outputCol="words", pattern="\\W")
remover1 = StopWordsRemover(inputCol="words", outputCol="filtered")
remover2 = StopWordsRemover(inputCol="filtered", outputCol="rem")
remover3 = StopWordsRemover(inputCol="rem", outputCol="filtered", stopWords = additionalstopwords)

lemmatizer = WordNetLemmatizer()
words = []
all = []
hilf = []

cv = CountVectorizer(inputCol="filtered", outputCol="tf")
idf = IDF(inputCol="tf", outputCol="features")

lr = LogisticRegression()

####################################### Model erstellen

# Txt Dateien einlesen neg: \0_3.txt pos: \0_9.txt
negdf = spark.read.text(r"C:\Users\A704194\projects\feedback-data-analyzer\Daten\TR_Englisch\neg")
posdf = spark.read.text(r"C:\Users\A704194\projects\feedback-data-analyzer\Daten\TR_Englisch\pos")

# Label Spalte einfügen
# 0=negative, 1=positive
negdf = negdf.withColumn("label", lit(0))
posdf = posdf.withColumn("label", lit(1))

# DataFrames zusammenfügen
neg = negdf.take(negdf.count())
pos = posdf.take(posdf.count())
# Listen erstellen:
neglist = [[neg[x][0],neg[x][1],] for x in range(negdf.count())]
poslist = [[pos[x][0],pos[x][1],] for x in range(posdf.count())]
# Listen vereinen
mergedlist = neglist + poslist

# Aus vereinter Liste DataFrame erstellen
label = spark.createDataFrame(mergedlist,["value","label"])

regtok = regexTokenizer.transform(label)
rem1 = remover1.transform(regtok)
fil = rem1.select("filtered", "label")
filrow = fil.take(fil.count())

for k in filrow:
    for n in k[0]:
        n = lemmatizer.lemmatize(n, "v")
        n = lemmatizer.lemmatize(n)
        words.append(n)
    label = k[1]
    lab.append(words)
    lab.append(label)
    all.append(lab)
    lab = []
    words = []

fertig = spark.createDataFrame(all, ["filtered", "label"])

# Stopwords entfernen:
rem2 = remover2.transform(fertig).select("rem", "label")
fertig = remover3.transform(rem2).select("filtered", "label")

cvmodel = cv.fit(fertig)
dict = cvmodel.vocabulary
tf = cvmodel.transform(fertig)
idfmodel = idf.fit(tf)
tfidflab = idfmodel.transform(tf).select("features", "label")

lrmodel = lr.fit(tfidflab)

# Logistic Regression mit Pipeline
cvmodel.write().overwrite().save(r"C:\Users\A704194\projects\Spark_PP1\model\cv")
idfmodel.write().overwrite().save(r"C:\Users\A704194\projects\Spark_PP1\model\idf")
lrmodel.write().overwrite().save(r"C:\Users\A704194\projects\Spark_PP1\model\lr")

####################################### Def Anwendung

from pyspark.ml.feature import RegexTokenizer
from pyspark.ml.feature import StopWordsRemover
from nltk.stem import WordNetLemmatizer

from pyspark.ml.feature import CountVectorizer, CountVectorizerModel
from pyspark.ml.feature import IDF, IDFModel
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel

from pyspark.sql import Row

additionalstopwords = ["doesn", "didn", "isn", "wasn", "get", "realli", "re", "shouldn", "tho", "everi", "br"]

regexTokenizer = RegexTokenizer(inputCol="value", outputCol="words", pattern="\\W")
remover1 = StopWordsRemover(inputCol="words", outputCol="filtered")
remover2 = StopWordsRemover(inputCol="value", outputCol="rem")
remover3 = StopWordsRemover(inputCol="rem", outputCol="filtered", stopWords = additionalstopwords)

lemmatizer = WordNetLemmatizer()
words = []
all = []
final = []
hilf = []

lr = LogisticRegression()

idfmodel = IDFModel.load(r"C:\Users\A704194\projects\Spark_PP1\model\idf")
cvmodel = CountVectorizerModel.load(r"C:\Users\A704194\projects\Spark_PP1\model\cv")
lrmodel = LogisticRegressionModel.load(r"C:\Users\A704194\projects\Spark_PP1\model\lr")

Counternull = 0
Countereins = 0
ergebnis = 5

####################################### Anwendung

df = spark.read.text(r"C:\Users\A704194\projects\feedback-data-analyzer\TR\Projekt\test\pos\0_10.txt")

regtok = regexTokenizer.transform(df)
filrow = remover1.transform(regtok).select("filtered").take(regtok.count())

##nur möglich, wenn nur eine Zeile/ein Dokument:
for k in filrow:
	for n in k[0]:	
		n = lemmatizer.lemmatize(n, "v")
		n = lemmatizer.lemmatize(n)
		all.append(n)

dfneu = spark.createDataFrame([(all,)], ["value"])
rem2 = remover2.transform(dfneu)
fertig = remover3.transform(rem2).select("filtered")

#Vektor aus Dokumenten, tfidf-df erstellen
tf = cvmodel.transform(fertig)
tfidf = idfmodel.transform(tf)
end = lrmodel.transform(tfidf).select("prediction")

for x in end.take(end.count()):
	if int(x[0]) == 0:
		ergebnis = "neg"
	elif int(x[0]) == 1:
		ergebnis = "pos"
	print(ergebnis)

