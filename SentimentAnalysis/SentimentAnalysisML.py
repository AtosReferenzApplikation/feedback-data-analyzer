from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local").appName("SentimentAnalysis").getOrCreate()

import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)

from pyspark.sql.functions import lit
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.common import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import IDF
from pyspark.ml.clustering import LDA
from pyspark.ml.classification import LogisticRegression

# Txt Dateien einlesen
#negdf=spark.read.text(r"C:\Users\A704081\Desktop\Daten")
#posdf=spark.read.text(r"C:\Users\A704081\Desktop\PosDaten")
negdf=spark.read.text(r"C:\Users\A704081\Downloads\Projekt\aclImdb_v1\aclImdb\train\neg")
posdf=spark.read.text(r"C:\Users\A704081\Downloads\Projekt\aclImdb_v1\aclImdb\train\pos")

# Label Spalte einfügen
# 0=negative, 1=positive
negdf=negdf.withColumn("label", lit(0))
posdf=posdf.withColumn("label", lit(1))

# DataFrames zusammenfügen
neg=negdf.take(negdf.count())
pos=posdf.take(posdf.count())
# Listen erstellen:
neglist=[[neg[x][0],neg[x][1],] for x in range(negdf.count())]
poslist=[[pos[x][0],pos[x][1],] for x in range(posdf.count())]
# Listen vereinen
mergedlist=neglist + poslist

# Aus vereinter Liste DataFrame erstellen
data=spark.createDataFrame(mergedlist,["value","label"])

# Text in SparkNLP einlesen:
documentAssembler = DocumentAssembler() \
	.setInputCol("value") \
	.setOutputCol("document")
data2=documentAssembler.transform(data)

# Text in SparkNLP's Token umwandeln:
tokenizer = Tokenizer() \
	.setInputCols(["document"]) \
	.setOutputCol("token").addInfixPattern("(\p{L}+)(n't\b)")
data3=tokenizer.transform(data2)

# Satzzeichen entfernen:
normalizer = Normalizer() \
	.setInputCols(["token"]) \
	.setOutputCol("normalized")
norm=normalizer.fit(data3)

data4=norm.transform(data3)

# Lemmatizen:
lemmatizer = Lemmatizer() \
	.setInputCols(["normalized"]) \
	.setOutputCol("lemma") \
	.setDictionary(r"C:\Users\A704081\Downloads\lemmatization-lists-master\lemmatization-en.txt", key_delimiter="\t", value_delimiter="\n")
lem=lemmatizer.fit(data4)
data5=lem.transform(data4)

# Ich verwende keinen Stemmer, da die Sentiment Listen keine gestemmten Wörter enthalten
# Aus SparkNLP auslesen:
finisher = Finisher() \
	.setInputCols(["lemma"]) \
	.setIncludeMetadata(False)\
	.setOutputAsArray(True)
data6=finisher.transform(data5)

# Stopwords entfernen:
remover = StopWordsRemover(inputCol="finished_lemma", outputCol="filtered")
data7=remover.transform(data6)

# Feature Vectoren erstellen:
cv = CountVectorizer(inputCol="filtered", outputCol="features")
model = cv.fit(data7)
vocablist=model.vocabulary
data8 = model.transform(data7)

# IDF auf Term-Frequency-Vectoren:
idf = IDF(inputCol="features", outputCol="IDFFeatures")
idfModel = idf.fit(data8)
data9 = idfModel.transform(data8)

# Logistic Regression
lr=LogisticRegression()
lrmodel=lr.fit(data9)
data10=lrmodel.transform(data9)
data10.show()

norm.write().overwrite().save(r"C:\Users\A704081\projects\feedback-data-analyzer\SentimentAnalysis\Models\Norm")
lem.write().overwrite().save(r"C:\Users\A704081\projects\feedback-data-analyzer\SentimentAnalysis\Models\Lemmatizer")
model.write().overwrite().save(r"C:\Users\A704081\projects\feedback-data-analyzer\SentimentAnalysis\Models\CV")
idfModel.write().overwrite().save(r"C:\Users\A704081\projects\feedback-data-analyzer\SentimentAnalysis\Models\IDF")
lrmodel.write().overwrite().save(r"C:\Users\A704081\projects\feedback-data-analyzer\SentimentAnalysis\Models\LogisticRegression")
spark.stop()