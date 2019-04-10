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

documentAssembler = DocumentAssembler() \
	.setInputCol("value") \
	.setOutputCol("document")

tokenizer = Tokenizer() \
	.setInputCols(["document"]) \
	.setOutputCol("token") \
	.addInfixPattern("(\p{L}+)(n't\b)")

normalizer = Normalizer() \
	.setInputCols(["token"]) \
	.setOutputCol("normalized")

lemmatizer = Lemmatizer() \
	.setInputCols(["normalized"]) \
	.setOutputCol("lemma") \
	.setDictionary(r"C:\Users\A704081\Downloads\lemmatization-lists-master\lemmatization-en.txt", key_delimiter="\t", value_delimiter="\n")

finisher = Finisher() \
	.setInputCols(["lemma"]) \
	.setIncludeMetadata(False)\
	.setOutputAsArray(True)

NLPpipeline=Pipeline(stages=[documentAssembler, tokenizer, normalizer, lemmatizer, finisher])
NLPpipelineModel=NLPpipeline.fit(data)

data2=NLPpipelineModel.transform(data)

# Stopwords entfernen:
remover = StopWordsRemover(inputCol="finished_lemma", outputCol="filtered")
data3=remover.transform(data2)

# Feature Vectoren erstellen:
cv = CountVectorizer(inputCol="filtered", outputCol="features")
model = cv.fit(data3)
vocablist=model.vocabulary
data4 = model.transform(data3)

# IDF auf Term-Frequency-Vectoren:
idf = IDF(inputCol="features", outputCol="IDFFeatures")
idfModel = idf.fit(data4)
data5 = idfModel.transform(data4)

# Logistic Regression
lr=LogisticRegression()
lrmodel=lr.fit(data5)
data6=lrmodel.transform(data5)

SparkPipeline=Pipeline().setStages([remover,cv,idf,lr])
SparkPipelineModel=SparkPipeline.fit(data2)

data7=SparkPipelineModel.transform(data2)
data7.show()

NLPpipelineModel.write().overwrite().save(r"C:\Users\A704081\projects\feedback-data-analyzer\SentimentAnalysis\Models\NLPPipeline")
SparkPipelineModel.write().overwrite().save(r"C:\Users\A704081\projects\feedback-data-analyzer\SentimentAnalysis\Models\SparkPipeline")

spark.stop()