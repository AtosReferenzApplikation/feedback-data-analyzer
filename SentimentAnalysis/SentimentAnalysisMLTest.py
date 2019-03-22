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
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel
from pyspark.ml.feature import IDF, IDFModel
from pyspark.ml.clustering import LDA, LDAModel
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
norm=Normalizer.load(r"C:\Users\A704081\projects\feedback-data-analyzer\SentimentAnalysis\Models\Norm")
lem=Lemmatizer.load(r"C:\Users\A704081\projects\feedback-data-analyzer\SentimentAnalysis\Models\Lemmatizer")
model = CountVectorizerModel.load(r"C:\Users\A704081\projects\feedback-data-analyzer\SentimentAnalysis\Models\CV")
idfModel=IDFModel.load(r"C:\Users\A704081\projects\feedback-data-analyzer\SentimentAnalysis\Models\IDF")
lrmodel=LogisticRegressionModel.load(r"C:\Users\A704081\projects\feedback-data-analyzer\SentimentAnalysis\Models\LogisticRegression")

# Testdaten
testadf=spark.read.text(r"C:\Users\A704081\Downloads\Projekt\aclImdb_v1\aclImdb\test\neg")
testbdf=spark.read.text(r"C:\Users\A704081\Downloads\Projekt\aclImdb_v1\aclImdb\test\pos")
#testadf=spark.read.text(r"C:\Users\A704081\Desktop\TestA")
#testbdf=spark.read.text(r"C:\Users\A704081\Desktop\TestB")

testadf=testadf.withColumn("label", lit(0))
testbdf=testbdf.withColumn("label", lit(1))
testneg=testadf.take(testadf.count())
testpos=testbdf.take(testbdf.count())
testalist=[[testneg[x][0],testneg[x][1],] for x in range(testadf.count())]
testblist=[[testpos[x][0],testpos[x][1],] for x in range(testbdf.count())]
testlist=testalist + testblist
df=spark.createDataFrame(testlist,["value","label"])
df2=documentAssembler.transform(df)
df3=tokenizer.transform(df2)
df4=norm.transform(df3)
df5=lem.transform(df4)
df6=finisher.transform(df5)
df7=remover.transform(df6)
testvocablist=model.vocabulary
df8 = model.transform(df7)
df9 = idfModel.transform(df8)
df10=lrmodel.transform(df9)

df10.show()
accuracy = df10.filter(df10.label == df10.prediction).count() / float(df10.count())
print("\n",accuracy,"\n")

spark.stop()