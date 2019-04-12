from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local").appName("SentimentAnalysis").getOrCreate()
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)

from pyspark.sql.functions import lit
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import NGram
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import IDF
from pyspark.ml.classification import LogisticRegression

# Txt Dateien einlesen
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

# Pipeline
tokenizer = Tokenizer(inputCol="value", outputCol="tokens")
ngrams = NGram(n=2, inputCol="tokens", outputCol="ngrams")
cv = CountVectorizer(inputCol="ngrams", outputCol="features")
idf = IDF(inputCol="features", outputCol="IDFFeatures")
lr=LogisticRegression()

SparkPipeline=Pipeline().setStages([tokenizer,ngrams,cv,idf,lr])
SparkPipelineModel=SparkPipeline.fit(data)
data2=SparkPipelineModel.transform(data)
SparkPipelineModel.write().overwrite().save(r"C:\Users\A704081\projects\feedback-data-analyzer\SA\Models\SAPipeline")

data2.show()

spark.stop()