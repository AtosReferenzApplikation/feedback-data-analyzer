# SparkSession und Zeichen
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("TopicExtraction").getOrCreate()

import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)

# Importe
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.common import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import CountVectorizer,CountVectorizerModel
from pyspark.ml.feature import IDF
from pyspark.ml.clustering import KMeans,KMeansModel

# Daten einlesen
data=spark.read.text(r"C:\Users\A704081\Downloads\Projekt\aclImdb_v1\aclImdb\train\neg")

# NLPPipeline bauen
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

stemmer = Stemmer() \
	.setInputCols(["lemma"]) \
	.setOutputCol("stem")	

normalizer2 = Normalizer() \
	.setInputCols(["stem"]) \
	.setOutputCol("norm")

finisher = Finisher() \
	.setInputCols(["norm"]) \
	.setIncludeMetadata(False)\
	.setOutputAsArray(True)

KNLPPipeline=Pipeline(stages=[documentAssembler,tokenizer,normalizer,lemmatizer,stemmer,normalizer2,finisher])
KNLPPipelineModel=KNLPPipeline.fit(data)
data2=KNLPPipelineModel.transform(data)
KNLPPipelineModel.write().overwrite().save(r"C:\Users\A704081\projects\feedback-data-analyzer\k-Means\Models\KNLPPipeline")

# SparkPipeline bauen
remover = StopWordsRemover(inputCol="finished_norm", outputCol="filtered")
stoplist=["br", "movi", "film", "thi", "hi", "thei", "episod", "seri", "much", "sai"]
remov=StopWordsRemover(inputCol="filtered", outputCol="filtered2", stopWords=stoplist)
cv = CountVectorizer(inputCol="filtered2", outputCol="features", minDF=10)
idf = IDF(inputCol="features", outputCol="IDFFeatures")
kmeans=KMeans(featuresCol="IDFFeatures",k=100,seed=1)

KSparkPipeline=Pipeline(stages=[remover,remov,cv,idf,kmeans])
KSparkPipelineModel=KSparkPipeline.fit(data2)
data3=KSparkPipelineModel.transform(data2)
KSparkPipelineModel.write().overwrite().save(r"C:\Users\A704081\projects\feedback-data-analyzer\k-Means\Models\KSparkPipeline")

data3.select("value","prediction").show()

stages = KSparkPipelineModel.stages
kmlist = [s for s in stages if isinstance(s, KMeansModel)]
km=kmlist[0]

centers = km.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)







spark.stop()