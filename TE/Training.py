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
from pyspark.ml.clustering import LDA, LDAModel

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

TENLPPipeline=Pipeline(stages=[documentAssembler,tokenizer,normalizer,lemmatizer,stemmer,normalizer2,finisher])
TENLPPipelineModel=TENLPPipeline.fit(data)
data2=TENLPPipelineModel.transform(data)
TENLPPipelineModel.write().overwrite().save(r"C:\Users\A704081\projects\feedback-data-analyzer\TE\Models\TENLPPipeline")

# SparkPipeline bauen
remover = StopWordsRemover(inputCol="finished_norm", outputCol="filtered")
stoplist=["br", "movi", "film", "thi", "hi", "thei", "episod", "seri", "much", "sai"]
remov=StopWordsRemover(inputCol="filtered", outputCol="filtered2", stopWords=stoplist)
cv = CountVectorizer(inputCol="filtered2", outputCol="features", minDF=10)
idf = IDF(inputCol="features", outputCol="IDFFeatures")
lda = LDA(k=80, maxIter=10)

TESparkPipeline=Pipeline(stages=[remover,remov,cv,idf,lda])
TESparkPipelineModel=TESparkPipeline.fit(data2)
data3=TESparkPipelineModel.transform(data2)
TESparkPipelineModel.write().overwrite().save(r"C:\Users\A704081\projects\feedback-data-analyzer\TE\Models\TESparkPipeline")
#data3.write.save("TEdata3.parquet",format="parquet")
data3.write.save("TEdata3.json",format="json")
# Topics
stages = TESparkPipelineModel.stages
ldalist = [s for s in stages if isinstance(s, LDAModel)]
lda=ldalist[0]
topics=lda.describeTopics(5)
#topics.write.save("TEtopics.parquet",format="parquet")
topics.write.save("TEtopics.json",format="json")

ll = lda.logLikelihood(data3)
lp = lda.logPerplexity(data3)
print("The lower bound on the log likelihood of the entire corpus: " + str(ll))
print("The upper bound on perplexity: " + str(lp))
spark.stop()