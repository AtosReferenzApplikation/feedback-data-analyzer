from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local").appName("TESingleDocumentWordCloud").getOrCreate()
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)

# Importe
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.common import *
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import CountVectorizer,CountVectorizerModel
from pyspark.ml.feature import IDF
from pyspark.ml.clustering import LDA, LDAModel
from operator import itemgetter
from collections import OrderedDict

# PipelineModel
#TENLPPipelineModel=PipelineModel.load(r".\projects\feedback-data-analyzer\TE\Models\TENLPPipeline"")
TESparkPipelineModel = PipelineModel.load(r"C:\Users\A704081\projects\feedback-data-analyzer\TE\Models\TESparkPipeline")
# Vocabulary
stages = TESparkPipelineModel.stages
vectorizers = [s for s in stages if isinstance(s, CountVectorizerModel)]
vocablists=[v.vocabulary for v in vectorizers]
vocablist=vocablists[0]
# Data
data=spark.read.load(r"C:\Users\A704081\projects\feedback-data-analyzer\TE\TEdata3.parquet",format="parquet")
topics=spark.read.load(r"C:\Users\A704081\projects\feedback-data-analyzer\TE\TEtopics.json",format="json")
# Topics
indices=topics.select("termIndices").take(topics.count())
topiclist=[[vocablist[i] for i in indices[x][0]] for x in range(len(indices))]

distribution=data.select("topicDistribution").take(data.count())

# distridict = Zahl:Distribution-Wert nach Zahl/Key von 0 an
# Distribution für Dokument x
x=78
distridict={a:distribution[x][0][a] for a in range(len(distribution[x][0]))}

# gibt key:value pairs in nach größe geordneter Reihenfolge an, lässt sich aber nicht durch iterieren: 
sorted_distridict=OrderedDict(sorted(distridict.items(), key=itemgetter(1), reverse=True))

# keys und values in geordnete Listen, die sich durchiterieren lassen. Indices sind gleich
sorted_keys=list(sorted_distridict)
sorted_values=list(sorted_distridict.values())

# importantTopics zeigt die Wörter der Topics mit der höchsten Distribution an
# importantTopicsDistribution die dazugehörige Distribution
# Sollen mehr Topics angezeigt werden, muss der range() angepasst werden.
importantTopics = [topiclist[sorted_keys[x]] for x in range(3)]
importantTopicsDistribution=[sorted_values[x] for x in range(3)]

# TopicDistribution für Dokument x
importantTopics
importantTopicsDistribution
spark.stop()