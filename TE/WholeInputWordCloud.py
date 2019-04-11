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
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# PipelineModel
#TENLPPipelineModel=PipelineModel.load(r".\projects\feedback-data-analyzer\TE\Models\TENLPPipeline"")
TESparkPipelineModel = PipelineModel.load(r"C:\Users\A704081\projects\feedback-data-analyzer\TE\Models\TESparkPipeline")
# Data
data=spark.read.load(r"C:\Users\A704081\projects\feedback-data-analyzer\TE\TEdata3.parquet")
# Vocabulary
stages = TESparkPipelineModel.stages
vectorizers = [s for s in stages if isinstance(s, CountVectorizerModel)]
vocablists=[v.vocabulary for v in vectorizers]
vocablist=vocablists[0]

text=data.select("filtered2").take(data.count())
help=[y for x in range(data.count()) for y in text[x][0]]
freq={vocablist[i]:0 for i in range(len(vocablist))}	
for i in help:
	if i in freq:
		freq[i]+=1

# WordCloud:
wordcloud = WordCloud(width = 3000,height = 2000,background_color = 'black').fit_words(freq)
	
fig = plt.figure(
    figsize = (40, 30),\
    facecolor = 'k',\
    edgecolor = 'k')
	
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

spark.stop()