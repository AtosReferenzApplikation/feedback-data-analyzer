from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.common import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import IDF
from pyspark.ml.clustering import LDA


data=spark.read.text(r"C:\Users\A704081\Desktop\Daten")

# Text in SparkNLP einlesen:
documentAssembler = DocumentAssembler() \
	.setInputCol("value") \
	.setOutputCol("document")
data2=documentAssembler.transform(data)

# Text in SparkNLP's Token umwandeln:
tokenizer = Tokenizer() \
	.setInputCols(["document"]) \
	.setOutputCol("token") \
	.addInfixPattern("(\p{L}+)(n't\b)")
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

# Stemmen:
stemmer = Stemmer() \
	.setInputCols(["lemma"]) \
	.setOutputCol("stem")	
data6=stemmer.transform(data5)

# Nochmal Normalizen, um leere Token zu entfernen:
normalizer = Normalizer() \
	.setInputCols(["stem"]) \
	.setOutputCol("norm")
norm=normalizer.fit(data6)
data7=norm.transform(data6)

# Aus SparkNLP auslesen:
finisher = Finisher() \
	.setInputCols(["norm"]) \
	.setIncludeMetadata(False)\
	.setOutputAsArray(True)
data8=finisher.transform(data7)

# Stopwords entfernen:
remover = StopWordsRemover(inputCol="finished_norm", outputCol="filtered")
data9=remover.transform(data8)

# Custom StopWords entfernen:
stoplist=["br", "movi", "film", "thi", "hi"]
remov=StopWordsRemover(inputCol="filtered", outputCol="filtered2", stopWords=stoplist)
data10=remov.transform(data9)

# Feature Vectoren erstellen:
cv = CountVectorizer(inputCol="filtered2", outputCol="features", vocabSize=50, minDF=2)
model = cv.fit(data10)
vocablist=model.vocabulary
data11 = model.transform(data10)

# IDF auf Term-Frequency-Vectoren:
idf = IDF(inputCol="features", outputCol="IDFFeatures")
idfModel = idf.fit(data11)
data12 = idfModel.transform(data11)

# LDA Clustering:
# k=Anzahl der Themen, maxIter=Anzahl der Wörter im Topic
lda = LDA(k=5, maxIter=5)
LDAmodel = lda.fit(data12)
ll = LDAmodel.logLikelihood(data12)
lp = LDAmodel.logPerplexity(data12)
print("The lower bound on the log likelihood of the entire corpus: " + str(ll))
print("The upper bound on perplexity: " + str(lp))
# die Zahl hinter describeTopics ist die Anzahl der Wörter die angezeigt werden soll, maximal maxIter von oben
topics = LDAmodel.describeTopics(5)
print("The topics described by their top-weighted terms:")
topics.show(truncate=False)
data13 = LDAmodel.transform(data12)

# Topics mit Wörtern anzeigen lassen:

indices=topics.select("termIndices").take(topics.count())
for x in range(len(indices)):
	for i in indices[x][0]:
		print(i, "\t" ,":" ,"\t" ,vocablist[i])
	print("\n")
	

data14= data13.select("filtered2","IDFFeatures","topicDistribution")

