from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local").appName("SentimentAnalysis").getOrCreate()

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
norm=Normalizer.load(r"C:\Users\A704081\projects\feedback-data-analyzer\SentimentAnalysis\Models\Norm")
lem=Lemmatizer.load(r"C:\Users\A704081\projects\feedback-data-analyzer\SentimentAnalysis\Models\Lemmatizer")
model = CountVectorizer.load(r"C:\Users\A704081\projects\feedback-data-analyzer\SentimentAnalysis\Models\CV")
idfModel=IDF.load(r"C:\Users\A704081\projects\feedback-data-analyzer\SentimentAnalysis\Models\IDF")
lrmodel=LogisticRegression.load(r"C:\Users\A704081\projects\feedback-data-analyzer\SentimentAnalysis\Models\LogisticRegression")

# Testdaten
#testadf=spark.read.text(r"C:\Users\A704081\Downloads\Projekt\aclImdb_v1\aclImdb\test\neg")
#testbdf=spark.read.text(r"C:\Users\A704081\Downloads\Projekt\aclImdb_v1\aclImdb\test\pos")
testadf=spark.read.text(r"C:\Users\A704081\Desktop\TestA")
testbdf=spark.read.text(r"C:\Users\A704081\Desktop\TestB")

testadf=testadf.withColumn("label", lit(0))
testbdf=testbdf.withColumn("label", lit(1))
testneg=testadf.take(testadf.count())
testpos=testbdf.take(testbdf.count())
testalist=[[testneg[x][0],testneg[x][1],] for x in range(testadf.count())]
testblist=[[testpos[x][0],testpos[x][1],] for x in range(testbdf.count())]
testlist=testalist + testblist
df=spark.createDataFrame(testlist,["value","label"])

documentAssembler = DocumentAssembler() \
	.setInputCol("value") \
	.setOutputCol("document")
df2=documentAssembler.transform(df)

tokenizer = Tokenizer() \
	.setInputCols(["document"]) \
	.setOutputCol("token") \
	.addInfixPattern("(\p{L}+)(n't\b)")
df3=tokenizer.transform(df2)

normalizer = Normalizer() \
	.setInputCols(["token"]) \
	.setOutputCol("normalized")
df4=norm.transform(df3)

lemmatizer = Lemmatizer() \
	.setInputCols(["normalized"]) \
	.setOutputCol("lemma") \
	.setDictionary(r"C:\Users\A704081\Downloads\lemmatization-lists-master\lemmatization-en.txt", key_delimiter="\t", value_delimiter="\n")
df5=lem.transform(df4)

finisher = Finisher() \
	.setInputCols(["lemma"]) \
	.setIncludeMetadata(False)\
	.setOutputAsArray(True)
df6=finisher.transform(df5)

remover = StopWordsRemover(inputCol="finished_lemma", outputCol="filtered")
df7=remover.transform(df6)

cv = CountVectorizer(inputCol="filtered", outputCol="features")
testvocablist=model.vocabulary
df8 = model.transform(df7)

idf = IDF(inputCol="features", outputCol="IDFFeatures")
df9 = idfModel.transform(df8)

lr=LogisticRegression(maxIter=100)
df10=lrmodel.transform(df9)

df10.select("value","label","prediction").show()
accuracy = df10.filter(df10.label == df10.prediction).count() / float(df10.count())
print("\n",accuracy,"\n")

spark.stop()