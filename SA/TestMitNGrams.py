from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local").appName("SentimentAnalysisTestdaten").getOrCreate()
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)

from pyspark.sql.functions import lit
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel
from pyspark.ml.feature import IDF, IDFModel
from pyspark.ml.clustering import LDA, LDAModel
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel

SparkPipelineModel = PipelineModel.load(r"C:\Users\A704081\projects\feedback-data-analyzer\SA\Models\SAPipeline")

# Testdaten
testadf=spark.read.text(r"C:\Users\A704081\Downloads\Projekt\aclImdb_v1\aclImdb\test\neg")
testbdf=spark.read.text(r"C:\Users\A704081\Downloads\Projekt\aclImdb_v1\aclImdb\test\pos")

testadf=testadf.withColumn("label", lit(0))
testbdf=testbdf.withColumn("label", lit(1))
testneg=testadf.take(testadf.count())
testpos=testbdf.take(testbdf.count())
testalist=[[testneg[x][0],testneg[x][1],] for x in range(testadf.count())]
testblist=[[testpos[x][0],testpos[x][1],] for x in range(testbdf.count())]
testlist=testalist + testblist
df=spark.createDataFrame(testlist,["value","label"])
df.show()

# Zuordnung
df2=SparkPipelineModel.transform(df)
df2.select("value","label","prediction").show()
accuracy = df2.filter(df2.label == df2.prediction).count() / float(df2.count())
print("\n","Die Genauigkeit der Zuordnung beträgt: ",accuracy,"\n")

spark.stop()