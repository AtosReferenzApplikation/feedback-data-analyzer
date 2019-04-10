from pyspark.ml.feature import RegexTokenizer
from pyspark.ml.feature import StopWordsRemover
from nltk.stem import WordNetLemmatizer

from pyspark.ml.feature import CountVectorizer, CountVectorizerModel
from pyspark.ml.feature import IDF, IDFModel
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel

from pyspark.sql import Row

regexTokenizer = RegexTokenizer(inputCol="value", outputCol="words", pattern="\\W")
remover1 = StopWordsRemover(inputCol="words", outputCol="filtered")
remover2 = StopWordsRemover(inputCol="value", outputCol="filtered")

lemmatizer = WordNetLemmatizer()
words = []
all = []
final = []
hilf = []

lr = LogisticRegression()

idfmodel = IDFModel.load(r"C:\Users\A704194\projects\Spark_PP1\model\idf")
cvmodel = CountVectorizerModel.load(r"C:\Users\A704194\projects\Spark_PP1\model\cv")
lrmodel = LogisticRegressionModel.load(r"C:\Users\A704194\projects\Spark_PP1\model\lr")

Counternull = 0
Countereins = 0
ergebnis = 5