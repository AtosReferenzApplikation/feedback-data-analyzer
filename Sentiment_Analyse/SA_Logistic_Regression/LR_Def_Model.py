from pyspark.sql.functions import lit
from pyspark.ml.feature import RegexTokenizer
from pyspark.ml.feature import StopWordsRemover
from nltk.stem import WordNetLemmatizer

from pyspark.ml.feature import CountVectorizer, CountVectorizerModel
from pyspark.ml.feature import IDF, IDFModel
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel

additionalstopwords = ["doesn", "didn", "isn", "wasn", "get", "realli", "re", "shouldn", "tho", "everi", "br"]

regexTokenizer = RegexTokenizer(inputCol="value", outputCol="words", pattern="\\W")
remover1 = StopWordsRemover(inputCol="words", outputCol="filtered")
remover2 = StopWordsRemover(inputCol="filtered", outputCol="rem")
remover3 = StopWordsRemover(inputCol="rem", outputCol="filtered", stopWords = additionalstopwords)

lemmatizer = WordNetLemmatizer()
words = []
all = []
hilf = []

cv = CountVectorizer(inputCol="filtered", outputCol="tf")
idf = IDF(inputCol="tf", outputCol="features")

lr = LogisticRegression()