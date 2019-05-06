from pyspark.sql.functions import lit
from pyspark.ml.feature import RegexTokenizer
from pyspark.ml.feature import StopWordsRemover
from nltk.stem import WordNetLemmatizer

from pyspark.ml.feature import CountVectorizer, CountVectorizerModel
from pyspark.ml.feature import IDF, IDFModel
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel

from pyspark.ml.feature import NGram

regexTokenizer = RegexTokenizer(inputCol="value", outputCol="words", pattern="\\W")
remover1 = StopWordsRemover(inputCol="words", outputCol="filtered")
remover2 = StopWordsRemover(inputCol="value", outputCol="filtered")

lemmatizer = WordNetLemmatizer()
words = []
all = []
hilf = []
lab = []

##ngram = NGram(n=2, inputCol="filtered", outputCol="ngrams")
##cv = CountVectorizer(inputCol="ngrams", outputCol="tf")

cv = CountVectorizer(inputCol="filtered", outputCol="tf")
idf = IDF(inputCol="tf", outputCol="features")

lr = LogisticRegression()