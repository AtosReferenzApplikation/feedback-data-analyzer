from pyspark.sql.functions import lit
from pyspark.ml.feature import RegexTokenizer
from pyspark.ml.feature import StopWordsRemover
from nltk.stem import WordNetLemmatizer

from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline, PipelineModel

additionalstopwords = ["doesn", "didn", "isn", "wasn", "get", "time", "know", "feel", "instead", "realli", "re", \
        "shouldn", "tho", "even", "well", "film","movi", "like", \
        "see", "everi", "great", "br", "good", "go", "think", "stori", "one", "make", "bad", "watch"]

regexTokenizer = RegexTokenizer(inputCol="value", outputCol="words", pattern="\\W")
remover1 = StopWordsRemover(inputCol="words", outputCol="filtered")
remover2 = StopWordsRemover(inputCol="filtered", outputCol="rem")
remover3 = StopWordsRemover(inputCol="rem", outputCol="filtered", stopWords = additionalstopwords)

lemmatizer = WordNetLemmatizer()
words = []
lab = []
all = []

cv = CountVectorizer(inputCol="filtered", outputCol="tf")
idf = IDF(inputCol="tf", outputCol="features")

lr = LogisticRegression()
pipeline = Pipeline().setStages([lr])