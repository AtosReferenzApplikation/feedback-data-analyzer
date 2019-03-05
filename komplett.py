from pyspark.ml.feature import RegexTokenizer
from pyspark.ml.feature import StopWordsRemover
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer

df = spark.read.text(r"C:\Users\A704194\projects\Spark_PP1\Testdaten\GrosseTR")
regexTokenizer = RegexTokenizer(inputCol="value", outputCol="words", pattern="\\W")
regtok = regexTokenizer.transform(df)
remover1 = StopWordsRemover(inputCol="words", outputCol="filtered")
rem1 = remover1.transform(regtok)
fil = rem1.select("filtered")
filrow = fil.take(fil.count())

lemmatizer = WordNetLemmatizer()
lemv = []
lem = []
words = []
sno = SnowballStemmer("english")
stemsno = []

all = []
for k in filrow:
	for m in k:
		for n in m:	
			words.append(n)
		all.append(words)
		words = []

for p in all:
	for q in p:
		words.append(lemmatizer.lemmatize(q, "v"))
	lemv.append(words)
	words = []

for p in lemv:
	for q in p:
		words.append(lemmatizer.lemmatize(q))
	lem.append(words)
	words = []

for w in lem:
	for x in w:
		words.append(sno.stem(x))
	stemsno.append(words)
	words = []

brauch = []
hilf = []
for i in stemsno:
	hilf.append(i)
	brauch.append(hilf)
	hilf = []

fertig = spark.createDataFrame(brauch, ["value"])

remover2 = StopWordsRemover(inputCol="value", outputCol="rem")
rem2 = remover2.transform(fertig)

additionalstopwords = ["doesn", "didn", "isn", "wasn", "shouldn", "tho", "even", "well", "great", "br"]
remover3 = StopWordsRemover(inputCol="rem", outputCol="addsw", stopWords = additionalstopwords)
rem3 = remover3.transform(rem2)

fertig = rem3.selectExpr("addsw as value")

from pyspark.ml.feature import CountVectorizer
cv = CountVectorizer(inputCol="value", outputCol="tf", vocabSize = 300, minDF = 10)
model = cv.fit(fertig)

dict = model.vocabulary

tf = model.transform(fertig)
## result = DF mit value (array mit Wörtern) und features (TF)
## dic ist liste mit Wörtern, Dimemnsion 5 ist dic[5]

from pyspark.ml.feature import HashingTF, IDF, Tokenizer

idf = IDF(inputCol="tf", outputCol="idf")
idfModel = idf.fit(tf)
tfidf = idfModel.transform(tf)