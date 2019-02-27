from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.sql.functions import col, udf
from nltk.tokenize import word_tokenize
from pyspark.ml.feature import StopWordsRemover
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer

df = spark.read.text(r"C:\Users\A704194\projects\Spark_PP1\Testdaten")
tokenizer = Tokenizer(inputCol="value", outputCol="words")
tok = tokenizer.transform(df)
regexTokenizer = RegexTokenizer(inputCol="value", outputCol="words", pattern="\\W")
regtok = regexTokenizer.transform(df)
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
rem = remover.transform(regtok)
remrow = remover.transform(regtok).take(regtok.count())
fil = rem.select("filtered")
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
fertig

remover2 = StopWordsRemover(inputCol="value", outputCol="filtered")
rem2 = remover2.transform(fertig)

fertig = rem2.selectExpr("filtered as value")