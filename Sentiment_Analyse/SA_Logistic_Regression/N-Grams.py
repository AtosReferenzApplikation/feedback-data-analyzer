from pyspark.ml.feature import RegexTokenizer
from pyspark.ml.feature import StopWordsRemover
from nltk.stem import WordNetLemmatizer
from pyspark.ml.feature import NGram

regexTokenizer = RegexTokenizer(inputCol="value", outputCol="words", pattern="\\W")
remover1 = StopWordsRemover(inputCol="words", outputCol="filtered")
remover2 = StopWordsRemover(inputCol="value", outputCol="filtered")

lemmatizer = WordNetLemmatizer()
all = []

df = spark.read.text(r"C:\Users\A704194\projects\feedback-data-analyzer\Daten\TE_Englisch\pos\0_10.txt")

regtok = regexTokenizer.transform(df)
filrow = remover1.transform(regtok).select("filtered").take(regtok.count())

##nur möglich, wenn nur eine Zeile/ein Dokument:
for k in filrow:
	for n in k[0]:	
		n = lemmatizer.lemmatize(n, "v")
		n = lemmatizer.lemmatize(n)
		all.append(n)

dfneu = spark.createDataFrame([(all,)], ["value"])
fertig = remover2.transform(dfneu).select("filtered")

ngram = NGram(n=2, inputCol="filtered", outputCol="ngrams")
ngdf = ngram.transform(fertig)