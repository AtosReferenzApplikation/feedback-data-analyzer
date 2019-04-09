from pyspark.ml.feature import RegexTokenizer
from pyspark.ml.feature import StopWordsRemover
from nltk.stem import WordNetLemmatizer

additionalstopwords = ["doesn", "didn", "isn", "wasn", "get", "realli", "re", \
        "shouldn", "tho", "everi", "br"]

regexTokenizer = RegexTokenizer(inputCol="value", outputCol="words", pattern="\\W")
remover1 = StopWordsRemover(inputCol="words", outputCol="filtered")

lemmatizer = WordNetLemmatizer()
words = []
all = []
final = []

# Englische Sentiment Liste:
neg = spark.read.text(r"C:\Users\A704194\projects\Spark_PP1\opinion-lexicon-English\negative-words.txt")
pos = spark.read.text(r"C:\Users\A704194\projects\Spark_PP1\opinion-lexicon-English\positive-words.txt")
negl = neg.take(neg.count())
neglist = [negl[x][0] for x in range(len(negl))]
posl = pos.take(pos.count())
poslist = [posl[x][0] for x in range(len(posl))]

# Datei als DataFrame einlesen:
df = spark.read.text(r"C:\Users\A704194\projects\feedback-data-analyzer\TR\Projekt\test\pos")

regtok = regexTokenizer.transform(df)
rem1 = remover1.transform(regtok)
filrow = rem1.select("filtered").take(rem1.count())

for k in filrow:
    for n in k[0]:
        n = lemmatizer.lemmatize(n, "v")
        n = lemmatizer.lemmatize(n)
        words.append(n)
    all.append(words)
    words = []

for x in all:
    words = [y for y in x if y not in additionalstopwords]
    final.append(words)

# List as Counter:
CounterList = [0 for x in range(len(final))]

for x in range(len(final)):
    for y in final[x]:
        if y in poslist:
            CounterList[x] += 1
        if y in neglist:
            CounterList[x] -= 1


# Separated Counter/ nested list [pos,neg]:
CounterList2=[[0,0] for x in range(len(final))]

for x in range(len(final)):
    for y in final[x]:
        if y in poslist:
            CounterList2[x][0] += 1
        if y in neglist:
            CounterList2[x][1] -= 1

Auswertung = [0, 0, 0]
for x in range(len(CounterList)):
    if CounterList[x] == 0:
        Auswertung[1] += 1
    elif CounterList[x] > 0:
        Auswertung[0] += 1
    elif CounterList[x] < 0:
        Auswertung[2] += 1

## Ergebnis:
# 17736 pos = 70.944%
# 5892 neg = 23.568%
# 1372 neut = 5.488%