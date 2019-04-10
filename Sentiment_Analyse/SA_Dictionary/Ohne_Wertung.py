from pyspark.ml.feature import RegexTokenizer
from pyspark.ml.feature import StopWordsRemover
from nltk.stem import WordNetLemmatizer

additionalstopwords = ["doesn", "didn", "isn", "wasn", "get", "realli", "re", "shouldn", "tho", "everi", "br"]

regexTokenizer = RegexTokenizer(inputCol="value", outputCol="words", pattern="\\W")
remover1 = StopWordsRemover(inputCol="words", outputCol="filtered")

lemmatizer = WordNetLemmatizer()
words = []
all = []

# Englische Sentiment Liste:
neg = spark.read.text(r"C:\Users\A704194\projects\Spark_PP1\opinion-lexicon-English\negative-words.txt")
pos = spark.read.text(r"C:\Users\A704194\projects\Spark_PP1\opinion-lexicon-English\positive-words.txt")
negl = neg.take(neg.count())
neglist = [negl[x][0] for x in range(len(negl))]
posl = pos.take(pos.count())
poslist = [posl[x][0] for x in range(len(posl))]

# Datei als DataFrame einlesen:
df = spark.read.text(r"C:\Users\A704194\projects\feedback-data-analyzer\Daten\TE_Englisch\neg\0_2.txt")

regtok = regexTokenizer.transform(df)
rem1 = remover1.transform(regtok)
filrow = rem1.select("filtered").take(rem1.count())

for k in filrow:
    for n in k[0]:
        n = lemmatizer.lemmatize(n, "v")
        n = lemmatizer.lemmatize(n)
        words.append(n)

all = [x for x in words if x not in additionalstopwords]

# List as Counter : !nur mgl wenn nur ein Dokument eingelesen, da nur eine Zeile
CounterList = [0]

for x in range(len(all)):
    if all[x] in poslist:
        CounterList[0] += 1
    if all[x] in neglist:
        CounterList[0] -= 1


# Separated Counter/ nested list [pos,neg]: !nur mgl wenn nur ein Dokument eingelesen, da nur eine Zeile
CounterList2=[0,0]

for x in range(len(all)):
        if all[x] in poslist:
            CounterList2[0] += 1
        if all[x] in neglist:
            CounterList2[1] -= 1

if CounterList[0] == 0:
        "Dokument ist neutral"
elif CounterList[0] > 0:
        "Dokument ist positiv"
elif CounterList[0] < 0:
        "Dokument ist negativ"