##Achtung: Schlechte Ergbnisse, da deutsche Daten mit englischem Lemmatizer

import re
from pyspark.ml.feature import RegexTokenizer
from pyspark.ml.feature import StopWordsRemover
from nltk.stem import WordNetLemmatizer

from pyspark.sql import Row

additionalstopwords = ["doesn", "didn", "isn", "wasn", "get", "time", "know", "feel", "instead", "realli", "re", \
        "shouldn", "tho", "even", "well", "film","movi", "like", \
        "see", "everi", "great", "br", "good", "go", "think", "stori", "one", "make", "bad", "watch"]

regexTokenizer = RegexTokenizer(inputCol="value", outputCol="words", pattern="\\W")
remover1 = StopWordsRemover(inputCol="words", outputCol="filtered")
remover2 = StopWordsRemover(inputCol="value", outputCol="rem")
remover3 = StopWordsRemover(inputCol="rem", outputCol="filtered", stopWords = additionalstopwords)

lemmatizer = WordNetLemmatizer()
words = []
all = []
final = []
rowsneu = []

############################# Deutsche Sentiment Listen:
ngstmt = spark.read.load(r"C:\Users\A704194\projects\feedback-data-analyzer\Daten\SentiWS_v1.8c_Negative.txt",format="csv",sep="\t")
ngstmtrows = ngstmt.select("_c0").take(ngstmt.count())
negwordlist = [re.split("[|]", ngstmtrows[x][0]) for x in range(len(ngstmtrows))]
negscore = ngstmt.select("_c1").take(ngstmt.count())
NegScoreDict = {negwordlist[x][0]:float(negscore[x][0]) for x in range(len(negscore))}

negzusatz = ngstmt.select("_c2").take(ngstmt.count())
negzusatzSplit = [re.split("," ,negzusatz[x][0]) if negzusatz[x][0]!=None else "Leer" for x in range(len(negzusatz))]
negWortDict = {wort:float(negscore[x][0]) for x in range(len(negscore)) for wort in negzusatzSplit[x]}

negdict = {**NegScoreDict, **negWortDict}
############################# positive Liste

psstmt = spark.read.load(r"C:\Users\A704194\projects\feedback-data-analyzer\Daten\SentiWS_v1.8c_Positive.txt",format="csv",sep="\t")
psstmtrows = psstmt.select("_c0").take(psstmt.count())
poswordlist = [re.split("[|]", psstmtrows[x][0]) for x in range(len(psstmtrows))]
posscore = psstmt.select("_c1").take(psstmt.count())
PosScoreDict = {poswordlist[x][0]:float(posscore[x][0]) for x in range(psstmt.count())}

poszusatz = psstmt.select("_c2").take(psstmt.count())
poszusatzSplit = [re.split(",", poszusatz[x][0]) if poszusatz[x][0] != None else "Leer" for x in range(len(poszusatz))]
posWortDict = {wort:float(posscore[x][0]) for x in range(psstmt.count()) for wort in poszusatzSplit[x]}

posdict = {**posWortDict, **PosScoreDict}
############################# 
#Daten Einlesen:
df = spark.read.text(r"C:\Users\A704194\projects\feedback-data-analyzer\Daten\TE_Deutsch\Testtweets.txt")

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
    rowsneu.append(Row(value=x))

worddf = spark.createDataFrame(rowsneu)

rem2 = remover2.transform(worddf)
fertig = remover3.transform(rem2).select("filtered")
take = fertig.take(fertig.count())
for x in take:
    words = [y for y in x[0]]
    final.append(words)
    words = []

######
CounterList = [0]

for x in range(fertig.count()):
    for y in all[x]:
        if y in posdict:
            CounterList[0] += posdict[y]
        elif y in negdict:
            CounterList[0] += negdict[y]


CounterList2 = [0,0]

for x in range(fertig.count()):
    for y in all[x]:
        if y in posdict:
            CounterList2[0] += posdict[y]
        elif y in negdict:
            CounterList2[1] += negdict[y]