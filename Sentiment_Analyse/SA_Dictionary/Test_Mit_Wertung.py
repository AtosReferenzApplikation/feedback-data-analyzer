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
ngstmt = spark.read.load(r"C:\Users\A704081\Downloads\Projekt\SentiWS_v1.8c_Negative.txt",format="csv",sep="\t")
negc0 = ngstmt.select("_c0").take(ngstmt.count())
negwordlist = [re.split("[|]", negc0[x][0]) for x in range(len(negc0))]
negsentiment2 = spark.createDataFrame(negwordlist)
negscore = ngstmt.select("_c1").take(ngstmt.count())

NegScoreDict = {negwordlist[x][0]:float(negscore[x][0]) for x in range(negscore.count())}

############################# zusatzSplit und zusatzList sind die selbe Liste, ich lösche aber keine, um Folgefehler zu vermeiden

zusatz = ngstmt.select("_c2").take(ngstmt.count())
zusatzSplit = [re.split("," ,zusatz[x][0]) if zusatz[x][0]!=None else "Leer" for x in range(len(zusatz))]
zusatzList = [[i for i in zusatzSplit[x]] if zusatzSplit[x]!= "Leer" else "Leer" for x in range(len(zusatz))]
negWortDict = {wort:float(negscore[x][0]) for x in range(negscore.count()) for wort in zusatzList[x]}

#############################

psstmt = spark.read.load(r"C:\Users\A704081\Downloads\Projekt\SentiWS_v1.8c_Positive.txt",format="csv",sep="\t")
posc0 = ngstmt.select("_c0").take(ngstmt.count())
poswordlist = [re.split("[|]", posc0[x][0]) for x in range(len(posc0))]
possentiment2 = spark.createDataFrame(poswordlist)
posscore = psstmt.select("_c1").take(psstmt.count())

PosScoreDict = {poswordlist[x][0]:float(posscore[x][0]) for x in range(psstmt.count())}

############################# poszusatzSplit und poszusatzList sind die selbe Liste, ich lösche aber keine, um Folgefehler zu vermeiden
poszusatz = psstmt.select("_c2").take(psstmt.count())
poszusatzSplit = [re.split("," ,poszusatz[x][0]) if poszusatz[x][0]!=None else "Leer" for x in range(len(poszusatz))]
poszusatzList = [[i for i in poszusatzSplit[x]] if poszusatzSplit[x]!= "Leer" else "Leer" for x in range(len(poszusatz))]
posWortDict = {wort:float(posscore[x][0]) for x in range(psstmt.count()) for wort in poszusatzList[x]}

############################# 
#Daten Einlesen:
#tweets=spark.read.load(r"C:\Users\A704081\Downloads\Projekt\corpus_v1.0.tsv",format="csv",sep="\t")
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
CounterList = [[0] for x in range(fertig.count())]

for x in range(fertig.count()):
    for y in all[x]:
        if y in PosScoreDict:
            CounterList[x] += PosScoreDict[x]
        if y in NegScoreDict:
            CounterList[x] += NegScoreDict[x]
        if y in posWortDict:
            CounterList[x] += posWortDict[x]
        if y in negWortDict:
            CounterList[x] += negWortDict[x]


CounterList2 = [[0,0] for x in range(fertig.count())]

for x in range(fertig.count()):
    for y in all[x]:
        if y in PosScoreDict:
            CounterList2[x][0] += PosScoreDict[x]
        if y in NegScoreDict:
            CounterList2[x][1] += NegScoreDict[x]
        if y in posWortDict:
            CounterList2[x][0] += posWortDict[x]
        if y in negWortDict:
            CounterList2[x][1] += negWortDict[x]

posWortDict = {}
negWortDict = {}
NegScoreDict = {}
PosScoreDict = {}
listepwd = [[0.1,"go"], [0.2, "see"],[0.4,"know"],[0.9,"ashton"],[0.9,"kutcher"],[0.9,"comedy"],[0.9,"friend"],[0.1,"mine"],[0.2,"play"],[0.3,"theater"],[0.4,"emotions"]]
listenwd = [[-0.85, "judge"], [-0.9, "wrong"], [-0.6, "night"], [-0.4, "coax"], [-0.2, "admit"], [-0.3, "reluctant"], [-0.1, "only"]]

for x in listepwd:
    posWortDict[x[1]] = x[0]

for x in listenwd:
    negWortDict[x[1]] = x[0]

PosScoreDict = posWortDict
NegScoreDict = negWortDict