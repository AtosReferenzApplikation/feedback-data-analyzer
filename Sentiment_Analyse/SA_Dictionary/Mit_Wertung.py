import re
from pyspark.ml.feature import RegexTokenizer
from pyspark.ml.feature import StopWordsRemover
from nltk.stem import WordNetLemmatizer

additionalstopwords = ["doesn", "didn", "isn", "wasn", "get", "time", "know", "feel", "instead", "realli", "re", \
        "shouldn", "tho", "even", "well", "film","movi", "like", \
        "see", "everi", "great", "br", "good", "go", "think", "stori", "one", "make", "bad", "watch"]

regexTokenizer = RegexTokenizer(inputCol="_c0", outputCol="words", pattern="\\W")
remover1 = StopWordsRemover(inputCol="words", outputCol="filtered")
remover2 = StopWordsRemover(inputCol="value", outputCol="rem")
remover3 = StopWordsRemover(inputCol="rem", outputCol="filtered", stopWords = additionalstopwords)

lemmatizer = WordNetLemmatizer()
words = []

############################# negative Liste
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
df = spark.read.load(r"C:\Users\A704194\projects\feedback-data-analyzer\Daten\TR_Deutsch\Trainingstweets.txt",format="csv",sep="\t")

regtok = regexTokenizer.transform(df)
rem1 = remover1.transform(regtok)
filrow = rem1.select("filtered").take(rem1.count())

for k in filrow:
    for n in k[0]:
        n = lemmatizer.lemmatize(n, "v")
        n = lemmatizer.lemmatize(n)
        words.append(n)

all = [x for x in words if x not in additionalstopwords]
CounterList = [0]
words = []

for x in all:
	if x in posdict:
		CounterList[0] += posdict[x]
	elif x in negdict:
		CounterList[0] += negdict[x]
	else:
		words.append(x)

CounterList2 = [0,0]

for x in all:
	if x in posdict:
		CounterList2[0] += posdict[x]
	elif x in negdict:
		CounterList2[1] += negdict[x]