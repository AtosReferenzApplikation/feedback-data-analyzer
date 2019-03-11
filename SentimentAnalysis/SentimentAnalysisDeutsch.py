import re
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.common import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import StopWordsRemover

# Deutsche Sentiment Listen:
# Die Wortvarianten rechts werden vernachlässigt.
sentiment=spark.read.load(r"C:\Users\A704081\Downloads\Projekt\SentiWS_v1.8c_Negative.txt",format="csv",sep="\t")
c0=sentiment.select("_c0").take(sentiment.count())
WortWortartList=[re.split("[|]",c0[x][0]) for x in range(len(c0))]
sentiment2=spark.createDataFrame(WortWortartList)
Score=sentiment.select("_c1")
score=Score.take(Score.count())
WortScoreDict={WortWortartList[x][0]:float(score[x][0]) for x in range(Score.count())}

possenti=spark.read.load(r"C:\Users\A704081\Downloads\Projekt\SentiWS_v1.8c_Positive.txt",format="csv",sep="\t")
posc0=sentiment.select("_c0").take(sentiment.count())
posWortWortartList=[re.split("[|]",posc0[x][0]) for x in range(len(posc0))]
possentiment2=spark.createDataFrame(posWortWortartList)
posScore=possenti.select("_c1")
posscore=posScore.take(posScore.count())
posWortScoreDict={posWortWortartList[x][0]:float(posscore[x][0]) for x in range(posScore.count())}

# Daten Einlesen:
#tweets=spark.read.load(r"C:\Users\A704081\Downloads\Projekt\corpus_v1.0.tsv",format="csv",sep="\t")
data=spark.read.load(r"C:\Users\A704081\Downloads\GermEval-2018-Data-master\germeval2018.training.txt",format="csv",sep="\t")

# noch nicht angepasst:
# Text in SparkNLP einlesen:
documentAssembler = DocumentAssembler() \
	.setInputCol("_c0") \
	.setOutputCol("document")
data2=documentAssembler.transform(data)

# Text in SparkNLP's Token umwandeln:
tokenizer = Tokenizer() \
	.setInputCols(["document"]) \
	.setOutputCol("token") \
	.addInfixPattern("(\p{L}+)(n't\b)")
data3=tokenizer.transform(data2)

# Satzzeichen entfernen:
normalizer = Normalizer() \
	.setInputCols(["token"]) \
	.setOutputCol("normalized")
norm=normalizer.fit(data3)
data4=norm.transform(data3)

# Lemmatizen:
lemmatizer = Lemmatizer() \
	.setInputCols(["normalized"]) \
	.setOutputCol("lemma") \
	.setDictionary(r"C:\Users\A704081\Downloads\lemmatization-lists-master\lemmatization-de.txt", key_delimiter="\t", value_delimiter="\n")
lem=lemmatizer.fit(data4)
data5=lem.transform(data4)

# Ich verwende keinen Stemmer, da die Sentiment Listen keine gestemmten Wörter enthalten

# Aus SparkNLP auslesen:
finisher = Finisher() \
	.setInputCols(["lemma"]) \
	.setIncludeMetadata(False)\
	.setOutputAsArray(True)
data6=finisher.transform(data5)

# Stopwords entfernen:
stoplist=StopWordsRemover.loadDefaultStopWords("german")
remover = StopWordsRemover(inputCol="finished_lemma", outputCol="filtered", stopWords=stoplist)
data7=remover.transform(data6)


# Auswertung in Listenform
docs=data7.select("filtered").take(data7.count())
docList=[[docs[y][0][x] for x in range(len(docs[y][0]))] for y in range(len(docs))]

CounterList=[0 for x in range(data7.count())]

for x in range(data7.count()):
    for y in docList[x]:
        if y in posWortScoreDict:
            CounterList[x]+=posWortScoreDict[y]
        if y in WortScoreDict:
            CounterList[x]+=WortScoreDict[y]


CounterList2=[[0,0] for x in range(data7.count())]

for x in range(data7.count()):
    for y in docList[x]:
        if y in posWortScoreDict:
            CounterList2[x][0]+=posWortScoreDict[y]
        if y in WortScoreDict:
            CounterList2[x][1]+=WortScoreDict[y]