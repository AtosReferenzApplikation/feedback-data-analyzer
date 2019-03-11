from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.common import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import StopWordsRemover

# Englische Sentiment Liste:
neg=spark.read.text(r"C:\Users\A704081\Downloads\opinion-lexicon-English\negative-words.txt")
pos=spark.read.text(r"C:\Users\A704081\Downloads\opinion-lexicon-English\positive-words.txt")
Neg=neg.take(neg.count())
neglist=[Neg[x][0] for x in range(len(Neg))]
Pos=pos.take(pos.count())
poslist=[Pos[x][0] for x in range(len(Pos))]

# Daten als DataFrame einlesen:
#data=spark.read.text(r"C:\Users\A704081\Downloads\Projekt\aclImdb_v1\aclImdb\train\neg")
data=spark.read.text(r"C:\Users\A704081\Desktop\Daten")

# Text in SparkNLP einlesen:
documentAssembler = DocumentAssembler() \
	.setInputCol("value") \
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
	.setDictionary(r"C:\Users\A704081\Downloads\lemmatization-lists-master\lemmatization-en.txt", key_delimiter="\t", value_delimiter="\n")
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
remover = StopWordsRemover(inputCol="finished_lemma", outputCol="filtered")
data7=remover.transform(data6)

#DataFrame zu nested list:
words=data7.select("filtered").take(data7.count())
docWordsList=[[words[y][0][x] for x in range(len(words[y][0]))] for y in range(len(words))]

# List as Counter :
CounterList=[0 for x in range(data7.count())]

for x in range(data7.count()):
    for y in docWordsList[x]:
        if y in poslist:
            CounterList[x]+=1
        if y in neglist:
            CounterList[x]-=1


# Separated Counter/ nested list [pos,neg]:
CounterList2=[[0,0] for x in range(data7.count())]

for x in range(data7.count()):
    for y in docWordsList[x]:
        if y in poslist:
            CounterList2[x][0]+=1
        if y in neglist:
            CounterList2[x][1]-=1