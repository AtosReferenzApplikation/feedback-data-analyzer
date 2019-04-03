plmodel = PipelineModel.load(r"C:\Users\A704194\projects\Spark_PP1\model\pl")

df = spark.read.text(r"C:\Users\A704194\projects\feedback-data-analyzer\TR\Projekt\test\pos\0_10.txt")

regtok = regexTokenizer.transform(df)
filrow = remover1.transform(regtok).select("filtered").take(fil.count())

all = []

##nur möglich, wenn nur eine Zeile/ein Dokument:
for k in filrow:
        for m in k:
		for n in m:	
			n = lemmatizer.lemmatize(n, "v")
			n = lemmatizer.lemmatize(n)
			all.append(n)

dfneu = spark.createDataFrame([(all,)], ["value"])
rem2 = remover2.transform(dfneu)
fertig = remover3.transform(rem2).select("filtered")

#Vektor aus Dokumenten, tfidf-df erstellen
model = cv.fit(fertig)
dict = model.vocabulary
tf = model.transform(fertig).select("tf")
idfModel = idf.fit(tf)
tfidf = idfModel.transform(tf).select("features")

end = plmodel.transform(tfidf)