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
rem2 = remover2.transform(dfneu)
fertig = remover3.transform(rem2).select("filtered")

#Vektor aus Dokumenten, tfidf-df erstellen
tf = cvmodel.transform(fertig)
tfidf = idfmodel.transform(tf)
end = lrmodel.transform(tfidf).select("prediction")

for x in end.take(end.count()):
	if int(x[0]) == 0:
		ergebnis = "neg"
	elif int(x[0]) == 1:
		ergebnis = "pos"
	print(ergebnis)