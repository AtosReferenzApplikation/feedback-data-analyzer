df = spark.read.text(r"C:\Users\A704194\projects\feedback-data-analyzer\Daten\TE_Englisch\pos")

regtok = regexTokenizer.transform(df)
filrow = remover1.transform(regtok).select("filtered").take(regtok.count())

for k in filrow:
    for n in k[0]:	
        n = lemmatizer.lemmatize(n, "v")
        n = lemmatizer.lemmatize(n)
        words.append(n)
    final.append(words)
    words = []

#####fertig erstellen
for x in final:
    hilf.append(Row(filtered=x))

fertig = spark.createDataFrame(hilf)

##ngdf = ngram.transform(fertig)
#vec = cvmodel.transform(ngdf)
vec = cvmodel.transform(fertig)
idf = idfmodel.transform(vec)
end = lrmodel.transform(idf)

langeliste = end.select("prediction").take(end.count())
for x in langeliste:
    for y in x:
        if int(y) == 0:
            Counternull += 1
        elif int(y) == 1:
            Countereins += 1

print("Auswertung = Positiv sind etwa", round(Countereins/12500 * 100), "Prozent, negativ sind etwa", round(Counternull/12500 * 100), "Prozent")

##Bei positiven Testdaten: 83% positiv, 17% negativ
##Bei negativen Testdaten: 81% negativ, 19% positiv
##Mit n grams pos: 1% schlechter, neg: gleiche Ergebnisse