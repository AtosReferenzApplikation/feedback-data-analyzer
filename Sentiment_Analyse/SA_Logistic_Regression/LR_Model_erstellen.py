# Txt Dateien einlesen neg: \0_3.txt pos: \0_9.txt
negdf = spark.read.text(r"C:\Users\A704194\projects\feedback-data-analyzer\TR\Projekt\train\neg")
posdf = spark.read.text(r"C:\Users\A704194\projects\feedback-data-analyzer\TR\Projekt\train\pos")

# Label Spalte einfügen
# 0=negative, 1=positive
negdf = negdf.withColumn("label", lit(0))
posdf = posdf.withColumn("label", lit(1))

# DataFrames zusammenfügen
neg = negdf.take(negdf.count())
pos = posdf.take(posdf.count())
# Listen erstellen:
neglist = [[neg[x][0],neg[x][1],] for x in range(negdf.count())]
poslist = [[pos[x][0],pos[x][1],] for x in range(posdf.count())]
# Listen vereinen
mergedlist = neglist + poslist

# Aus vereinter Liste DataFrame erstellen
label = spark.createDataFrame(mergedlist,["value","label"])

regtok = regexTokenizer.transform(label)
rem1 = remover1.transform(regtok)
fil = rem1.select("filtered", "label")
filrow = fil.take(fil.count())

for k in filrow:
    for n in k[0]:
        n = lemmatizer.lemmatize(n, "v")
        n = lemmatizer.lemmatize(n)
        words.append(n)
    label = k[1]
    lab.append(words)
    lab.append(label)
    all.append(lab)
    lab = []
    words = []

fertig = spark.createDataFrame(all, ["filtered", "label"])

# Stopwords entfernen:
rem2 = remover2.transform(fertig).select("rem", "label")
fertig = remover3.transform(rem2).select("filtered", "label")

cvmodel = cv.fit(fertig)
dict = model.vocabulary
tf = model.transform(fertig)
idfmodel = idf.fit(tf)
tfidflab = idfmodel.transform(tf).select("features", "label")

lrmodel = lr.fit(tfidflab)

# Logistic Regression mit Pipeline
cvmodel.write().overwrite().save(r"C:\Users\A704194\projects\Spark_PP1\model\cv")
idfmodel.write().overwrite().save(r"C:\Users\A704194\projects\Spark_PP1\model\idf")
lrmodel.write().overwrite().save(r"C:\Users\A704194\projects\Spark_PP1\model\lr")