from pyspark.sql.functions import lit


# Txt Dateien einlesen
negdf=spark.read.text(r"C:\Users\A704081\Desktop\Daten")
posdf=spark.read.text(r"C:\Users\A704081\Desktop\PosDaten")
# Label Spalte einfügen
negdf=negdf.withColumn("label", lit("negative"))
posdf=posdf.withColumn("label", lit("positive"))

# DataFrames zusammenfügen
neg=negdf.take(negdf.count())
pos=posdf.take(posdf.count())
# Listen erstellen:
neglist=[[neg[x][0],neg[x][1],] for x in range(negdf.count())]
poslist=[[pos[x][0],pos[x][1],] for x in range(posdf.count())]
# Listen vereinen
mergedlist=neglist + poslist

# Aus vereinter Liste DataFrame erstellen
data=spark.createDataFrame(mergedlist,["value","label"])
