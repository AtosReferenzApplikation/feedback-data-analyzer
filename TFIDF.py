from pyspark.ml.feature import HashingTF, IDF, Tokenizer

## fertig == DF von Vorverarbeitung, "value" mit Arrays

hashingTF = HashingTF(inputCol="value", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(fertig)
# alternatively, CountVectorizer can also be used to get term frequency vectors

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

rescaledData.select("value", "features").show()