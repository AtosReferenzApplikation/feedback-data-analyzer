##Importe von Def_Model, Unterschied nur Cols bei remover2 und remover3

remover2 = StopWordsRemover(inputCol="value", outputCol="rem")
remover3 = StopWordsRemover(inputCol="rem", outputCol="filtered", stopWords = additionalstopwords)