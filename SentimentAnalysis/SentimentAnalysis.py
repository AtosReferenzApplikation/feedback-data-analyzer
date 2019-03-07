import pyspark.sql.functions as f
import re

sentiment=spark.read.load(r"C:\Users\A704081\Downloads\Projekt\SentiWS_v1.8c_Negative.txt",format="csv",sep="\t")

c0=sentiment.select("_c0").take(sentiment.count())
WortWortartList=[re.split("[|]",c0[x][0]) for x in range(len(c0))]
sentiment2=spark.createDataFrame(WortWortartList)
Score=sentiment.select("_c1")
score=Score.take(Score.count())

WortScoreDict={WortWortartList[x][0]:score[x][0] for x in range(Score.count())}

tweets=spark.read.load(r"C:\Users\A704081\Downloads\Projekt\corpus_v1.0.tsv",format="csv",sep="\t")