# Code für WordCloud aus Features und IDF für 1 Dokument
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from TopicExtraction import data13, vocablist

# Data for WordCloud:
# This is a WordCloud for Document 0, change i for other Documents in dict comprehension below:
# text[i][0][x]
text=data13.select("IDFFeatures").take(data13.count())
dict={vocablist[x]:text[0][0][x] for x in range(len(text[0][0])) if text[0][0][x]!=0}

		
# Draw WordCloud:
wordcloud = WordCloud(width = 3000,height = 2000,background_color = 'black').fit_words(dict)
	
fig = plt.figure(
    figsize = (40, 30),\
    facecolor = 'k',\
    edgecolor = 'k')
	
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
