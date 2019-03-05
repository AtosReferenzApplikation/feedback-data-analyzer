# Code für WordCloud aus TF aus allen Dokumenten
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from ver2 import data13, vocablist
# Data:

text=data13.select("filtered2").take(data13.count())

help=[y for x in range(data13.count()) for y in text[x][0]]		
freq={vocablist[i]:0 for i in range(len(vocablist))}	
for i in help:
	if i in freq:
		freq[i]+=1


# WordCloud:
wordcloud = WordCloud(width = 3000,height = 2000,background_color = 'black').fit_words(freq)
	
fig = plt.figure(
    figsize = (40, 30),\
    facecolor = 'k',\
    edgecolor = 'k')
	
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
