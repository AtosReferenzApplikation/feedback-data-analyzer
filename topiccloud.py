# Code für WordCloud aus Topic Wörtern und TopicWeight für 1 Topic
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from ver2 import topics, vocablist

term=topics.select("termIndices").take(topics.count())
weight=topics.select("termWeights").take(topics.count())

topicdict={vocablist[i]:weight[0][0][i] for i in range(len(term[0][0]))}

# WordCloud:
		
wordcloud = WordCloud(width = 3000,height = 2000,background_color = 'black').fit_words(topicdict)
	
fig = plt.figure(
    figsize = (40, 30),\
    facecolor = 'k',\
    edgecolor = 'k')
	
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
