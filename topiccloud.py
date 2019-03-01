from wordcloud import WordCloud
import matplotlib.pyplot as plt


# Dictionary mit Zahl:Gewichtung
topic0dict={}
term=topics.select("termIndices").take(topics.count())
weight=topics.select("termWeights").take(topics.count())

for i in range(len(term[0][0])):
    topic0dict[term[0][0][i]]=weight[0][0][i]


# Dictionary mit Wort:Gewichtung
dict0={}
for i in term[0][0]:
    dict0[vocablist[i]]=topic0dict[i]

# WordCloud:
		
wordcloud = WordCloud(width = 3000,height = 2000,background_color = 'black').fit_words(dict0)
	
fig = plt.figure(
    figsize = (40, 30),\
    facecolor = 'k',\
    edgecolor = 'k')
	
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
