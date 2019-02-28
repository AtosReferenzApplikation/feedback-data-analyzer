## tfidf mit "value" "tf" und "idf"

tf = tfidf.select("tf").take(tfidf.count())
dictf = {}
for i in range (len(tf)):
    for j in range (len(tf[i])):
        for k in range (len(tf[i][j])):
            if k not in dictf:
                dictf [k] = 0


for i in range (len(tf)):
    for j in range (len(tf[i])):
        for k in range (len(tf[i][j])):
            dictf[k] += tf[i][j][k]

## erstellen eines Dicts mit value[i] und tf[i] 
dicwords = {}
for i in range(len(dict)):
    dicwords[dict[i]] = dictf[i]


from wordcloud import WordCloud
import matplotlib.pyplot as plt
wordcloud = WordCloud(width = 3000,height = 2000,background_color = 'black').fit_words(dicwords)
	
fig = plt.figure(
    figsize = (40, 30),\
    facecolor = 'k',\
    edgecolor = 'k')
	
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()