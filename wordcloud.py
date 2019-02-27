from wordcloud import WordCloud
import matplotlib.pyplot as plt

# WordCloud:

text=data13.select("filtered2").take(data13.count())
helpdict=[]
freqdict={}
for x in range(data13.count()):
	for y in text[x][0]:
		helpdict.append(y)

		
for i in vocablist:
	freqdict[i]=0

	
for i in helpdict:
	if i in freqdict:
		freqdict[i]+=1

		
wordcloud = WordCloud(width = 3000,height = 2000,background_color = 'black').fit_words(freqdict)
	
fig = plt.figure(
    figsize = (40, 30),\
    facecolor = 'k',\
    edgecolor = 'k')
	
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
