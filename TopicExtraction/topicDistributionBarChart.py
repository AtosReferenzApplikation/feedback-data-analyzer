# Dies ist Code für ein Bar chart für 5 Dokumente für 5 Topics 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy
from TopicExtraction import topics, data14

top=topics.select("topic").take(topics.count())
#x=[top[i][0] for i in range(len(top))]

distri=data14.select("topicDistribution").take(data14.count())
y0=[distri[0][0][i] for i in range(len(distri[0][0]))]
y1=[distri[1][0][i] for i in range(len(distri[1][0]))]
y2=[distri[2][0][i] for i in range(len(distri[2][0]))]
y3=[distri[3][0][i] for i in range(len(distri[3][0]))]
y4=[distri[4][0][i] for i in range(len(distri[4][0]))]

x=numpy.arange(len(y0))

w = 0.15
plt.bar(x, y0, width=w, color='red', zorder=2)
plt.bar(x+ w, y1, width=w, color='blue', zorder=2)
plt.bar(x+ w*2, y2, width=w, color='orange', zorder=2)
plt.bar(x+ w*3, y3, width=w, color='green', zorder=2)
plt.bar(x+ w*4, y4, width=w, color='yellow', zorder=2)

plt.xticks(x+w*2,['Topic0','Topic1','Topic2','Topic3','Topic4'])
plt.title('Topic Distribution')
plt.xlabel('Topics')
plt.ylabel('Distribution')

red=mpatches.Patch(color='red',label='Document0')
blue=mpatches.Patch(color='blue',label='Document1')
orange=mpatches.Patch(color='orange',label='Document2')
green=mpatches.Patch(color='green',label='Document3')
yellow=mpatches.Patch(color='yellow',label='Document4')

plt.legend(handles=[red,blue,orange,green,yellow])
plt.grid(axis='y')
plt.show()