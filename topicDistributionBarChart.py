import matplotlib.pyplot as plt
## Überarbeiten, da so nicht lösbar. Muss gruppiert werden
top=topics.select("topic").take(topics.count())
x=[top[i][0] for i in range(len(top))]

distri=data14.select("topicDistribution").take(data14.count())
y0=[distri[0][0][i] for i in range(len(distri[0][0]))]
y1=[distri[0][0][i] for i in range(len(distri[1][0]))]
y2=[distri[0][0][i] for i in range(len(distri[2][0]))]
y3=[distri[0][0][i] for i in range(len(distri[3][0]))]
y4=[distri[0][0][i] for i in range(len(distri[4][0]))]

w=0,2
ax=plt.subplot(111)
ax.bar(x,y0,label="Topic0",color='k')
ax.bar(x,y1,label="Topic1",color='r')
ax.bar(x,y2,label='Topic2',color='b')
ax.bar(x,y3,label='Topic3',color='g')
ax.bar(x,y4,label='Topic4',color='y')


plt.xlabel('topic')
plt.ylabel('distribution')
plt.title('Topic Distribution')
plt.legend()
plt.show()