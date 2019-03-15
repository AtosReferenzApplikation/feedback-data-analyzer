# Code zum Darstellen der Topic mit der höchsten übereinstimmung für ein Dokument 
from operator import itemgetter
from collections import OrderedDict
from TopicExtraction import data14, topiclist

distribution=data14.select("topicDistribution").take(data14.count())

# distridict = Zahl:Distribution-Wert nach Zahl/Key von 0 an
distridict={x:distribution[0][0][x] for x in range(len(distribution[0][0]))}

# gibt key:value pairs in nach größe geordneter Reihenfolge an, lässt sich aber nicht durch iterieren: 
sorted_distridict=OrderedDict(sorted(distridict.items(), key=itemgetter(1), reverse=True))

# keys und values in geordnete Listen, die sich durchiterieren lassen. Indices sind gleich
sorted_keys=list(sorted_distridict)
sorted_values=list(sorted_distridict.values())

# importantTopics zeigt die Wörter der Topics mit der höchsten Distribution an
# importantTopicsDistribution die dazugehörige Distribution
# Sollen mehr Topics angezeigt werden, muss der range() angepasst werden.
importantTopics = [topiclist[sorted_keys[x]] for x in range(3)]
importantTopicsDistribution=[sorted_values[x] for x in range(3)]