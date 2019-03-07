# Graph zur Verteilung der Dokumente in Cluster
 
import pandas as pd
import matplotlib.pyplot as plt

x = pd.Series(dictpred)
y = pd.Series.sort_values(x)
z = pd.DataFrame(y)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,9))

#specify which column of the dataframe to plot (here 0)
z.plot(y=0, kind = 'pie', ax = axes[0])
z.plot(kind = 'bar', ax = axes[1])
# make aspect equal (such that circle is not eliptic)
axes[0].set_aspect("equal")
#place the legend at a decent position
axes[0].legend(loc=1, bbox_to_anchor= (0,1.1), fontsize=8)
plt.show()