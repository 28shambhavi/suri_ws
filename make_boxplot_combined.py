import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('timesteps_boxplot.csv')
print(df)

# Create a figure instance

# sns boxplot with x = 'l' and y = 'd', 'p', 'e'
# sns.boxplot(x='l', y='d', data=df)

g = sns.boxplot(y = 'v', x='l', hue='m', data=df)
# g.set(yscale="log")
handles, labels = g.get_legend_handles_labels()
g.legend(title = '')
plt.ylabel("No. of timesteps")
plt.xlabel("Structural Occupancy Percentage")

plt.show()