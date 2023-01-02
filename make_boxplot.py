import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('only_last.csv')
print(df)
occ = df['o']
decom = df['d']
parallel = df['p']
exact = df['e']

# Create a figure instance

# sns boxplot with x = 'l' and y = 'd', 'p', 'e'
# sns.boxplot(x='l', y='d', data=df)

df.boxplot(column=['d'], by='l', grid=False, showfliers=False)
plt.ylabel("Solution Computation Time")
plt.xlabel("Structural Occupancy Percentage")
df.boxplot(column=['p'], by='l', grid=False, showfliers=False)
plt.ylabel("Solution Computation Time")
plt.xlabel("Structural Occupancy Percentage")
df.boxplot(column=['e'], by='l', grid=False, showfliers=False)

plt.ylabel("Solution Computation Time")
plt.xlabel("Structural Occupancy Percentage")
plt.show()