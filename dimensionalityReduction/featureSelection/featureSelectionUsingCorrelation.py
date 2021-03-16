import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Loading the data from a CSV file
data = pd.read_csv('./data/train.csv')
print(data.head())

# Getting X and Y from the dataframe
X = data.iloc[:, 0:20]
Y = data.iloc[:, -1]

# Finding the correlation between various variables
data_corr = data.corr()
print(data_corr)

# Plotting the heatmap for the above correlation data
plt.figure(figsize=(20, 20))
sns.heatmap(data_corr, annot=True)
plt.show()
