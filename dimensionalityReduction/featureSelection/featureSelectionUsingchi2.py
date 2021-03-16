import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

# Loading the data from a CSV file
data = pd.read_csv('./data/train.csv')
print(data.head())

# Getting X and Y from the dataframe
X = data.iloc[:, 0:20]
Y = data.iloc[:, -1]

# Initialising the function of SelectKBest
best_features = SelectKBest(score_func=chi2, k=10)

# Fitting the above for our X and Y to get the 10 best features
fit = best_features.fit(X, Y)
# .scores_ gives us the relevance of each feature with respect to the output
# so the higher the score the better the feature
print(fit.scores_)
# Converting the above fit.scores_ into a pandas dataframe to apply sorting easily
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
# Concatenating the above two dataframes
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ["Features", "Scores"]
featureScores.sort_values(by='Scores', ascending=False, inplace=True)
print(featureScores)

# Plotting the relevance of various features on a bar graph
plt.figure(figsize=(20, 10))
plt.bar(featureScores['Features'], featureScores['Scores'])
plt.show()

# So now we can easily select the top 10 or so features accordingly as we have a sorted table with each
# features relevance to the output and finally after choosing the features we can train our model accordingly
