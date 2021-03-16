import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

# Loading the data from a CSV file
data = pd.read_csv('./data/train.csv')
print(data.head())

# Getting X and Y from the dataframe
X = data.iloc[:, 0:20]
Y = data.iloc[:, -1]

# Initialising the RandomForestClassifier model
rfc = RandomForestClassifier()

# Finding scores using cross_val_score
# Read more about cross_val_score at -
# https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
scores = cross_val_score(rfc, X, Y, cv=10)
print(scores)
mean_score = scores.mean()
print(mean_score)

# Now implementing feature selection and then using that to get the scores using cross_val_score
# Initialising the function of SelectKBest
best_features = SelectKBest(score_func=chi2, k=10)
# Fitting the above for our X and Y to get the 10 best features
fit = best_features.fit(X, Y)
# .scores_ gives us the relevance of each feature with respect to the output
# so the higher the score the better the feature
# Converting the above fit.scores_ into a pandas dataframe to apply sorting easily
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
# Concatenating the above two dataframes
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ["Features", "Scores"]
featureScores.sort_values(by='Scores', ascending=False, inplace=True)
ten_features = featureScores.head(10)['Features'].values

# Again calculating cross_val_score
# Read more about cross_val_score at -
# https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
scores = cross_val_score(rfc, X[ten_features], Y, cv=10)
print(scores)
mean_score = scores.mean()
print(mean_score)
