from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Loading the data from a CSV file
data = pd.read_csv('./data/train.csv')
print(data.head())

# Getting X and Y from the dataframe
X = data.iloc[:, 0:20]
Y = data.iloc[:, -1]

# Initialising RandomForestClassifier model
model = RandomForestClassifier()
model.fit(X, Y)
# .feature_importances_ gives the importance of each feature with respect to the output
print(model.feature_importances_)
# Converting the above model.feature_importances_ into a pandas dataframe to apply sorting
featureImportance = pd.DataFrame(
    model.feature_importances_, index=X.columns, columns=['Importance'])
featureImportance.sort_values(by='Importance', ascending=False, inplace=True)
print(featureImportance)

# Plotting the importance of various features on a bar graph
plt.figure(figsize=(20, 10))
plt.bar(featureImportance.index, featureImportance['Importance'])
plt.show()
