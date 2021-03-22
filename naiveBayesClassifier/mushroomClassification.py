import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Loading the data
df = pd.read_csv('./data/mushrooms.csv')
print(df.head())
print(df.shape)

# Encoding the categorical data into numerical data
le = LabelEncoder()
# .apply method is used to apply a method on each column or row of the dataframe
ds = df.apply(le.fit_transform)
print(ds.head())
print(ds.shape)
# Converting the ds dataframe into a numpy array now
data = ds.values
print(data.shape)
print(type(data))
print(data[:5, :])
# Also breaking the data into X and Y arrays
data_X = data[:, 1:]
data_Y = data[:, 0]
print(data_X.shape, data_Y.shape)

# Break the data into train and test
x_train, x_test, y_train, y_test = train_test_split(
    data_X, data_Y, test_size=0.2)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# Checking how many unique classes of mushroom we have
np.unique(y_train)

# De-encoding data back from numerical data to categorical data
# For this purpose you can use inverse_transform method . Read more about it in documentation of scikit-learn


# Building our classifier


# Creating a method to find prior probability in Bayes formula
def prior_prob(y_train, label):
    total_examples = y_train.shape[0]
    # Finding out the total number of cases in which y_train is same as label
    class_examples = np.sum(y_train == label)
    prior_probability = class_examples/total_examples
    # Note that we apply a float type conversion over our denominator as our probability is going to be
    # between 0 and 1 so returning an integer would always give 0 or 1 but probability is a floating point
    # number hence we do the type conversion so that the final answer of our division is of the type float
    return prior_probability


# Creating a method to find conditional probability P(x_i|y=c)
def conditional_prob(x_train, y_train, feature_col, feature_val, label):
    x_filtered = x_train[y_train == label]
    number_of_examples_with_feature_val_and_label = np.sum(
        x_filtered[:, feature_col] == feature_val)
    total_examples_with_label = np.sum(y_train == label)
    conditional_probability = number_of_examples_with_feature_val_and_label / \
        total_examples_with_label
    return conditional_probability


# Creating a method to find predict/classify our test value x_test using Bayes formula
def predict(x_train, y_train, x_test):
    '''x_test is a single testing point, containing n features'''
    classes = np.unique(y_train)
    n_features = x_train.shape[1]
    # posterior_probs is a list containing probabilities for all classes for the given x_test
    posterior_probs = []
    # Computing posterior probability for each label/class
    for label in classes:
        # Calculating likelihood
        likelihood = 1.0
        for f in range(n_features):
            cond = conditional_prob(x_train, y_train, f, x_test[f], label)
            likelihood *= cond
        # Calculating prior probability
        prior = prior_prob(y_train, label)
        # Putting above values in the formula - posterior = likelihood * prior_probability to get posterior
        posterior_prob = likelihood*prior
        posterior_probs.append(posterior_prob)
    pred = np.argmax(posterior_probs)
    return pred


output = predict(x_train, y_train, x_test[0])
print(output)
print(y_test[0])


# Creating a function to find out the score of our model
def score(x_train, y_train, x_test, y_test):
    predictions = []
    for i in range(x_test.shape[0]):
        pred = predict(x_train, y_train, x_test[i])
        predictions.append(pred)

    predictions = np.array(predictions)
    accuracy = np.sum(predictions == y_test)/y_test.shape[0]
    return accuracy


# Finding the score of our model
model_score = score(x_train, y_train, x_test, y_test)
print(model_score*100)
