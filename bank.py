import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the data
train_data = pd.read_csv('train.csv', delimiter=';')
test_data = pd.read_csv('test.csv', delimiter=';')

# Check for missing values
print("Missing values in train data:\n", train_data.isnull().sum())
print("Missing values in test data:\n", test_data.isnull().sum())

# Prepare data for Logistic Regression
train_data.y.replace({'no': 0, 'yes': 1}, inplace=True)
test_data.y.replace({'no': 0, 'yes': 1}, inplace=True)
train_data.loan.replace({'no': 0, 'yes': 1}, inplace=True)
test_data.loan.replace({'no': 0, 'yes': 1}, inplace=True)
X_train = train_data.drop(columns=['job', 'marital', 'education', 'default', 'housing', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y'])
y_train = train_data.y
X_test = test_data.drop(columns=['job', 'marital', 'education', 'default', 'housing', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y'])
y_test = test_data.y
X_train.head()

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy}')

# Data analysis
def plot_bar_chart(data, feature, target='y', bins=None, bin_labels=None):
    data_copy = data.copy()  # Create a copy of the data
    if bins and bin_labels:
        data_copy[feature] = pd.cut(data_copy[feature], bins=bins, labels=bin_labels)
    data_copy[target] = data_copy[target].map({0: 'no', 1: 'yes'})  # Map 0 and 1 to 'no' and 'yes'
    counts = data_copy.groupby([feature, target]).size().unstack().fillna(0)
    counts.plot(kind='barh', stacked=True)  # Change to horizontal bar chart
    plt.title(f'Number of time deposits purchased by {feature}')
    plt.xlabel('Count')
    plt.ylabel(feature)
    plt.show()

# Define age bins and labels
age_bins = range(0, 101, 10)
age_labels = [f"{i}~{i+10}" for i in age_bins[:-1]]

# Plot bar charts for each characteristic
plot_bar_chart(train_data, 'age', bins=age_bins, bin_labels=age_labels)
plot_bar_chart(train_data, 'job')
plot_bar_chart(train_data, 'marital')
plot_bar_chart(train_data, 'education')
plot_bar_chart(train_data, 'loan')
plot_bar_chart(train_data, 'housing')
plot_bar_chart(train_data, 'contact')
plot_bar_chart(train_data, 'month')
plot_bar_chart(train_data, 'poutcome')