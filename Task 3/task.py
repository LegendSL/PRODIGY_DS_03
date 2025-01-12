# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
file_path = 'bank-additional-full.csv'
# Adding error handling and specifying quoting behavior
# Replace deprecated arguments with on_bad_lines
data = pd.read_csv(file_path, sep=';', quoting=0, on_bad_lines='warn') 
# quoting=pd.QUOTE_MINIMAL: Ensures that only unescaped quotes are considered string delimiters.
# on_bad_lines='warn': Prints a warning for lines with parsing errors and skips them.

# Display the first few rows of the dataset to understand its structure
data.head()

# Convert target variable 'y' to binary
data['y'] = data['y'].map({'yes': 1, 'no': 0})

# Drop the 'duration' column
data = data.drop(columns=['duration'])

# One-hot encode categorical variables
data = pd.get_dummies(data, drop_first=True)

# Split the data into features and target variable
X = data.drop(columns=['y'])
y = data['y']

# Handle NaN values in 'y' before splitting
# You can either drop rows with NaN values or impute them
# Here, we drop rows with NaN values in 'y'
X = X[y.notna()]  # Select rows in X where y is not NaN
y = y[y.notna()]  # Select rows in y where y is not NaN
# Alternatively, you could impute NaN values using SimpleImputer
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(strategy='most_frequent') # Replace with your desired strategy
# y = imputer.fit_transform(y.values.reshape(-1, 1))

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Display the shape of the datasets
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Train a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

accuracy, report

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train a Decision Tree Classifier on the resampled data
clf_resampled = DecisionTreeClassifier(random_state=42)
clf_resampled.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
y_pred_resampled = clf_resampled.predict(X_test)

# Evaluate the resampled model's performance
accuracy_resampled = accuracy_score(y_test, y_pred_resampled)
report_resampled = classification_report(y_test, y_pred_resampled)

print("Accuracy after resampling:", accuracy_resampled)
print("\nClassification Report:\n", report_resampled)

# Visualize the decision tree
plt.figure(figsize=(20, 10))
plot_tree(clf_resampled, feature_names=X.columns, class_names=['No', 'Yes'], filled=True, max_depth=3)
plt.title("Decision Tree Visualization")
plt.show()