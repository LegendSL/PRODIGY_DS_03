import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
train_path = 'train.csv'
test_path = 'test.csv'
gender_submission_path = 'gender_submission.csv'

# Load the train dataset
train_data = pd.read_csv(train_path)

# Display the first few rows and basic information to understand the structure
train_data.head(), train_data.info()

# Check the missing values in the dataset
missing_values = train_data.isnull().sum()

# Handle missing values
# Fill missing 'Age' with the median age
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)

# Fill missing 'Embarked' with the most common value (mode)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

# Drop the 'Cabin' column due to a high percentage of missing values
train_data.drop(columns=['Cabin'], inplace=True)

# Verify the changes
missing_values_after = train_data.isnull().sum()

missing_values, missing_values_after

# Set up visual styles
sns.set_theme(style="whitegrid")

# Survival distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=train_data, x='Survived', palette='pastel')
plt.title('Survival Count')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# Survival by gender
plt.figure(figsize=(6, 4))
sns.countplot(data=train_data, x='Survived', hue='Sex', palette='pastel')
plt.title('Survival Count by Gender')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.legend(title='Gender')
plt.show()

# Survival by passenger class
plt.figure(figsize=(6, 4))
sns.countplot(data=train_data, x='Survived', hue='Pclass', palette='pastel')
plt.title('Survival Count by Passenger Class')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.legend(title='Passenger Class')
plt.show()

# Survival distribution by Age
plt.figure(figsize=(8, 5))
sns.histplot(data=train_data, x='Age', hue='Survived', kde=True, palette='pastel', bins=30)
plt.title('Age Distribution by Survival')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

# Survival distribution by Fare
plt.figure(figsize=(8, 5))
sns.histplot(data=train_data, x='Fare', hue='Survived', kde=True, palette='pastel', bins=30)
plt.title('Fare Distribution by Survival')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

# Correlation heatmap for numerical features
plt.figure(figsize=(10, 6))
# Select only numerical columns for correlation calculation
numerical_columns = train_data.select_dtypes(include=['number'])  
correlation = numerical_columns.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()